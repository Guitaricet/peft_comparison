#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on summarization.
"""
# You can also adapt this script on your own summarization task. Pointers for this are left as comments.

import os
import argparse
import json
import math
import os
import random
from pprint import pformat

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import nltk
import datasets
import evaluate

import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
    set_seed,
)

from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset

import wandb
from tqdm.auto import tqdm, trange
from loguru import logger

from peft_comparison.collation import DataCollatorForSeq2SeqWithMetadata

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
datasets.utils.logging.set_verbosity_error()
transformers.utils.logging.set_verbosity_error()

nltk.data.find("tokenizers/punkt")

str2bool = lambda x: x.lower() in ["true", "1"]

summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")

    # Dataset Configuration
    parser.add_argument("--dataset_name", type=str, default="cnn_dailymail", help="The name of the dataset to use via the datasets library.")
    parser.add_argument("--dataset_config_name", type=str, default="3.0.0", help="The configuration name of the dataset to use via the datasets library.")
    parser.add_argument("--max_source_length", type=int, default=1024, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--source_prefix", type=str, default="", help="A prefix to add before every source text, useful for T5 models.")
    parser.add_argument("--preprocessing_num_workers", type=int, default=8, help="The number of processes to use for the preprocessing.")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--subsample_data", type=int, default=None, help="If passed, will subsample the dataset to this many examples. (debug only)")

    # Target Text Configuration
    parser.add_argument("--max_target_length", type=int, default=128, help="The maximum total sequence length for target text after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded during evaluate and predict.")
    parser.add_argument("--val_max_target_length", type=int, default=None, help="The maximum total sequence length for validation target text after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded. Will default to max_target_length. This argument is also used to override the max_length param of model.generate, which is used during evaluate and predict.")
    parser.add_argument("--num_beams", type=int, default=None, help="Number of beams to use for evaluation. This argument will be passed to model.generate, which is used during evaluate and predict.")
    parser.add_argument("--pad_to_max_length", action="store_true", help="If passed, pad all samples to max_length. Otherwise, dynamic padding is used.")
    parser.add_argument("--max_eval_steps_durig_validation", type=int, default=100, help="Maximum number of evaluation steps to perform during validation. Useful to save time when you don't need to run the full validation set.")

    # Model Configuration
    parser.add_argument("--model_name_or_path", type=str, default="t5-base", help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--text_column", type=str, default=None, help="The name of the column in the datasets containing the full texts for summarization.")
    parser.add_argument("--summary_column", type=str, default=None, help="The name of the column in the datasets containing the summaries for summarization.")

    # Batch Configuration
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per device for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2, help="Batch size per device for the evaluation dataloader.")
    parser.add_argument("--total_batch_size", type=int, default=32, help="Total batch size per_device_batch_size * num_devices * gradient_accumulation")

    # Training Configuration
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate after the potential warmup period to use.")
    parser.add_argument("--lr_scheduler_warmup_percent", type=float, default=0.06, help="Percentage of steps for the warmup in the lr scheduler.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--eval_every_steps", type=int, default=None, help="Evaluate model after these many steps.")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear", help="The scheduler type to use, choices: linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup")
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")

    # Output and Tracking
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument("--checkpointing_steps", type=str, default=None, help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="If the training should continue from a checkpoint folder.")

    # PEFT Configuration
    parser.add_argument("--adapter_config_string", default=None, type=str, help="The adapter config string to use for adapter-transformers, ignored if --peft_library is not adapter-transformers")

    # Memory Management
    parser.add_argument("--load_in_8bit", action="store_true", help="Enable 8bit quantization.")
    parser.add_argument("--torch_dtype", type=torch.dtype, default=torch.bfloat16, help="This sets the dtype of the remaining non quantized layers. 'bitsandbytes' library suggests to set the value to 'torch.float16' for 8 bit model and use the same dtype as the compute dtype for 4 bit model")

    # Weight and Biases Configuration
    parser.add_argument("--wandb_project", type=str, default="PEFT_comparison_v2", help="Name to be given to Weight and Biases logging repository")
    parser.add_argument("--tags", type=str, default=None, help="Tags to be given to individual runs in WandB repository, e.g. 'trial, t5-base, classification'")
    parser.add_argument("--wandb_name", type=str, default=None, help="Display name for the run")

    # Misc
    parser.add_argument("--verbocity", type=int, default=1, help="Verbocity of the logger (1 or 2 for now)")
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None:
        raise ValueError("Need dataset name")

    if args.tags is not None:
        args.tags = args.tags.split(",")

    if args.val_max_target_length is None:
        args.val_max_target_length = args.max_target_length

    return args


def get_model(args):
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=args.torch_dtype,
        device_map={"": torch.cuda.current_device()},
        load_in_8bit=args.load_in_8bit,
    )

    for param in model.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        set_pad_to = tokenizer.eos_token
        tokenizer.add_special_tokens({'pad_token': set_pad_to})
        model.config.pad_token_id = model.config.eos_token_id

    model.add_adapter("adapter", config=args.adapter_config_string, set_active=True)
    model.train()
    model.train_adapter("adapter")

    if not args.load_in_8bit:
        model = model.to(dtype=args.torch_dtype)

    if args.load_in_8bit:
        # adapter is not yet on the device
        for name, module in model.named_modules():
            if "adapter" in name:
                module.to(device=torch.cuda.current_device(), dtype=args.torch_dtype)

    if args.verbocity > 1:
        logger.info(model)

    dtype_counts = {}
    for p in model.parameters():
        dtype_counts[p.dtype] = dtype_counts.get(p.dtype, 0) + p.numel()

    total_parameters = sum(dtype_counts.values())
    dtype_info = [f"{dtype}: {count} ({count / total_parameters * 100:.2f}%)" for dtype, count in dtype_counts.items()]
    logger.info("Model dtypes: ", " | ".join(dtype_info))

    return model, tokenizer


def postprocess_text_for_eval(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


@torch.no_grad()
def evaluate_model(
    model,
    *,
    metric,
    tokenizer,
    dataloader,
    accelerator,
    max_length=None,
    num_beams=None,
    max_iters=None,
):
    model.eval()
    pbar = tqdm(
        dataloader,
        desc="Evaluating",
        disable=not accelerator.is_local_main_process,
        total=max_iters or len(dataloader),
        ncols=80,
    )
    for eval_step, batch in enumerate(dataloader):
        pbar.update()
        if max_iters is not None and eval_step > max_iters:
            logger.info(f"{max_iters} evaluation steps reached. Stopping evaluation.")
            break

        with torch.no_grad():
            unwrapped_model = accelerator.unwrap_model(model)
            generated_tokens = unwrapped_model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=max_length,
                num_beams=num_beams,
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

            generated_tokens = generated_tokens.cpu().numpy()
            labels = labels.cpu().numpy()

            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]

            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            labels_str = [_l["targets"] for _l in batch["metadata"]]
            if len(decoded_preds) != len(labels_str):
                print(f"Input ids shape: {batch['input_ids'].shape}"),
                print(f"{len(decoded_preds)} != {len(labels_str)}")

            if eval_step == 0:
                logger.info(f"Example of predictions: {decoded_preds[0]}")
                logger.info(f"Example of labels: {labels_str[0]}")

            decoded_preds, labels_str = postprocess_text_for_eval(decoded_preds, labels_str)
            metric.add_batch(predictions=decoded_preds, references=labels_str)

    result = metric.compute(use_stemmer=True)
    result = {f"eval/{k}": round(v * 100, 4) for k, v in result.items()}

    model.train()
    return result


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator = Accelerator(project_dir=args.output_dir, log_with="wandb")
    if not accelerator.is_main_process:
        logger.remove()

    if args.total_batch_size is not None:
        if args.gradient_accumulation_steps is not None:
            logger.warning("`--total_batch_size` overrides --gradient_accumulation_steps")
        if args.total_batch_size % (accelerator.num_processes * args.per_device_train_batch_size) != 0:
            raise ValueError(f"`--total_batch_size` ({args.total_batch_size}) is not divisible by "
                             f"num_processes * per_device_train_batch_size ({accelerator.num_processes} * {args.per_device_train_batch_size})")
        args.gradient_accumulation_steps = args.total_batch_size // (args.per_device_train_batch_size * accelerator.num_processes)
        logger.info(f"Setting gradient accumulation steps to {args.gradient_accumulation_steps}.")
    else:
        args.total_batch_size = args.gradient_accumulation_steps * args.per_device_train_batch_size * accelerator.num_processes

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    accelerator.init_trackers(args.wandb_project, init_kwargs={"wandb": {"tags": args.tags}})
    if accelerator.is_main_process:
        wandb.save(os.path.abspath(__file__), policy="now") # save current script

    logger.info("*" * 40)
    for k, v in vars(args).items():
        logger.info(f"{k:30}: {v}")
    logger.info("*" * 40)

    raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    if args.subsample_data is not None:
        logger.warning(f"Subsampling the dataset to {args.subsample_data} first examples.")
        # remember that it's dataset dict
        def get_subsample_data(subset_name):
            if subset_name == "train": return args.subsample_data
            return max(100, args.subsample_data // 10)
        raw_datasets = {k: v.select(range(get_subsample_data(k))) for k, v in raw_datasets.items()}

    # Load pretrained model and tokenizer
    model, tokenizer = get_model(args)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if "t5" in args.model_name_or_path and args.source_prefix is None:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(args.dataset_name, None)
    if args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = args.text_column
        if text_column not in column_names:
            raise ValueError(f"--text_column' value '{args.text_column}' needs to be one of: {', '.join(column_names)}")
    if args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = args.summary_column
        if summary_column not in column_names:
            raise ValueError(f"--summary_column' value '{args.summary_column}' needs to be one of: {', '.join(column_names)}")

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples, is_eval=False):
        inputs = examples[text_column]
        targets = examples[summary_column]
        inputs = [args.source_prefix + inp for inp in inputs]

        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=args.max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        if is_eval:
            model_inputs["metadata"] = [{"targets": t} for t in targets]
        return model_inputs

    with accelerator.main_process_first():
        eval_dataset = raw_datasets["validation"].map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            desc="Running tokenizer on val dataset  ",
            fn_kwargs={"is_eval": True},
        )
        train_dataset = raw_datasets["train"].map(
            preprocess_function,
            batched=True,
            batch_size=min(5000, len(raw_datasets["train"]) // args.preprocessing_num_workers),
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            desc="Running tokenizer on train dataset",
        )

    # Log a few random samples from the training set:
    if args.verbocity > 1:
        for index in random.sample(range(len(train_dataset)), 1):
            logger.info(f"Sample {index} of the training set:")
            for k, v in train_dataset[index].items():
                if hasattr(v, "shape"):
                    logger.info(f"  {k}.shape: {v.shape}")
                logger.info(f"  {k}: {v}")

    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2SeqWithMetadata(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8,
    )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm", "layer_norm"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total model parameters: {total_parameters:,}")
    logger.info(f"Trainable parameters  : {trainable_parameters:,} ({trainable_parameters / total_parameters * 100:.4f}%)")
    if args.verbocity > 1:
        logger.info("Trainable model parameters")
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f"{name}: {param.numel():,}")
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(args.max_train_steps * args.lr_scheduler_warmup_percent),
        num_training_steps=args.max_train_steps,
    )

    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    experiment_config = vars(args)
    experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
    if accelerator.is_main_process:
        wandb.config.update(experiment_config)

    # Metric
    metric = evaluate.load("rouge")

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Total warmup steps for LR = {int(args.max_train_steps * args.lr_scheduler_warmup_percent)}")

    # Only show the progress bar once on each machine.
    progress_bar = trange(
        args.max_train_steps,
        desc="Training (total steps)",
        disable=not accelerator.is_local_main_process,
        total=args.max_train_steps,
        ncols=80,
        leave=True,
    )

    update_step = 0
    starting_epoch = 0
    global_step = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        logger.info(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            update_step = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
            update_step = resume_step // args.gradient_accumulation_stepp

        progress_bar.update(update_step)

    active_dataloader = train_dataloader
    if args.resume_from_checkpoint and resume_step is not None:
        active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()

        for batch in active_dataloader:
            global_step += 1

            outputs = model(**batch)
            loss = outputs.loss

            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if global_step % args.gradient_accumulation_steps == 0 or global_step == len(train_dataloader) - 1:
                progress_bar.update()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                update_step += 1

            if update_step >= args.max_train_steps:
                logger.info("Max number of steps reached. Stopping training")
                break

            accelerator.log(
                {
                    "train/loss": loss,
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                    "update_step": update_step,
                    "global_steps": global_step,
                },
                step=update_step,
            )

            if (update_step + 1) % args.eval_every_steps == 0:
                logger.info(f"Evaluating model at step {update_step}")
                result = evaluate_model(
                    model=model,
                    metric=metric,
                    tokenizer=tokenizer,
                    dataloader=eval_dataloader,
                    accelerator=accelerator,
                    max_length=args.val_max_target_length,
                    num_beams=1,
                    max_iters=args.max_eval_steps_durig_validation,
                )
                logger.info(pformat(result))
                accelerator.log(result, step=update_step)

    # final evaluation
    if update_step % args.eval_every_steps != 0:
        logger.info(f"Final evaluation (step={update_step})")
        result = evaluate_model(
            model=model,
            metric=metric,
            tokenizer=tokenizer,
            dataloader=eval_dataloader,
            accelerator=accelerator,
            max_length=args.val_max_target_length,
            num_beams=args.num_beams,
            max_iters=None,
        )
        logger.info(pformat(result))
        accelerator.log(result, step=update_step)

    # save results and all arguments
    all_results = result.copy()
    all_results["args"] = vars(args)
    for k, v in all_results["args"].items():
        if not isinstance(v, (float, int, bool, str, list)):
            all_results["args"][k] = str(v)

    with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=4)

    logger.info("Script successfully finished!")


if __name__ == "__main__":
    main()
