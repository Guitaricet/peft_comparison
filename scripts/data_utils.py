import re
import random
import nltk
from torch.utils.data import DataLoader
import datasets
from datasets import load_dataset
from transformers import (
    DataCollatorForSeq2Seq,
)

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

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("premise", "hypothesis"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "boolq": ("passage", "question"),
    "cb": ("premise", "hypothesis"),
    "copa": ("premise", "choice1"),
    "multirc": ("paragraph", "question"),
}

label_names_mapping = {
    # taken from, 
    # glue: https://github.com/tensorflow/datasets/blob/1e12611a0be5f4753d271d7eb1dde15eb8f0185c/docs/community_catalog/huggingface/glue.md
    # super_glue: https://github.com/tensorflow/datasets/blob/1e12611a0be5f4753d271d7eb1dde15eb8f0185c/docs/community_catalog/huggingface/super_glue.md
    "cola": [
        "unacceptable",
        "acceptable"
    ],
    "mnli": [
        "entailment",
        "neutral",
        "contradiction"
    ],
    "mrpc": [
        "not_equivalent",
        "equivalent"
    ],
    "qnli": [
        "entailment",
        "not_entailment"
    ],
    "qqp": [
        "not_duplicate",
        "duplicate"
    ],
    "rte": [
        "entailment",
        "not_entailment"
    ],
    "sst2": [
        "negative",
        "positive"
    ],
    "stsb": None,
    "ax": [
        "entailment",
        "neutral",
        "contradiction"
    ],
    "wnli": [
        "not_entailment",
        "entailment"
    ],
    "boolq": [
        "False",
        "True"
    ],
    "cb": [
        "entailment",
        "contradiction",
        "neutral"
    ],
    "copa": [
        "choice1",
        "choice2"
    ],
    "multirc": [
        "False",
        "True"
    ],
    "record": None,
    "wic": [
        "False",
        "True"
    ],
    "wsc": [
        "False",
        "True"
    ],
    "wsc.fixed": [
        "False",
        "True"
    ],
    "axb": [
        "entailment",
        "not_entailment"
    ],
    "axg": [
        "entailment",
        "not_entailment"
    ],
}

def preprocess_glue_one_example(x, benchmark_name, label_names, feature_names=None, id_key='idx'):
    """

    CODE SOURCE: https://github.com/google-research/text-to-text-transfer-transformer/blob/24d9d3b89b129e586bbfe35cffbc5926d88adc5e/t5/data/preprocessors.py#L734C1-L812C12
    
    Convert a dataset from glue to text2text examples.

    This function uses the feature names from the dataset to unpack examples into
    a format amenable for a text2text problem. For example, consider the Quora
    Question Pairs (QQP) benchmark, which would suggest
    benchmark_name="qqp"
    label_names=['not_duplicate', 'duplicate']
    For QQP, a typical example might look like
    {
        "question1": "Why do I easily get bored of my friends?",
        "question2": "Why do I get bored of friends so quickly?",
        "label": 1,
        "idx": 10,
    }

    This example would be transformed to
    {
        "inputs": (
            "qqp question1: Why do I easily get bored of my friends? question2: "
            "Why do I get bored of my friends so quickly?"
        ),
        "targets": "duplicate",
        "idx": 10,
    }

    Args:
    x: an example to process.
    benchmark_name: the name of the GLUE benchmark for this dataset.
    label_names: a list of label names corresponding to class index.
    feature_names: an optional ordered list of feature names. If provided,
        features will be ordered in this way in the output. If not provided, all
        features (except 'idx' and 'label') will be used, sorted by name.
    id_key: str, key for id in the dataset. If not provided, 'idx' will be used.
        if None, no id will be added to the dataset.

    Returns:
    A preprocessed example.
    """
    # If an ordering is not provided, sort feature keys to ensure a consistent
    # order.
    feature_keys = (
        feature_names or sorted(set(x.keys()).difference(['label', 'idx'])))
    # Pack keys (formatted as " key: ") and corresponding text feature
    strs_to_join = []
    for key in feature_keys:
        strs_to_join.append('{}:'.format(key))
        strs_to_join.append(str(x[key]))
    # Add benchmark name at the start
    strs_to_join.insert(0, benchmark_name)

    # label name
    if x['label'] == -1:
        label_name = "<unk>"
    else:
        label_name = ""
        for label_ in label_names:
            label_name += label_names[x["label"]] + " "

    ex = {}
    joined = " ".join(strs_to_join)
    if benchmark_name == 'multirc':
        # Remove HTML markup.
        joined = re.sub(r"<br>", " ", joined)
        joined = re.sub(r"<(/)?b>", "", joined)

        # Store the data index in the returned example (used by eval)
        ex['idx/paragraph'] = x['idx']['paragraph']
        ex['idx/question'] = x['idx']['question']
        ex['idx/answer'] = x['idx']['answer']
    else:
        # Store the data index in the returned example (used by eval)
        if id_key:
            ex['idx'] = x[id_key]

    ex['inputs'] = joined
    ex['targets'] = label_name

    return ex

def preprocess_glue(
    args,
    model,
    tokenizer,
    accelerator,
    logger,
):
    
    #
    raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    label_names = label_names_mapping[args.dataset_config_name]

    #
    train_dataset = []
    for instance in raw_datasets["train"]:
        instance_dict = preprocess_glue_one_example(
            x=instance,
            benchmark_name=args.dataset_config_name, 
            label_names=label_names,
        )
        train_dataset.append(instance_dict)
    train_dataset = datasets.Dataset.from_dict(train_dataset)
    
    eval_dataset = []
    for instance in raw_datasets["eval"]:
        instance_dict = preprocess_glue_one_example(
            x=instance,
            benchmark_name=args.dataset_config_name, 
            label_names=label_names,
        )
        eval_dataset.append(instance_dict)
    eval_dataset = datasets.Dataset.from_dict(eval_dataset)

    #
    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    #
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    return args, model, tokenizer, accelerator, logger, train_dataloader, eval_dataloader

def preprocess_summarization(
        args,
        model,
        tokenizer,
        accelerator,
        logger,
    ):

    #
    raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    prefix = args.source_prefix if args.source_prefix is not None else ""
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

    if args.val_max_target_length is None:
        args.val_max_target_length = args.max_target_length

    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]

        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    with accelerator.main_process_first():
        train_dataset = raw_datasets["train"].map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

        # Temporarily set max_target_length for validation.
        max_target_length = args.val_max_target_length
        eval_dataset = raw_datasets["validation"].map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on val dataset",
        )

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    return args, model, tokenizer, accelerator, logger, train_dataloader, eval_dataloader

def preprocess_data(
    args,
    model,
    tokenizer,
    accelerator,
    logger,
):
    
    if args.task_type == "summarization":
        args, model, tokenizer, accelerator, logger, train_dataloader, eval_dataloader = preprocess_summarization(
            args=args,
            model=model,
            tokenizer=tokenizer,
            accelerator=accelerator,
            logger=logger,
        )
    
    elif args.task_type == "glue":
        args, model, tokenizer, accelerator, logger, train_dataloader, eval_dataloader = preprocess_glue(
            args=args,
            model=model,
            tokenizer=tokenizer,
            accelerator=accelerator,
            logger=logger,
        )


    
    
    return args, model, tokenizer, accelerator, logger, train_dataloader, eval_dataloader

"""Following functions are postprocessing functions.

Functions which process model output bytes to make them ready for eval.

Note: postprocessors must either accept an `example` and `is_target` kwargs
or include `**unused_kwargs` in their signature. The `example` will be the
full example.

These functions should assume input strings to be unicode, but that strings
in the `example` dict will be in bytes.
"""

def string_to_float(string, default=-1., **unused_kwargs):
    """Converts string to float, using default when conversion not possible."""
    try:
        return float(string)
    except ValueError:
        return default


def lower_text(string, **unused_kwargs):
    """Lowercases text."""
    return string.lower()


def string_label_to_class_id(string_label,
                             label_classes,
                             default=-1,
                             **unused_kwargs):
    """Returns index of string_label in label_classes or default if not found."""
    if string_label in label_classes:
        return label_classes.index(string_label)
    else:
        return default


def multirc(string_label, example=None, is_target=False):
    """Returns dict containing the class with the question index for grouping."""
    res = {
        "value":
            string_label_to_class_id(
                string_label, example=example, label_classes=("False", "True"))
    }
    # Add the group, if present, since the model outputs will not have it.
    if is_target:
        res["group"] = example["idx/question"]
    return res

def to_unicode(input_string):
    """Converts any string-like python input types to unicode.

    Args:
    input_string: A string-like python input type.

    Returns:
    A unicode string.
    """

    if isinstance(input_string, bytes):
        return input_string.decode("utf-8")
    elif isinstance(input_string, str):
        return input_string
    else:
        raise TypeError("Expected string type but got %s" % type(input_string))

def record(answer, example=None, is_target=False):
    """Returns dict with answer, or all answers + grouping key for a target."""
    if is_target:
        return {
            "value": [to_unicode(a) for a in example["answers"]],
            # Add the group since the model output will not have it.
            "group": (example["idx/passage"], example["idx/query"])
        }
    return {"value": answer}


def qa(answer, example=None, is_target=False):
  """Returns answer, or all answers if the full example is provided."""
  if is_target:
    return [to_unicode(a) for a in example["answers"]]
  return answer


def span_qa(answer, example=None, is_target=False):
  """Returns answer, or a dict with answers and context if the example is provided."""

  if is_target:
    return {
        "answers": [to_unicode(a) for a in example["answers"]],
        "context": to_unicode(example["context"])
    }

  return answer


def wsc_simple(prediction, example=None, is_target=False):
  """Sees whether we predicted the referent or not."""
  if is_target:
    return example["label"]

  determiners = {
      "a", "an", "few", "her", "his", "each", "every", "many", "much", "my",
      "our", "some", "that", "the", "their", "these", "this", "those", "which",
      "whose", "your"
  }

  def clean(s):
    """Ignore capitalization and determiners."""
    s = to_unicode(s).strip().lower()
    return " ".join([w for w in s.split(" ") if w not in determiners])

  prediction = clean(prediction)
  if not prediction:
    # We don't want an empty prediction to accidentally return 0 and spuriously
    # match the label.
    return -1

  # We aren't using the label but rather using the extracted referent so that we
  # can see if the prediction is equivalent to the referent.
  referent = clean(example["targets_pretokenized"])

  if ("'" in prediction) != ("'" in referent):
    # Make sure we don't mark cases where the prediction is "Bob" and the
    # referent is "Bob's hat" as predicting the referent.
    predicted_referent = False
  else:
    prediction_words = set(prediction.split(" "))
    referent_words = set(referent.split(" "))

    # Handle cases where the prediction is "fuzzy bunny" and the referent is
    # "bunny".
    predicted_referent = prediction_words.issubset(
        referent_words) or referent_words.issubset(prediction_words)

  return int(predicted_referent)