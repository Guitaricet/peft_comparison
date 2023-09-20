#!/bin/bash

#
export CUDA_VISIBLE_DEVICES=1

#
experiment_name="t5_base_cb_lora"
if [[ -f results/$experiment_name/all_results.json ]]; then
    echo "Experiment $experiment_name is already completed. Continuing to next experiment."
else
    echo "STARTING EXP $experiment_name..."
    python scripts/finetuning_seq2seq.py \
        --output_dir "results/$experiment_name"\
        --seed 0 \
        --task_type "classification" \
        --model_name_or_path "t5-base" \
        --dataset_name "super_glue" \
        --dataset_config_name "cb" \
        --peft_method "lora" \
        --r 8 \
        --lora_alpha 16 \
        --lora_dropout 0 \
        --target_modules "q, v" \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --total_batch_size 16 \
        --gradient_accumulation_steps 1 \
        --max_source_length 128 \
        --max_target_length 8 \
        --num_beams 5 \
        --learning_rate 0.0001 \
        --lr_scheduler_type "linear" \
        --lr_scheduler_warmup_percent 0.06 \
        --weight_decay 0 \
        --num_train_epochs 20 \
        --eval_every_steps 10000 \
        --wandb_project "PEFT_comparison" \
        --wandb_tags "classification, seq2seq" \
        --source_prefix "" \

    echo "FINISHED EXP $experiment_name!!!"
fi


#
experiment_name="t5_base_copa_lora"
if [[ -f results/$experiment_name/all_results.json ]]; then
    echo "Experiment $experiment_name is already completed. Continuing to next experiment."
else
    echo "STARTING EXP $experiment_name..."
    python scripts/finetuning_seq2seq.py \
        --output_dir "results/$experiment_name"\
        --seed 0 \
        --task_type "classification" \
        --model_name_or_path "t5-base" \
        --dataset_name "super_glue" \
        --dataset_config_name "copa" \
        --peft_method "lora" \
        --r 8 \
        --lora_alpha 16 \
        --lora_dropout 0 \
        --target_modules "q, v" \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --total_batch_size 16 \
        --gradient_accumulation_steps 1 \
        --max_source_length 128 \
        --max_target_length 8 \
        --num_beams 5 \
        --learning_rate 0.0001 \
        --lr_scheduler_type "linear" \
        --lr_scheduler_warmup_percent 0.06 \
        --weight_decay 0 \
        --num_train_epochs 20 \
        --eval_every_steps 10000 \
        --wandb_project "PEFT_comparison" \
        --wandb_tags "classification, seq2seq" \
        --source_prefix "" \

    echo "FINISHED EXP $experiment_name!!!"
fi

experiment_name="t5_base_rte_lora"
if [[ -f results/$experiment_name/all_results.json ]]; then
    echo "Experiment $experiment_name is already completed. Continuing to next experiment."
else
    echo "STARTING EXP $experiment_name..."
    python scripts/finetuning_seq2seq.py \
        --output_dir "results/$experiment_name" \
        --seed 0 \
        --task_type "classification" \
        --model_name_or_path "t5-base" \
        --dataset_name "super_glue" \
        --dataset_config_name "rte" \
        --peft_method "lora" \
        --r 8 \
        --lora_alpha 16 \
        --lora_dropout 0 \
        --target_modules "q, v" \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --total_batch_size 16 \
        --gradient_accumulation_steps 1 \
        --max_source_length 128 \
        --max_target_length 8 \
        --num_beams 5 \
        --learning_rate 0.0001 \
        --lr_scheduler_type "linear" \
        --lr_scheduler_warmup_percent 0.06 \
        --weight_decay 0 \
        --num_train_epochs 20 \
        --eval_every_steps 10000 \
        --wandb_project "PEFT_comparison" \
        --wandb_tags "classification, seq2seq" \
        --source_prefix "" \

    echo "FINISHED EXP $experiment_name!!!"
fi

experiment_name="t5_base_boolq_lora"
if [[ -f results/$experiment_name/all_results.json ]]; then
    echo "Experiment $experiment_name is already completed. Continuing to next experiment."
else
    echo "STARTING EXP $experiment_name..."
    python scripts/finetuning_seq2seq.py \
        --output_dir "results/$experiment_name"\
        --seed 0 \
        --task_type "classification" \
        --model_name_or_path "t5-base" \
        --dataset_name "super_glue" \
        --dataset_config_name "boolq" \
        --peft_method "lora" \
        --r 8 \
        --lora_alpha 16 \
        --lora_dropout 0 \
        --target_modules "q, v" \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --total_batch_size 16 \
        --gradient_accumulation_steps 1 \
        --max_source_length 256 \
        --max_target_length 8 \
        --num_beams 5 \
        --learning_rate 0.0001 \
        --lr_scheduler_type "linear" \
        --lr_scheduler_warmup_percent 0.06 \
        --weight_decay 0 \
        --num_train_epochs 20 \
        --eval_every_steps 10000 \
        --wandb_project "PEFT_comparison" \
        --wandb_tags "classification, seq2seq" \
        --source_prefix "" \

    echo "FINISHED EXP $experiment_name!!!"
fi



#
echo "Starting IA_3 experiments"

#
experiment_name="t5_base_cb_ia_3"
if [[ -f results/$experiment_name/all_results.json ]]; then
    echo "Experiment $experiment_name is already completed. Continuing to next experiment."
else
    echo "STARTING EXP $experiment_name..."
    python scripts/finetuning_seq2seq.py \
        --output_dir "results/$experiment_name"\
        --seed 0 \
        --task_type "classification" \
        --model_name_or_path "t5-base" \
        --dataset_name "super_glue" \
        --dataset_config_name "cb" \
        --peft_method "ia_3" \
        --target_modules "k, v" \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --total_batch_size 16 \
        --gradient_accumulation_steps 1 \
        --max_source_length 128 \
        --max_target_length 8 \
        --num_beams 5 \
        --learning_rate 0.001 \
        --lr_scheduler_type "linear" \
        --lr_scheduler_warmup_percent 0.06 \
        --weight_decay 0 \
        --num_train_epochs 20 \
        --eval_every_steps 10000 \
        --wandb_project "PEFT_comparison" \
        --wandb_tags "classification, seq2seq" \
        --source_prefix "" \

    echo "FINISHED EXP $experiment_name!!!"
fi


#
experiment_name="t5_base_copa_ia_3"
if [[ -f results/$experiment_name/all_results.json ]]; then
    echo "Experiment $experiment_name is already completed. Continuing to next experiment."
else
    echo "STARTING EXP $experiment_name..."
    python scripts/finetuning_seq2seq.py \
        --output_dir "results/$experiment_name"\
        --seed 0 \
        --task_type "classification" \
        --model_name_or_path "t5-base" \
        --dataset_name "super_glue" \
        --dataset_config_name "copa" \
        --peft_method "ia_3" \
        --target_modules "k, v" \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --total_batch_size 16 \
        --gradient_accumulation_steps 1 \
        --max_source_length 128 \
        --max_target_length 8 \
        --num_beams 5 \
        --learning_rate 0.001 \
        --lr_scheduler_type "linear" \
        --lr_scheduler_warmup_percent 0.06 \
        --weight_decay 0 \
        --num_train_epochs 20 \
        --eval_every_steps 10000 \
        --wandb_project "PEFT_comparison" \
        --wandb_tags "classification, seq2seq" \
        --source_prefix "" \

    echo "FINISHED EXP $experiment_name!!!"
fi

experiment_name="t5_base_rte_ia_3"
if [[ -f results/$experiment_name/all_results.json ]]; then
    echo "Experiment $experiment_name is already completed. Continuing to next experiment."
else
    echo "STARTING EXP $experiment_name..."
    python scripts/finetuning_seq2seq.py \
        --output_dir "results/$experiment_name"\
        --seed 0 \
        --task_type "classification" \
        --model_name_or_path "t5-base" \
        --dataset_name "super_glue" \
        --dataset_config_name "rte" \
        --peft_method "ia_3" \
        --target_modules "k, v" \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --total_batch_size 16 \
        --gradient_accumulation_steps 1 \
        --max_source_length 128 \
        --max_target_length 8 \
        --num_beams 5 \
        --learning_rate 0.001 \
        --lr_scheduler_type "linear" \
        --lr_scheduler_warmup_percent 0.06 \
        --weight_decay 0 \
        --num_train_epochs 20 \
        --eval_every_steps 10000 \
        --wandb_project "PEFT_comparison" \
        --wandb_tags "classification, seq2seq" \
        --source_prefix "" \

    echo "FINISHED EXP $experiment_name!!!"
fi

experiment_name="t5_base_boolq_ia_3"
if [[ -f results/$experiment_name/all_results.json ]]; then
    echo "Experiment $experiment_name is already completed. Continuing to next experiment."
else
    echo "STARTING EXP $experiment_name..."
    python scripts/finetuning_seq2seq.py \
        --output_dir "results/$experiment_name"\
        --seed 0 \
        --task_type "classification" \
        --model_name_or_path "t5-base" \
        --dataset_name "super_glue" \
        --dataset_config_name "boolq" \
        --peft_method "ia_3" \
        --target_modules "k, v" \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --total_batch_size 16 \
        --gradient_accumulation_steps 1 \
        --max_source_length 256 \
        --max_target_length 8 \
        --num_beams 5 \
        --learning_rate 0.001 \
        --lr_scheduler_type "linear" \
        --lr_scheduler_warmup_percent 0.06 \
        --weight_decay 0 \
        --num_train_epochs 20 \
        --eval_every_steps 10000 \
        --wandb_project "PEFT_comparison" \
        --wandb_tags "classification, seq2seq" \
        --source_prefix "" \

    echo "FINISHED EXP $experiment_name!!!"
fi

#
echo "Starting Prompt-Tuning experiments"

experiment_name="t5_base_cb_prompt_tuning"
if [[ -f "results/$experiment_name/all_results.json" ]]; then
    echo "Experiment $experiment_name is already completed. Continuing to next experiment."
else
    echo "STARTING EXP $experiment_name..."
    python scripts/finetuning_seq2seq.py \
        --output_dir "results/$experiment_name"\
        --seed 0 \
        --task_type "classification" \
        --model_name_or_path "t5-base" \
        --dataset_name "super_glue" \
        --dataset_config_name "cb" \
        --peft_method "prompt_tuning" \
        --num_virtual_tokens 20 \
        --prompt_tuning_init "text" \
        --prompt_tuning_init_text "this is an entailment classification task. " \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --total_batch_size 16 \
        --max_source_length 128 \
        --max_target_length 8 \
        --num_beams 5 \
        --learning_rate 0.001 \
        --lr_scheduler_type "linear" \
        --lr_scheduler_warmup_percent 0.06 \
        --weight_decay 0 \
        --num_train_epochs 20 \
        --eval_every_steps 10000 \
        --wandb_project "PEFT_comparison" \
        --wandb_tags "classification, seq2seq" \
        --source_prefix "" \

    echo "FINISHED EXP $experiment_name!!!"
fi

experiment_name="t5_base_copa_prompt_tuning"
if [[ -f "results/$experiment_name/all_results.json" ]]; then
    echo "Experiment $experiment_name is already completed. Continuing to next experiment."
else
    echo "STARTING EXP $experiment_name..."
    python scripts/finetuning_seq2seq.py \
        --output_dir "results/$experiment_name"\
        --seed 0 \
        --task_type "classification" \
        --model_name_or_path "t5-base" \
        --dataset_name "super_glue" \
        --dataset_config_name "copa" \
        --peft_method "prompt_tuning" \
        --num_virtual_tokens 20 \
        --prompt_tuning_init "text" \
        --prompt_tuning_init_text "this is commonsense reasoning classification task. " \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --total_batch_size 16 \
        --max_source_length 128 \
        --max_target_length 8 \
        --num_beams 5 \
        --learning_rate 0.001 \
        --lr_scheduler_type "linear" \
        --lr_scheduler_warmup_percent 0.06 \
        --weight_decay 0 \
        --num_train_epochs 20 \
        --eval_every_steps 10000 \
        --wandb_project "PEFT_comparison" \
        --wandb_tags "classification, seq2seq" \
        --source_prefix "" \

    echo "FINISHED EXP $experiment_name!!!"
fi

experiment_name="t5_base_rte_prompt_tuning"
if [[ -f "results/$experiment_name/all_results.json" ]]; then
    echo "Experiment $experiment_name is already completed. Continuing to next experiment."
else
    echo "STARTING EXP $experiment_name..."
    python scripts/finetuning_seq2seq.py \
        --output_dir "results/$experiment_name"\
        --seed 0 \
        --task_type "classification" \
        --model_name_or_path "t5-base" \
        --dataset_name "super_glue" \
        --dataset_config_name "rte" \
        --peft_method "prompt_tuning" \
        --num_virtual_tokens 20 \
        --prompt_tuning_init "text" \
        --prompt_tuning_init_text "this is an entailment classification task. " \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --total_batch_size 16 \
        --max_source_length 128 \
        --max_target_length 8 \
        --num_beams 5 \
        --learning_rate 0.001 \
        --lr_scheduler_type "linear" \
        --lr_scheduler_warmup_percent 0.06 \
        --weight_decay 0 \
        --num_train_epochs 20 \
        --eval_every_steps 10000 \
        --wandb_project "PEFT_comparison" \
        --wandb_tags "classification, seq2seq" \
        --source_prefix "" \

    echo "FINISHED EXP $experiment_name!!!"
fi

experiment_name="t5_base_boolq_prompt_tuning"
if [[ -f "results/$experiment_name/all_results.json" ]]; then
    echo "Experiment $experiment_name is already completed. Continuing to next experiment."
else
    echo "STARTING EXP $experiment_name..."
    python scripts/finetuning_seq2seq.py \
        --output_dir "results/$experiment_name"\
        --seed 0 \
        --task_type "classification" \
        --model_name_or_path "t5-base" \
        --dataset_name "super_glue" \
        --dataset_config_name "boolq" \
        --peft_method "prompt_tuning" \
        --num_virtual_tokens 20 \
        --prompt_tuning_init "text" \
        --prompt_tuning_init_text "this is a question answering classification task" \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --total_batch_size 16 \
        --max_source_length 256 \
        --max_target_length 8 \
        --num_beams 5 \
        --learning_rate 0.001 \
        --lr_scheduler_type "linear" \
        --lr_scheduler_warmup_percent 0.06 \
        --weight_decay 0 \
        --num_train_epochs 20 \
        --eval_every_steps 10000 \
        --wandb_project "PEFT_comparison" \
        --wandb_tags "classification, seq2seq" \
        --source_prefix "" \

    echo "FINISHED EXP $experiment_name!!!"
fi

#
echo "Starting Prefix-Tuning experiments"


experiment_name="t5_base_cb_prefix_tuning"
if [[ -f "results/$experiment_name/all_results.json" ]]; then
    echo "Experiment $experiment_name is already completed. Continuing to next experiment."
else
    echo "STARTING EXP $experiment_name..."
    python scripts/finetuning_seq2seq.py \
        --output_dir "results/$experiment_name"\
        --seed 0 \
        --task_type "classification" \
        --model_name_or_path "t5-base" \
        --dataset_name "super_glue" \
        --dataset_config_name "cb" \
        --peft_method "prefix_tuning" \
        --num_virtual_tokens 20 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --total_batch_size 16 \
        --max_source_length 128 \
        --max_target_length 8 \
        --num_beams 5 \
        --learning_rate 0.0001 \
        --lr_scheduler_type "linear" \
        --lr_scheduler_warmup_percent 0.06 \
        --weight_decay 0 \
        --num_train_epochs 20 \
        --eval_every_steps 10000 \
        --wandb_project "PEFT_comparison" \
        --wandb_tags "classification, seq2seq" \
        --source_prefix "" \

    echo "FINISHED EXP $experiment_name!!!"
fi

experiment_name="t5_base_copa_prefix_tuning"
if [[ -f "results/$experiment_name/all_results.json" ]]; then
    echo "Experiment $experiment_name is already completed. Continuing to next experiment."
else
    echo "STARTING EXP $experiment_name..."
    python scripts/finetuning_seq2seq.py \
        --output_dir "results/$experiment_name"\
        --seed 0 \
        --task_type "classification" \
        --model_name_or_path "t5-base" \
        --dataset_name "super_glue" \
        --dataset_config_name "copa" \
        --peft_method "prefix_tuning" \
        --num_virtual_tokens 20 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --total_batch_size 16 \
        --max_source_length 128 \
        --max_target_length 8 \
        --num_beams 5 \
        --learning_rate 0.0001 \
        --lr_scheduler_type "linear" \
        --lr_scheduler_warmup_percent 0.06 \
        --weight_decay 0 \
        --num_train_epochs 20 \
        --eval_every_steps 10000 \
        --wandb_project "PEFT_comparison" \
        --wandb_tags "classification, seq2seq" \
        --source_prefix "" \

    echo "FINISHED EXP $experiment_name!!!"
fi

experiment_name="t5_base_rte_prefix_tuning"
if [[ -f "results/$experiment_name/all_results.json" ]]; then
    echo "Experiment $experiment_name is already completed. Continuing to next experiment."
else
    echo "STARTING EXP $experiment_name..."
    python scripts/finetuning_seq2seq.py \
        --output_dir "results/$experiment_name"\
        --seed 0 \
        --task_type "classification" \
        --model_name_or_path "t5-base" \
        --dataset_name "super_glue" \
        --dataset_config_name "rte" \
        --peft_method "prefix_tuning" \
        --num_virtual_tokens 20 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --total_batch_size 16 \
        --max_source_length 128 \
        --max_target_length 8 \
        --num_beams 5 \
        --learning_rate 0.0001 \
        --lr_scheduler_type "linear" \
        --lr_scheduler_warmup_percent 0.06 \
        --weight_decay 0 \
        --num_train_epochs 20 \
        --eval_every_steps 10000 \
        --wandb_project "PEFT_comparison" \
        --wandb_tags "classification, seq2seq" \
        --source_prefix "" \

    echo "FINISHED EXP $experiment_name!!!"
fi

experiment_name="t5_base_boolq_prefix_tuning"
if [[ -f "results/$experiment_name/all_results.json" ]]; then
    echo "Experiment $experiment_name is already completed. Continuing to next experiment."
else
    echo "STARTING EXP $experiment_name..."
    python scripts/finetuning_seq2seq.py \
        --output_dir "results/$experiment_name"\
        --seed 0 \
        --task_type "classification" \
        --model_name_or_path "t5-base" \
        --dataset_name "super_glue" \
        --dataset_config_name "boolq" \
        --peft_method "prefix_tuning" \
        --num_virtual_tokens 20 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --total_batch_size 16 \
        --max_source_length 256 \
        --max_target_length 8 \
        --num_beams 5 \
        --learning_rate 0.0001 \
        --lr_scheduler_type "linear" \
        --lr_scheduler_warmup_percent 0.06 \
        --weight_decay 0 \
        --num_train_epochs 20 \
        --eval_every_steps 10000 \
        --wandb_project "PEFT_comparison" \
        --wandb_tags "classification, seq2seq" \
        --source_prefix "" \

    echo "FINISHED EXP $experiment_name!!!"
fi
