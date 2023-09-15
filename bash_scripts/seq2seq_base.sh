#!/bin/bash

#
export CUDA_VISIBLE_DEVICES=1
experiment_name="t5_base_cb_lora"
if [[ -f results/$experiment_name/all_results.json ]]; then
    echo "Experiment $experiment_name is already completed. Continuing to next experiment."
else
    echo "STARTING EXP $experiment_name..."
    python scripts/finetuning_seq2seq.py \
        --output_dir "results/$experiment_name"\
        --seed 0 \
        --model_name_or_path "t5-base" \
        --dataset_name "super_glue" \
        --dataset_config_name "cb" \
        --peft_method "lora" \
        --r 8 \
        --lora_alpha 16 \
        --lora_dropout 0 \
        --target_modules "q, v" \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 16 \
        --gradient_accumulation_steps 1 \
        --use_quantization false \
        --load_in_4bit false \
        --max_source_length 1024 \
        --max_target_length 128 \
        --num_beams 5 \
        --learning_rate 0.0001 \
        --lr_scheduler_type "linear" \
        --lr_scheduler_warmup_percent 0.06 \
        --weight_decay 0 \
        --num_train_epochs 1 \
        --eval_every_steps 1000 \
        --wandb_project "PEFT_comparison" \
        --source_prefix "" \

    echo "FINISHED EXP $experiment_name!!!"
fi

#
experiment_name="t5_base_cnn_dailymail_lora"
if [[ -f results/$experiment_name/all_results.json ]]; then
    echo "Experiment $experiment_name is already completed. Continuing to next experiment."
else
    echo "STARTING EXP $experiment_name..."
    python scripts/finetuning_seq2seq.py \
        --output_dir "results/$experiment_name"\
        --seed 0 \
        --model_name_or_path "t5-base" \
        --dataset_name "cnn_dailymail" \
        --dataset_config_name "3.0.0" \
        --peft_method "lora" \
        --r 8 \
        --lora_alpha 16 \
        --lora_dropout 0 \
        --target_modules "q, v" \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 16 \
        --gradient_accumulation_steps 1 \
        --use_quantization false \
        --load_in_4bit false \
        --max_source_length 1024 \
        --max_target_length 128 \
        --num_beams 5 \
        --learning_rate 0.0001 \
        --lr_scheduler_type "linear" \
        --lr_scheduler_warmup_percent 0.06 \
        --weight_decay 0 \
        --num_train_epochs 1 \
        --eval_every_steps 1000 \
        --wandb_project "PEFT_comparison" \
        --source_prefix "summarize: " \

    echo "FINISHED EXP $experiment_name!!!"
fi


experiment_name="t5_base_cnn_dailymail_ia_3"
if [[ -f "results/$experiment_name/all_results.json" ]]; then
    echo "Experiment $experiment_name is already completed. Continuing to next experiment."
else
    echo "STARTING EXP $experiment_name..."
    python scripts/finetuning_seq2seq.py \
        --output_dir "results/$experiment_name"\
        --seed 0 \
        --model_name_or_path "t5-base" \
        --dataset_name "cnn_dailymail" \
        --dataset_config_name "3.0.0" \
        --peft_method "ia_3" \
        --target_modules "k, v" \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 16 \
        --gradient_accumulation_steps 1 \
        --use_quantization false \
        --load_in_4bit false \
        --max_source_length 1024 \
        --max_target_length 128 \
        --num_beams 5 \
        --learning_rate 0.001 \
        --lr_scheduler_type "linear" \
        --lr_scheduler_warmup_percent 0.06 \
        --weight_decay 0 \
        --num_train_epochs 1 \
        --eval_every_steps 1000 \
        --wandb_project "PEFT_comparison" \
        --source_prefix "summarize: " \

    echo "FINISHED EXP $experiment_name!!!"
fi


experiment_name="t5_base_cnn_dailymail_prompt_tuning"
if [[ -f "results/$experiment_name/all_results.json" ]]; then
    echo "Experiment $experiment_name is already completed. Continuing to next experiment."
else
    echo "STARTING EXP $experiment_name..."
    python scripts/finetuning_seq2seq.py \
        --output_dir "results/$experiment_name"\
        --seed 0 \
        --model_name_or_path "t5-base" \
        --dataset_name "cnn_dailymail" \
        --dataset_config_name "3.0.0" \
        --peft_method "prompt_tuning" \
        --num_virtual_tokens 20 \
        --prompt_tuning_init "text" \
        --prompt_tuning_init_text "summarize the following document " \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 16 \
        --gradient_accumulation_steps 1 \
        --use_quantization false \
        --load_in_4bit false \
        --max_source_length 1024 \
        --max_target_length 128 \
        --num_beams 5 \
        --learning_rate 0.001 \
        --lr_scheduler_type "linear" \
        --lr_scheduler_warmup_percent 0.06 \
        --weight_decay 0 \
        --num_train_epochs 1 \
        --eval_every_steps 1000 \
        --wandb_project "PEFT_comparison" \
        --source_prefix "summarize: " \

    echo "FINISHED EXP $experiment_name!!!"
fi


experiment_name="t5_base_cnn_dailymail_prefix_tuning"
if [[ -f "results/$experiment_name/all_results.json" ]]; then
    echo "Experiment $experiment_name is already completed. Continuing to next experiment."
else
    echo "STARTING EXP $experiment_name..."
    python scripts/finetuning_seq2seq.py \
        --output_dir "results/$experiment_name"\
        --seed 0 \
        --model_name_or_path "t5-base" \
        --dataset_name "cnn_dailymail" \
        --dataset_config_name "3.0.0" \
        --peft_method "prefix_tuning" \
        --num_virtual_tokens 20 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 16 \
        --gradient_accumulation_steps 1 \
        --use_quantization false \
        --load_in_4bit false \
        --max_source_length 1024 \
        --max_target_length 128 \
        --num_beams 5 \
        --learning_rate 0.0001 \
        --lr_scheduler_type "linear" \
        --lr_scheduler_warmup_percent 0.06 \
        --weight_decay 0 \
        --num_train_epochs 1 \
        --eval_every_steps 1000 \
        --wandb_project "PEFT_comparison" \
        --source_prefix "summarize: " \

    echo "FINISHED EXP $experiment_name!!!"
fi
