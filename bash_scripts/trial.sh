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
        --task_type "classification" \
        --model_name_or_path "t5-base" \
        --dataset_name "super_glue" \
        --dataset_config_name "cb" \
        --peft_method "lora" \
        --r 8 \
        --lora_alpha 16 \
        --lora_dropout 0 \
        --target_modules "q, v" \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --max_source_length 128 \
        --max_target_length 128 \
        --num_beams 5 \
        --learning_rate 0.0001 \
        --lr_scheduler_type "linear" \
        --lr_scheduler_warmup_percent 0.06 \
        --weight_decay 0 \
        --num_train_epochs 1 \
        --eval_every_steps 1 \
        --wandb_project "PEFT_comparison" \
        --source_prefix "" \

    echo "FINISHED EXP $experiment_name!!!"
fi
