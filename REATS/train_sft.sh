set -x

MODEL=${MODEL:-./models/Qwen3-1.7B}
DATASET_DIR=${DATASET_DIR:-./datasets}
OUTPUT_BASE=${OUTPUT_BASE:-./output/sft}

DATASET=${DATASET:-""}
if [ -z "${DATASET}" ]; then
    echo "ERROR: DATASET not set. Usage: DATASET=xxx bash train_sft.sh"
    exit 1
fi

EPOCHS=${EPOCHS:-3}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
BATCH_SIZE=${BATCH_SIZE:-2}

cd ./LLaMA-Factory

PORT=$(shuf -i 10000-65000 -n 1)
FORCE_TORCHRUN=1 MASTER_PORT=${PORT} llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path ${MODEL} \
    --dataset_dir ${DATASET_DIR} \
    --dataset ${DATASET} \
    --template qwen3 \
    --finetuning_type full \
    --output_dir ${OUTPUT_BASE}/${DATASET}_sft \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 8192 \
    --preprocessing_num_workers 16 \
    --eval_strategy no \
    --save_only_model \
    --num_train_epochs ${EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --max_samples 100000 \
    --plot_loss \
    --gradient_checkpointing \
    --bf16 True

echo "=== SFT training completed ==="
