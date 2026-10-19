
set -ex

SWIFT_DIR=./ms-swift
GRPO_OUTPUT_BASE=./output/grpo
REWARD_PLUGIN=./grpo_reward.py

MODEL=${MODEL:-""}
LEARNING_RATE=${LEARNING_RATE:-5e-6}

if [ -z "${MODEL}" ]; then
    echo "ERROR: MODEL not set. Usage: MODEL=/path/to/sft_checkpoint bash train_grpo.sh"
    exit 1
fi

# GRPO dataset path
GRPO_DATA=${GRPO_DATA:-""}

if [ -z "${GRPO_DATA}" ]; then
    echo "ERROR: GRPO_DATA not set. Usage: GRPO_DATA=/path/to/grpo_train.jsonl bash train_grpo.sh"
    exit 1
fi

# Output directory name (auto-derived from model path if not set)
OUTPUT_DIR=${OUTPUT_DIR:-"${GRPO_OUTPUT_BASE}/$(basename $(dirname ${MODEL}))_grpo"}

cd ${SWIFT_DIR}

echo "================================================================"
echo "=== GRPO Training"
echo "=== Model: ${MODEL}"
echo "=== Data:  ${GRPO_DATA}"
echo "=== Output: ${OUTPUT_DIR}"
echo "================================================================"

# --- Launch vLLM rollout server (on last GPU) ---
CUDA_VISIBLE_DEVICES=3 swift rollout \
    --model_type qwen3 \
    --model "${MODEL}" \
    --port 8000 &
VLLM_PID=$!

echo "=== Waiting for vLLM server to start ==="
for i in $(seq 1 60); do
    if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
        echo "=== vLLM server is ready ==="
        break
    fi
    sleep 5
done

# --- GRPO training (on remaining GPUs) ---
CUDA_VISIBLE_DEVICES=0,1,2 \
NPROC_PER_NODE=3 \
swift rlhf \
    --rlhf_type grpo \
    --model_type qwen3 \
    --model "${MODEL}" \
    --external_plugins ${REWARD_PLUGIN} \
    --reward_funcs forecast_mixed \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --train_type full \
    --torch_dtype bfloat16 \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1 \
    --learning_rate ${LEARNING_RATE} \
    --max_length 8192 \
    --max_completion_length 8192 \
    --num_iterations 2 \
    --temperature 0.9 \
    --top_p 0.9 \
    --top_k 50 \
    --beta 0.02 \
    --dataset "${GRPO_DATA}" \
    --save_only_model true \
    --log_completions true \
    --dynamic_sample true \
    --overlong_filter true

# --- Cleanup ---
kill ${VLLM_PID} 2>/dev/null
wait ${VLLM_PID} 2>/dev/null

echo "=== GRPO training completed ==="
