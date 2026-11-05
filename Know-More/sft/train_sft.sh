set -x

# Resolve this script's own directory, so it works regardless of where the
# repository was cloned.
SFT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SFT_DIR}"
echo pwd: $(pwd)

# GPUs to train on, and the port DeepSpeed uses to rendezvous.
GPU_VIS=${GPU_VIS:-0,1,2,3}
MASTER_PORT=${MASTER_PORT:-2353}

# REQUIRED: absolute path to the vanilla backbone to fine-tune.
MODEL_PATH="${MODEL_PATH:-}"
# REQUIRED: the SFT set built by `knowmore build --config <cfg> --format sft`,
# and a held-out split in the same {question, answer} JSON-Lines format.
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-}"
# REQUIRED: where checkpoints and logs are written.
OUTPUT_ROOT="${OUTPUT_ROOT:-}"

if [ -z "${MODEL_PATH}" ] || [ -z "${TRAIN_DATA_PATH}" ] || [ -z "${EVAL_DATA_PATH}" ] || [ -z "${OUTPUT_ROOT}" ]; then
    echo "ERROR: set MODEL_PATH, TRAIN_DATA_PATH, EVAL_DATA_PATH and OUTPUT_ROOT before" >&2
    echo "       launching, either by editing this script or by exporting them, e.g.:" >&2
    echo "         MODEL_PATH=/path/to/Qwen2.5-7B-Instruct \\" >&2
    echo "         TRAIN_DATA_PATH=/path/to/6_train.jsonl \\" >&2
    echo "         EVAL_DATA_PATH=/path/to/dev.jsonl \\" >&2
    echo "         OUTPUT_ROOT=/path/to/runs bash $0" >&2
    exit 1
fi

EXPERIMENT_NAME=${EXPERIMENT_NAME:-know-more-sft}
OUTPUT_DIR=${OUTPUT_ROOT}/${EXPERIMENT_NAME}
mkdir -p ${OUTPUT_DIR}/checkpoint
mkdir -p ${OUTPUT_DIR}/logs

LEARNING_RATE=${LEARNING_RATE:-1e-5}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-1024}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}

nohup deepspeed --include localhost:${GPU_VIS} --master_port ${MASTER_PORT} ${SFT_DIR}/train_sft.py \
    --model_name_or_path ${MODEL_PATH} \
    --train_data_path ${TRAIN_DATA_PATH} \
    --eval_data_path ${EVAL_DATA_PATH} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --output_dir ${OUTPUT_DIR}/checkpoint/ \
    --save_steps 100 \
    --eval_steps 100 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --learning_rate ${LEARNING_RATE} \
    --evaluation_strategy steps \
    --logging_strategy steps \
    --logging_steps 10 \
    --logging_dir ${OUTPUT_DIR}/logs/ \
    --bf16 True \
    --use_lora \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --deepspeed ${SFT_DIR}/ds_config_zero2.json > ${OUTPUT_DIR}/run.log 2>&1 &
