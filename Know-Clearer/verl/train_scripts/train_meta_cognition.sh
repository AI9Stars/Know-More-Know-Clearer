set -x

# Resolve the repo root from this script's own location, so the script works
# regardless of where the repository was cloned.
WORKING_ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${WORKING_ROOT_DIR}"
echo pwd: $(pwd)
unset VLLM_USE_MODELSCOPE LMDEPLOY_USE_MODELSCOPE

export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-${WORKING_ROOT_DIR}/wandb_logs}
# Only needed when WANDB_MODE=online; export it in your shell instead if you prefer.
export WANDB_API_KEY=${WANDB_API_KEY:-xxxxx}

GPU_NUMS=4
WORLD_SIZE=0

DATA_NAME=know-clearer

TRAIN_DATA_PATH="[$(echo ${WORKING_ROOT_DIR}/data/train-clean.parquet)]"
TEST_DATA_PATH="[$(echo ${WORKING_ROOT_DIR}/data/dev-clean.parquet)]"


MODEL_NAME=Qwen2.5-7B
# REQUIRED: absolute path to the pretrained / SFT-warmed-up checkpoint used as the actor initialization.
MODEL_PATH="${MODEL_PATH:-}"

PROJECT_NAME=know-clearer
EXPERIMENT_NAME=${MODEL_NAME}-${DATA_NAME}-NODE${WORLD_SIZE}
# REQUIRED: absolute path to the root directory where PPO checkpoints and logs will be saved.
OUTPUT_ROOT="${OUTPUT_ROOT:-}"

if [ -z "${MODEL_PATH}" ] || [ -z "${OUTPUT_ROOT}" ]; then
    echo "ERROR: set MODEL_PATH and OUTPUT_ROOT before launching, either by editing" >&2
    echo "       this script or by exporting them, e.g.:" >&2
    echo "         MODEL_PATH=/path/to/ckpt OUTPUT_ROOT=/path/to/runs bash $0" >&2
    exit 1
fi

OUTPUT_DIR=${OUTPUT_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}
save_batch_dir=${OUTPUT_DIR}/ppo_batchs
mkdir -p ${OUTPUT_DIR}
mkdir -p ${save_batch_dir}
PROMPT_LENGTH=1024 
RESPONSE_LENGTH=32
NUM_BATCHED_TOKENS=$((${PROMPT_LENGTH} + ${RESPONSE_LENGTH}))
export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup bash -c "
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_DATA_PATH" \
    data.val_files="$TEST_DATA_PATH" \
    data.train_batch_size=128 \
    data.max_prompt_length=${PROMPT_LENGTH} \
    data.max_response_length=${RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.max_num_batched_tokens=${NUM_BATCHED_TOKENS} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_k=45 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.val_before_train=False \
    data.val_batch_size=128 \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.default_local_dir=${OUTPUT_DIR} \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.val_only=False \
    trainer.total_epochs=1 \
    2>&1 | tee ${OUTPUT_DIR}/train.log
" &