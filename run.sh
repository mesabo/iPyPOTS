#!/bin/bash


source ~/.bashrc
hostname

export CUDA_VISIBLE_DEVICES=3,4
NUM_GPUS=2

# Detect computation device
DEVICE="cpu"
if [[ -n "$CUDA_VISIBLE_DEVICES" && $(nvidia-smi | grep -c "GPU") -gt 0 ]]; then
    DEVICE="cuda"
fi

# Activate Conda environment
ENV_NAME="pypots"
if [[ -z "$CONDA_DEFAULT_ENV" || "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
    if ! command -v conda &>/dev/null; then
        echo "❌ Conda not found."
        exit 1
    fi
    source activate "$ENV_NAME" || { echo "❌ Could not activate Conda env '$ENV_NAME'."; exit 1; }
fi

# ========= EXPERIMENT CONFIGS =========

# Datasets & grid
DATASETS=("physionet_2012")
MISSING_RATES=("0.3")
BATCH_SIZES=("32")
D_MODELS=("64")
D_FFNS=("128")
N_HEADS=("4")
N_LAYERS=("3")
LLM_MODELS=("gpt2")
PROMPTS=("Impute missing values where mask is zero.")

# Ablations
TRAIN_GPT_MLP_VALUES=(false)
USE_LORA_VALUES=(false)
USE_HANN_VALUES=(false)
ENABLE_PROFILING_VALUES=(true)
USE_PROMPT_VALUES=(true)
USE_REPROGRAMMING_VALUES=(true)

# Profiling & paths (now include DEVICE segment)
ROOT_OUT="output/imputation/${DEVICE}"
PROFILING_PATH="${ROOT_OUT}/profiling"
PROFILING_PREFIX="backbone_llm4imp"
mkdir -p "${ROOT_OUT}" "${PROFILING_PATH}"

# Fixed config
EPOCH=2
PATIENCE=1
MODEL="llm4imp"

# Global session log
SESSION_TAG="$(date +%Y%m%dT%H%M%S)"
SESSION_LOG="${ROOT_OUT}/session_${SESSION_TAG}.log"
echo "Session log: ${SESSION_LOG}" | tee -a "${SESSION_LOG}"

# ---- helper: map dataset -> N_STEPS, N_FEATURES (same mapping you had) ----
set_dims_for_dataset () {
  local ds="$1"
  case "$ds" in
    "physionet_2012")
      N_STEPS=48; N_FEATURES=35 ;;
    "etth1" | "etth2" | "ettm1" | "ettm2")
      N_STEPS=96; N_FEATURES=7 ;;
    "air_quality" | "beijing_multisite_air_quality" | "italy_air_quality")
      N_STEPS=48; N_FEATURES=36 ;;
    "pems_traffic")
      N_STEPS=96; N_FEATURES=228 ;;
    "solar_alabama")
      N_STEPS=96; N_FEATURES=137 ;;
    "electricity_load_diagrams")
      N_STEPS=168; N_FEATURES=370 ;;
    # ===== UCR_UE Datasets =====
    "ucr_uea_MelbournePedestrian")
      N_STEPS=24; N_FEATURES=1 ;;
    "ucr_uea_ECG200")
      N_STEPS=96; N_FEATURES=1 ;;
    "ucr_uea_LargeKitchenAppliances")
      N_STEPS=720; N_FEATURES=3 ;;
    "ucr_uea_PowerCons")
      N_STEPS=144; N_FEATURES=1 ;;
    "ucr_uea_ItalyPowerDemand")
      N_STEPS=24; N_FEATURES=1 ;;
    *)
      echo "❌ Unknown dataset: $ds" | tee -a "${SESSION_LOG}"
      return 1 ;;
  esac
  return 0
}

# ---------------- main grid ----------------
for DATASET in "${DATASETS[@]}"; do
  if ! set_dims_for_dataset "$DATASET"; then
    exit 1
  fi

  for MISSING_RATE in "${MISSING_RATES[@]}"; do
    for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
      for D_MODEL in "${D_MODELS[@]}"; do
        for D_FFN in "${D_FFNS[@]}"; do
          for N_HEAD in "${N_HEADS[@]}"; do
            for N_LAYER in "${N_LAYERS[@]}"; do
              for LLM_MODEL in "${LLM_MODELS[@]}"; do
                for PROMPT in "${PROMPTS[@]}"; do
                  for TRAIN_GPT_MLP in "${TRAIN_GPT_MLP_VALUES[@]}"; do
                    for USE_LORA in "${USE_LORA_VALUES[@]}"; do
                      for USE_HANN in "${USE_HANN_VALUES[@]}"; do
                        for ENABLE_PROFILING in "${ENABLE_PROFILING_VALUES[@]}"; do
                          for USE_PROMPT in "${USE_PROMPT_VALUES[@]}"; do
                            for USE_REPROGRAMMING in "${USE_REPROGRAMMING_VALUES[@]}"; do

                              # Output dirs (device-aware) + logs
                              SAVE_DIR="${ROOT_OUT}/${MODEL}/${LLM_MODEL}/${DATASET}/epoch${EPOCH}/mr${MISSING_RATE}_bs${BATCH_SIZE}_dm${D_MODEL}_ffn${D_FFN}_h${N_HEAD}_ly${N_LAYER}/mlp${TRAIN_GPT_MLP}_lora${USE_LORA}_hann${USE_HANN}/prompt${USE_PROMPT}_reprog${USE_REPROGRAMMING}"
                              LOG_DIR="${SAVE_DIR}/logs"
                              mkdir -p "${SAVE_DIR}" "${LOG_DIR}"

                              RUN_TAG="mr${MISSING_RATE}_bs${BATCH_SIZE}_dm${D_MODEL}_ffn${D_FFN}_h${N_HEAD}_ly${N_LAYER}_mlp${TRAIN_GPT_MLP}_lora${USE_LORA}_hann${USE_HANN}_prompt${USE_PROMPT}_reprog${USE_REPROGRAMMING}"
                              RUN_LOG="${LOG_DIR}/run_${RUN_TAG}.log"
                              DONE_MARK="${SAVE_DIR}/.done"

                              if [[ -f "${DONE_MARK}" ]]; then
                                echo "⏭️  Skip (done): ${DATASET} | ${RUN_TAG}" | tee -a "${SESSION_LOG}"
                                continue
                              fi

                              echo "🔥 RUN: ${DATASET} | ${RUN_TAG} (DEVICE=${DEVICE}, N_STEPS=${N_STEPS}, N_FEATURES=${N_FEATURES})" | tee -a "${SESSION_LOG}"

                              # Build command as array (NO eval, safe quoting)
                              cmd=(python main.py
                                --model "${MODEL}"
                                --dataset_name "${DATASET}"
                                --epochs "${EPOCH}"
                                --patience "${PATIENCE}"
                                --missing_rate "${MISSING_RATE}"
                                --saving_path "${SAVE_DIR}"
                                --device "${DEVICE}"
                                --n_steps "${N_STEPS}"
                                --n_features "${N_FEATURES}"
                                --n_layers "${N_LAYER}"
                                --d_model "${D_MODEL}"
                                --d_ffn "${D_FFN}"
                                --n_heads "${N_HEAD}"
                                --llm_model_type "${LLM_MODEL}"
                                --batch_size "${BATCH_SIZE}"
                                --prompt_template "${PROMPT}"
                                --train_gpt_mlp "${TRAIN_GPT_MLP}"
                                --use_lora "${USE_LORA}"
                                --use_hann_window "${USE_HANN}"
                                --enable_profiling "${ENABLE_PROFILING}"
                                --profiling_path "${PROFILING_PATH}"
                                --profiling_prefix "${PROFILING_PREFIX}"
                                --use_prompt "${USE_PROMPT}"
                                --use_reprogramming "${USE_REPROGRAMMING}"
                              )

                              {
                                echo "===== CMD @ $(date -Is) =====" >> "${RUN_LOG}"
                                printf '%q ' "${cmd[@]}" >> "${RUN_LOG}"; echo >> "${RUN_LOG}"
                                echo "================================" >> "${RUN_LOG}"

                                # Prefer srun if available (cleaner under SLURM), else run directly
                                if command -v srun &>/dev/null; then
                                  srun --quiet --unbuffered "${cmd[@]}" |& tee -a "${RUN_LOG}"
                                else
                                  "${cmd[@]}" |& tee -a "${RUN_LOG}"
                                fi
                              }

                              status=${PIPESTATUS[0]}
                              if [[ $status -eq 0 ]]; then
                                touch "${DONE_MARK}"
                                echo "✅ DONE: ${DATASET} | ${RUN_TAG}" | tee -a "${SESSION_LOG}"
                              else
                                echo "❌ FAIL(${status}): ${DATASET} | ${RUN_TAG}" | tee -a "${SESSION_LOG}"
                              fi

                            done
                          done
                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

echo "✅ All LLM4IMP runs completed at $(date -Is)."