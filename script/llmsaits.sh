#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodelist=ai-gpgpu14

source ~/.bashrc
hostname

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

DATASETS=("physionet_2012")
MISSING_RATES=("0.3")
BATCH_SIZES=("32")
D_MODELS=("64")
D_FFNS=("128")
N_HEADS=("4")
N_LAYERS=("3")
LLM_MODELS=("gpt2")
PROMPTS=("Impute missing values where mask is zero.")

# Feature flag values for ablation
TRAIN_GPT_MLP_VALUES=(false) # true)
USE_LORA_VALUES=(true) # true)
USE_HANN_VALUES=(true)
ENABLE_PROFILING_VALUES=(false)

PROFILING_PATH="./output/imputation/profiling"
PROFILING_PREFIX="backbone_llmsaits"

# Fixed config
EPOCH=30
PATIENCE=5
N_STEPS=48
N_FEATURES=35
MODEL="llmsaits"

# Loop over datasets and hyperparameter configs
for DATASET in "${DATASETS[@]}"; do
  case "$DATASET" in
    "physionet_2012")
      N_STEPS=48
      N_FEATURES=35
      ;;
    "etth1" | "etth2" | "ettm1" | "ettm2")
      N_STEPS=96
      N_FEATURES=7
      ;;
    "air_quality" | "beijing_multisite_air_quality" | "italy_air_quality")
      N_STEPS=48
      N_FEATURES=36
      ;;
    "pems_traffic")
      N_STEPS=96
      N_FEATURES=228
      ;;
    "solar_alabama")
      N_STEPS=96
      N_FEATURES=137
      ;;
    "electricity_load_diagrams")
      N_STEPS=168
      N_FEATURES=370
      ;;
  # ===== UCR_UE Datasets =====
    "ucr_uea_MelbournePedestrian")
      N_STEPS=24
      N_FEATURES=1
      ;;
    "ucr_uea_ECG200")
      N_STEPS=96
      N_FEATURES=1
      ;;
    "ucr_uea_LargeKitchenAppliances")
      N_STEPS=720
      N_FEATURES=3
      ;;
    "ucr_uea_PowerCons")
      N_STEPS=144
      N_FEATURES=1
      ;;
    "ucr_uea_ItalyPowerDemand")
      N_STEPS=24
      N_FEATURES=1
      ;;
    *)
      echo "❌ Unknown dataset: $DATASET"
      exit 1
      ;;
  esac

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

                          SAVE_DIR="output/imputation/${MODEL}/${LLM_MODEL}/${DATASET}/epoch${EPOCH}/mr${MISSING_RATE}_bs${BATCH_SIZE}_dm${D_MODEL}_ffn${D_FFN}_h${N_HEAD}_ly${N_LAYER}/mlp${TRAIN_GPT_MLP}_lora${USE_LORA}_hann${USE_HANN}"

                          echo "🔥 Running llmsaits on ${DATASET}: mlp=$TRAIN_GPT_MLP, lora=$USE_LORA, hann=$USE_HANN, profiling=$ENABLE_PROFILING"

                          CMD="python main.py \
                            --model $MODEL \
                            --dataset_name $DATASET \
                            --epochs $EPOCH \
                            --patience $PATIENCE \
                            --missing_rate $MISSING_RATE \
                            --saving_path $SAVE_DIR \
                            --device $DEVICE \
                            --n_steps $N_STEPS \
                            --n_features $N_FEATURES \
                            --n_layers $N_LAYER \
                            --d_model $D_MODEL \
                            --d_ffn $D_FFN \
                            --n_heads $N_HEAD \
                            --llm_model_type $LLM_MODEL \
                            --batch_size $BATCH_SIZE \
                            --prompt_template \"$PROMPT\" \
                            --train_gpt_mlp $TRAIN_GPT_MLP \
                            --use_lora $USE_LORA \
                            --use_hann_window $USE_HANN \
                            --enable_profiling $ENABLE_PROFILING \
                            --profiling_path $PROFILING_PATH \
                            --profiling_prefix $PROFILING_PREFIX"

                          eval $CMD

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

echo "✅ All llmsaits runs completed."