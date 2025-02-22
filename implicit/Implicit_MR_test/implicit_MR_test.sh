#!/bin/bash

echo "Job starts at $(date "+%Y-%m-%d %H:%M:%S")"

# CUDA 版本列表 (You can adjust CUDA versions according to your local setup)
cuda_versions=("12.2" "12.1" "11.8" "11.3")

# Initialize Conda environment
eval "$(conda shell.bash hook)"

# Modify the paths and parameters
json_save_dir=""
RUN_PY_PATH="run.py"
is_equal='='
num=12
do_sample=0
num_sample=20
current_time=$(date "+%Y-%m-%d_%H-%M-%S")
month_day=$(date "+%m%d")
hour=$(date "+%H")

# Models and datasets
models=("clode")
datasets=("test_dataset")

declare -A env_map
env_map=(
    ["phi3-vision-128k-instruct"]="MiniCPM-V"
    ["qwen-vl-chat"]="mmstar"
    ["glm4v-9b-chat"]="dyk_glm"
    ["cogvlm2-19b-chat"]="mmstar"
    ["minicpm-v-v2-chat"]="mmstar"
    ["deepseek-vl-7b-chat"]="MiniCPM-V"
    ["minicpm-v-v2_5-chat"]="MiniCPM-V"
    ["llava-llama-3-8b-v1_1"]="dyk_llava"
    ["llava1_6-mistral-7b-instruct"]="dyk_llava"
    ["llava1_6-yi-34b-instruct"]="MiniCPM-V"
    ["yi-vl-6b-chat"]="dyk_llava"
    ["yi-vl-34b-chat"]="dyk_llava"
    ["internvl-chat-v1_5"]="dyk_llava"
)

# Loop over datasets and models
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        # Activate corresponding environment
        conda activate 'mmstar'

        # Define output directory
        save_directory="test${is_equal}_${num}/${model}"
        mkdir -p "${save_directory}"

        LOG_FILE="${save_directory}/${model}.log"
        echo "Starting: python $RUN_PY_PATH --model_type $model --dataset $dataset" | tee -a "$LOG_FILE"

        # Try each CUDA version
        for cuda_version in "${cuda_versions[@]}"; do
            export CUDA_VISIBLE_DEVICES=0  # Assuming 1 GPU is available, change if needed
            echo "Using CUDA version: $cuda_version"
            
            # Run the Python script with the selected CUDA version
            CUDA_VISIBLE_DEVICES=0 python $RUN_PY_PATH --do_sample $do_sample --model_type $model --num $num --json_save_dir $save_directory --is_equal $is_equal --num_sample $num_sample  2>> "$LOG_FILE"
            
            # Check if the command succeeded
            if [ $? -ne 0 ]; then
                echo "Error occurred during: python $RUN_PY_PATH --model_type $model --dataset $dataset with CUDA $cuda_version. Trying next CUDA version." | tee -a "$LOG_FILE"
                continue
            else
                break
            fi
        done

        # Check if all CUDA versions failed
        if [ $? -ne 0 ]; then
            echo "All CUDA versions failed for: python $RUN_PY_PATH --model_type $model --dataset $dataset. Check $LOG_FILE for details." | tee -a "$LOG_FILE"
        fi
    done
done

echo "Job ends at $(date "+%Y-%m-%d %H:%M:%S")"
