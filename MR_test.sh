echo "Job starts at $(date "+%Y-%m-%d %H:%M:%S")"

cuda_versions=("12.2" "12.1" "11.8" "11.3")


eval "$(conda shell.bash hook)"  

# parameters
json_save_dir=""
RUN_PY_PATH="${json_save_dir}/run.py"
is_equal='='
nums=(6 9 12)

do_sample=0



current_time=$(date "+%Y-%m-%d_%H-%M-%S")
month_day=$(date "+%m%d")
hour=$(date "+%H")

models=(
    "phi3-vision-128k-instruct"
    "minicpm-v-v2-chat"
    "yi-vl-6b-chat"
    "qwen-vl-chat"
    "deepseek-vl-7b-chat"   
    "llava1_6-mistral-7b-instruct"   
    "internvl-chat-v1_5"
    "minicpm-v-v2_5-chat"
    "glm4v-9b-chat" 
    "cogvlm2-en-19b-chat"
    "llava1_6-yi-34b-instruct"
    "yi-vl-34b-chat"
)

declare -A env_map
env_map=(
    ["phi3-vision-128k-instruct"]="MiniCPM-V"
    ["qwen-vl-chat"]="mmstar"
    ["glm4v-9b-chat"]="dyk_glm"
    ["cogvlm2-en-19b-chat"]="mmstar"
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

export CUDA_VISIBLE_DEVICES=0,1  
index=0
for model in "${models[@]}"; do
    for num in "${nums[@]}"; do
        conda activate "${env_map[$model]}"



        save_directory="${json_save_dir}/testdata_${is_equal}_${num}/${model}"
        mkdir -p "${save_directory}"


        LOG_FILE="${save_directory}/${model}.log"

        echo "Starting: python $RUN_PY_PATH --model_type $model " | tee -a "$LOG_FILE"
        for cuda_version in "${cuda_versions[@]}"; do
            export CUDA_VISIBLE_DEVICES=$((index % 4))

            python $RUN_PY_PATH --do_sample $do_sample --model_type $model --num $num --json_save_dir $save_directory --is_equal $is_equal 2>> "$LOG_FILE"
        if [ $? -ne 0 ]; then
            echo "Error occurred during: python $RUN_PY_PATH --model_type $model  with CUDA $cuda_version. Trying next CUDA version." | tee -a "$LOG_FILE"
            continue
            else
                break
            fi
        done

        if [ $? -ne 0 ]; then
            echo "All CUDA versions failed for: python $RUN_PY_PATH --model_type $model --dataset $dataset. Check $LOG_FILE for details." | tee -a "$LOG_FILE"
        fi
        ((index++))

    done
done
wait 
echo "Job ends at $(date "+%Y-%m-%d %H:%M:%S")"