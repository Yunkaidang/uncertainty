echo "Job starts at $(date "+%Y-%m-%d %H:%M:%S")"

# 遍历 CUDA 版本并加载
cuda_versions=("12.2" "12.1" "11.8" "11.3")
for cuda_version in "${cuda_versions[@]}"; do
    echo "Loading CUDA $cuda_version"
    module load cuda/$cuda_version

    if [ $? -eq 0 ]; then
        echo "CUDA $cuda_version loaded successfully."
        break
    else
        echo "Failed to load CUDA $cuda_version, trying next version."
        module unload cuda/$cuda_version
    fi
done
# 检查是否成功加载了CUDA
if [ $? -ne 0 ]; then
    echo "All CUDA versions failed to load. Exiting."
    exit 1
fi

# GCC 版本列表, 遍历 GCC 版本并加载
gcc_versions=("compilers/gcc-11.1.0" "compilers/gcc-12.2.0" "compilers/gcc-13.1.0")
for gcc_version in "${gcc_versions[@]}"; do
    echo "Loading GCC $gcc_version"
    module load $gcc_version

    if [ $? -eq 0 ]; then
        echo "GCC $gcc_version loaded successfully."
        break
    else
        echo "Failed to load GCC $gcc_version, trying next version."
        module unload $gcc_version
    fi
done

# 修改的参数
json_save_dir=""
RUN_PY_PATH="mislead_generate.py"
is_equal='='
num=12
do_sample=0
num_sample=20
current_time=$(date "+%Y-%m-%d_%H-%M-%S")
month_day=$(date "+%m%d")
hour=$(date "+%H")


models=(
clode
)

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


for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        micromamba activate 'mmstar'
        save_directory="mislead_gene/test${is_equal}_${num}"
        mkdir -p "${save_directory}"

        LOG_FILE="${save_directory}/${model}.log"
        echo "Starting: python $RUN_PY_PATH --model_type $model --dataset $dataset" | tee -a "$LOG_FILE"

        for cuda_version in "${cuda_versions[@]}"; do
            module load cuda/$cuda_version
            CUDA_VISIBLE_DEVICES=0  python $RUN_PY_PATH --do_sample $do_sample --model_type $model --num $num --json_save_dir $save_directory --is_equal $is_equal --num_sample $num_sample  2>> "$LOG_FILE"
            
            if [ $? -ne 0 ]; then
                echo "Error occurred during: python $RUN_PY_PATH --model_type $model --dataset $dataset with CUDA $cuda_version. Trying next CUDA version." | tee -a "$LOG_FILE"
                module unload cuda/$cuda_version
                continue
            else
                break
            fi

        done

        if [ $? -ne 0 ]; then
            echo "All CUDA versions failed for: python $RUN_PY_PATH --model_type $model --dataset $dataset. Check $LOG_FILE for details." | tee -a "$LOG_FILE"
        fi
    done
done

echo "Job ends at $(date "+%Y-%m-%d %H:%M:%S")"
