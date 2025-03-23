{

    source /map-vepfs/miniconda3/etc/profile.d/conda.sh
    conda activate openrlhf
    export HF_HOME=/map-vepfs/huggingface

    # this would block the private network connection
    unset https_proxy;
    unset http_proxy

    set -x 
    # export NCCL_P2P_DISABLE=1
    WORKING_DIR='/map-vepfs/yizhi/OpenRLHF'
    cd $WORKING_DIR

    # export https_proxy="http://100.64.117.161:3128"
    # export http_proxy="http://100.64.117.161:3128"

    # export CUDA_VISIBLE_DEVICES=0,1,2,3
    
    # skip tal-scq5k-cn
    # supported: tal-scq5k-en, numina

    # export CUDA_VISIBLE_DEVICES=0,1
    # pt_path=${HF_HOME}/models/OpenRLHF/Llama-3-8b-sft-mixture
    # python -u -m evaluation.math_eval_demo \
    #     --data_dir $WORKING_DIR/evaluation/data --data_names gsm8k --split test \
    #     --start 0 --end -1 \
    #     --prompt_type "direct" --num_shots 0 \
    #     --model_name_or_path ${pt_path} \
    #     --use_vllm 


    # export CUDA_VISIBLE_DEVICES=0,1
    # pt_path="${HF_HOME}/models/OpenO1/OpenO1-Qwen-7B-v0.1/checkpoint-1000"
    # # "qwen25-math-cot"
    # python -u -m evaluation.math_eval_demo \
    #     --data_dir $WORKING_DIR/evaluation/data --data_names gsm8k --split test \
    #     --start 0 --end -1 \
    #     --prompt_type "qwen25-math-cot" --num_shots 0 \
    #     --model_name_or_path ${pt_path} \
    #     --use_vllm 

    export CUDA_VISIBLE_DEVICES=0,1,2,3
    pt_path="${HF_HOME}/models/Qwen/Qwen2.5-Math-7B"
    python -u -m evaluation.math_eval_demo \
        --data_dir $WORKING_DIR/evaluation/data --data_names gsm8k --split test \
        --start 0 --end -1 \
        --prompt_type "qwen25-math-cot" --num_shots 0 \
        --model_name_or_path ${pt_path} \
        --use_vllm 

    # export CUDA_VISIBLE_DEVICES=0,1,2,3
    # pt_path=/map-vepfs/yizhi/OpenRLHF/checkpoint/OpenO1-Qwen-7B-v0.1-ppo-rule-based-rm-v0.1-data-mix-10k_kl-5e-4
    # for data_name in "gsm8k" "numina" "tal-scq5k-en"; do
    #     python -u -m evaluation.math_eval_demo \
    #         --data_dir $WORKING_DIR/evaluation/data --data_names $data_name --split test \
    #         --start 0 --end -1 \
    #         --prompt_type "qwen25-math-cot" --num_shots 0 \
    #         --model_name_or_path ${pt_path} \
    #         --use_vllm 
    # done

    # export CUDA_VISIBLE_DEVICES=4,5,6,7
    # pt_path=/map-vepfs/yizhi/OpenRLHF/checkpoint/OpenO1-Qwen-7B-v0.1-ppo-rule-based-rm-v0.1-data-mix-10k_kl-5e-4
    # for data_name in "gsm8k" "numina" "tal-scq5k-en"; do
    #     python -u -m evaluation.math_eval_demo \
    #         --data_dir $WORKING_DIR/evaluation/data --data_names $data_name --split test \
    #         --start 0 --end -1 \
    #         --prompt_type "direct" --num_shots 0 \
    #         --model_name_or_path ${pt_path} \
    #         --use_vllm 
    # done


    # export CUDA_VISIBLE_DEVICES=2
    # # "qwen25-math-cot"
    # pt_path="${HF_HOME}/models/Qwen/Qwen2.5-Math-1.5B"
    # python -u -m evaluation.math_eval_demo \
    #     --data_dir $WORKING_DIR/evaluation/data --data_names gsm8k --split test \
    #     --start 0 --end -1 \
    #     --prompt_type "direct" --num_shots 0 \
    #     --model_name_or_path ${pt_path} \
    #     --use_vllm 

    # export CUDA_VISIBLE_DEVICES=3
    # pt_path="${HF_HOME}/models/Qwen/Qwen2.5-Math-1.5B"
    # python -u -m evaluation.math_eval_demo \
    #     --data_dir $WORKING_DIR/evaluation/data --data_names tal-scq5k-en --split test \
    #     --start 0 --end -1 \
    #     --prompt_type "direct" --num_shots 0 \
    #     --model_name_or_path ${pt_path} \
    #     --use_vllm 

    echo "done ${pt_path}"

    exit
}