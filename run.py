import json
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'   
from tqdm import tqdm
import os
import datetime
import random
import re
import torch
from swift.tuners import Swift
from datasets import load_dataset
import argparse
from swift.llm import (
        get_model_tokenizer, get_template, inference, ModelType,
        get_default_template_type, inference_stream
    )
from swift.utils import seed_everything
import logging
from modelscope import (
    snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
)
from test_dataset import test_dataset
from utils import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--local_image_path', type=str, default='')
    parser.add_argument('--model_type', type=str, default='minicpm-v-v2_5-chat')
    parser.add_argument('--num', type=str, default=9)
    parser.add_argument('--do_sample', type=int, default= 1 )
    parser.add_argument('--num_sample', type=int, default= 5 )
    parser.add_argument('--api_model', type=str, default='')
    parser.add_argument('--tempeature', type=float, default=0.4 )
    parser.add_argument('--json_save_dir', type=str, default='')
    parser.add_argument('--is_equal', type=str, default='=')
    args = parser.parse_args()

    # load local fintune model:
    if args.model_type == 'Qwen-VL-Chat':
        model_id = 'qwen/Qwen-VL-Chat'
        revision = 'v1.0.0'
        model_dir = snapshot_download(model_id, revision=revision)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if not hasattr(tokenizer, 'model_dir'):
            tokenizer.model_dir = model_dir
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda", trust_remote_code=True).eval()
    if args.model_type == 'closed_model':
        model = None
        template = None
        print(args.api_model)
    elif args.model_type :
        # from IPython import embed; embed()
        model_type = args.model_type
        template_type = get_default_template_type(model_type)
        model, tokenizer = get_model_tokenizer(model_type, torch.float16, model_kwargs={'device_map': 'auto'})
        model.generation_config.max_new_tokens = 256
        model.generation_config.do_sample = True  # 启用采样模式
        model.generation_config.top_k = 80
        model.generation_config.top_p = 0.9
        template = get_template(template_type, tokenizer)

    try:
        path='all_combined_items.jsonl'
        with open(path, 'r', encoding='utf-8') as f:
            try:
                content = f.read()
                data = json.loads(content)
                if args.is_equal == '=' :
                    val_data=[]
                    for item in data:
                        # print(item)
                        if int(item['num'])==int(args.num):
                            val_data.append(item)
                elif args.is_equal == '>':                
                    val_data=[]
                    for item in data:
                        if int(item['num'])>=int(args.num):
                            val_data.append(item)
                else:
                    raise NotImplementedError
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file {path}: {e}")
        
    except Exception as e:
        print("")
        print(f"Error occurred: {e}. Loading 3000 samples dataset.")

    if args.do_sample:
        subset = []
        for i in range(args.num_sample):
            subset.append(val_data[i])
        val_data = subset

    initial_time = datetime.datetime.now()
    month_day = initial_time.strftime("%m%d")
    hour = initial_time.strftime("%H")
    formatted_time = initial_time.strftime("%Y-%m-%d_%H-%M-%S")
    save_directory = f"{args.json_save_dir}"
    os.makedirs(save_directory, exist_ok=True)

    log_filename = f'{save_directory}/{args.model_type}.txt'
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print('run',len(val_data))
    test_dataset(args,val_data, model, template )
