import json
from tqdm import tqdm
import random
from datasets import load_dataset
import ast
from swift.llm import (
        get_model_tokenizer, get_template, inference, ModelType,
        get_default_template_type, inference_stream
    )
from swift.utils import seed_everything
import logging
from utils import *
from PIL import Image

def test_dataset(args,val_data, model, template ):
    seed_everything(42)

    start_time = time.time()
    answer_all=[]
    for idx, item in enumerate(tqdm(val_data, desc="All data")):

        index = item["new_id"]
        image = index
        question = item["question"]
        answer = item["answer"]
        question_type_CHorYN = item["question_type_CHorYN"]
        dataset=item["dataset"]
        all_category=item["all_category"]

        if question_type_CHorYN=='CH':
            if dataset=='ConBench_CH':
                prompt=  SYSTEM_MSG_MC+question 
            else:
                choices = item['choices']       
                prompt= construct_query_base_MC(question,choices)
        else:
            prompt= SYSTEM_MSG_YN + question
        
        if args.api_model:
            response_temp = get_all_model_api_result(args, prompt, image)
        elif args.model_type == 'phi3-vision-128k-instruct':
            prompt = f'<img>{image}</img>{prompt}'
            response_temp, _ = inference(model, template, prompt, temperature=args.tempeature)
        else :
            response_temp, _ = inference(model, template, prompt, images=image,temperature=args.tempeature)

        response = extract_option(response_temp, question)

        if question_type_CHorYN=='CH' and dataset!='ConBench_CH':
            answer_data_json = {
            'index': index,
            'question': question,
            'question_type_CHorYN': question_type_CHorYN,
            'choices':choices,
            'answer' : answer,
            'response': response,
            'response_temp': response_temp,
            'all_category':all_category ,
            'dataset':dataset,
            'prompt': prompt,
            "num":item["num"],
        }
        else:
            answer_data_json = {
            'index': index,
            'question': question,
            'question_type_CHorYN': question_type_CHorYN,
            #'choices':choices,
            'answer' : answer,
            'response': response,
            'response_temp': response_temp,
            'all_category':all_category ,
            'dataset':dataset,
            'prompt': prompt,
            "num":item["num"],
        }
        
        answer_all.append(answer_data_json)

    save_path = cacluate_current_time_save_path('all', args)
    json.dump(answer_all, open(save_path, 'w',encoding='utf-8'),
                      indent=2, ensure_ascii=False)
    consistent_ratio, inconsistent_ratio = calculate_consistency_MC(answer_all,'All data-')


    end_time = time.time()
    execution_time_seconds = end_time - start_time
    execution_time_minutes = execution_time_seconds / 60
    logging.info(f"Execution time: {execution_time_minutes:.2f} minutes")

    start_time = time.time()

    filtered_data = []
    for item in  answer_all:
        filtered_data.append(item)


    filtered_data_false_true = []
    filtered_data_true_false = []
    for item in filtered_data:
        response = item['response']
        true_answer  = item['answer']
        if item['question_type_CHorYN']=='CH':
            if  true_answer in response :
                filtered_data_true_false.append(item)
            else:
                filtered_data_false_true.append(item)
        else:
            if  true_answer.replace(' ', '').lower() in response :
                filtered_data_true_false.append(item)
            else:
                filtered_data_false_true.append(item)

    save_path= cacluate_current_time_save_path('all_false',args)
    json.dump(filtered_data_false_true, open(save_path, 'w',encoding='utf-8'),
                        indent=2, ensure_ascii=False)

    save_path= cacluate_current_time_save_path('all_true',args)
    json.dump(filtered_data_true_false, open(save_path, 'w',encoding='utf-8'),
                        indent=2, ensure_ascii=False)
    
    #False->True
    logging.info(f"False Count:{len(filtered_data_false_true)}")
    for i in range(1):
        answer_new_F_T = []
        for qid, item in enumerate(tqdm(filtered_data_false_true, desc="False->True")):
            index = item["index"]
            image = index
            question = item["question"]
            answer = item["answer"]
            question_type_CHorYN = item["question_type_CHorYN"]
            dataset=item["dataset"]
            all_category=item["all_category"]

            if question_type_CHorYN=='CH':
                if dataset=='ConBench_CH':
                    prompt=  SYSTEM_MSG_MC+question 
                else:
                    choices = item['choices']       
                    prompt= construct_query_base_MC(question,choices)
            else:
                prompt= SYSTEM_MSG_YN + question
            
            if args.api_model:
                response_temp = get_all_model_api_result(args, prompt, image)
            elif args.model_type == 'phi3-vision-128k-instruct':
                prompt = f'<img>{image}</img>{prompt}'
                response_temp, _ = inference(model, template, prompt, temperature=args.tempeature)
            else :
                response_temp, _ = inference(model, template, prompt, images=image,temperature=args.tempeature)

            response = extract_option(response_temp, question)

            if question_type_CHorYN=='CH' and dataset!='ConBench_CH':
                answer_data_json = {
                'index': index,
                'question': question,
                'question_type_CHorYN': question_type_CHorYN,
                'choices':choices,
                'answer' : answer,
                'response': response,
                'response_temp': response_temp,
                'all_category':all_category ,
                'dataset':dataset,
                'prompt': prompt,
                "num":item["num"],
            }
            else:
                answer_data_json = {
                'index': index,
                'question': question,
                'question_type_CHorYN': question_type_CHorYN,
                #'choices':choices,
                'answer' : answer,
                'response': response,
                'response_temp': response_temp,
                'all_category':all_category ,
                'dataset':dataset,
                'prompt': prompt,
                "num":item["num"],
            }
            answer_new_F_T.append(answer_data_json)

        save_path= cacluate_current_time_save_path('false_true',args)
        json.dump(answer_new_F_T, open(save_path, 'w',encoding='utf-8'),
                        indent=2, ensure_ascii=False)

        consistent_ratio_false_true, inconsistent_ratio_false_true = calculate_consistency_MC(answer_new_F_T,'False->True-')

    #True->Fasle
    logging.info(f"True Count:{len(filtered_data_true_false)}")
    for i in range(1):
        answer_new_T_F = []
        for qid, item in enumerate(tqdm(filtered_data_true_false, desc="True->False")):
            index = item["index"]
            image = index
            question = item["question"]
            answer = item["answer"]
            question_type_CHorYN = item["question_type_CHorYN"]
            dataset=item["dataset"]
            all_category=item["all_category"]

            if question_type_CHorYN=='CH':
                if dataset=='ConBench_CH':
                    prompt=  SYSTEM_MSG_MC+question 
                else:
                    choices = item['choices']       
                    prompt= construct_query_base_MC(question,choices)
            else:
                prompt= SYSTEM_MSG_YN + question
                
            if args.api_model:
                response_temp = get_all_model_api_result(args, prompt, image)
            elif args.model_type == 'phi3-vision-128k-instruct':
                prompt = f'<img>{image}</img>{prompt}'
                response_temp, _ = inference(model, template, prompt, temperature=args.tempeature)
            else :
                response_temp, _ = inference(model, template, prompt, images=image,temperature=args.tempeature)

            response = extract_option(response_temp, question)

            if question_type_CHorYN=='CH' and dataset!='ConBench_CH':
                answer_data_json = {
                'index': index,
                'question': question,
                'question_type_CHorYN': question_type_CHorYN,
                'choices':choices,
                'answer' : answer,
                'response': response,
                'response_temp': response_temp,
                'all_category':all_category ,
                'dataset':dataset,
                'prompt': prompt,
                "num":item["num"],
            }
            else:
                answer_data_json = {
                'index': index,
                'question': question,
                'question_type_CHorYN': question_type_CHorYN,
                #'choices':choices,
                'answer' : answer,
                'response': response,
                'response_temp': response_temp,
                'all_category':all_category ,
                'dataset':dataset,
                'prompt': prompt,
                "num":item["num"],
            }

            answer_new_T_F.append(answer_data_json)

        save_path= cacluate_current_time_save_path('true_false',args)
        json.dump(answer_new_T_F, open(save_path, 'w',encoding='utf-8'),
                        indent=2, ensure_ascii=False)
        consistent_ratio_true_false, inconsistent_ratio_true_false = calculate_consistency_MC(answer_new_T_F, 'True->False')