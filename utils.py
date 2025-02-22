

import os
import datetime
import logging
import re
from my_tool import *
import random



SYSTEM_MSG_MC = '''
    You are an AI assistant who will help me to answer the question based on the given picture.
    Your output answer must be strictly one of the following: A, B or other capital letter given in the options.
    We hope you strictly follow the following format: only output \"A\" or \"B\" or \"Other letter\".
'''

SYSTEM_MSG_YN = '''
    You are an AI assistant who will help me to answer the question based on the given picture.
    The answer options are only Yes or No. 
    We hope you strictly follow the following format: only output \"Yes\" or \"No\"".
'''



def construct_query_base_MC(question, choices):
    prompt = f'''
    {SYSTEM_MSG_MC}
    {question}
    {dictionary_to_text(choices)}
    '''
    return prompt

# def construct_query_MC(question,choices,answer):
#     prompt = f'''
#     {SYSTEM_MSG_MC}
#     {SYSTEM_MISLEAD} {answer}
#     {question}
#     {dictionary_to_text(choices)}
#     '''
#     return prompt

# def construct_query_MMstar(question,answer):
#     prompt = f'''
#     {SYSTEM_MSG_MC}
#     {SYSTEM_MISLEAD} {answer}
#     {question}
#     '''
#     return prompt

# def construct_query_YN(query):
#     prompt = f'''
#     {SYSTEM_MISLEAD} {query} 
#     '''
#     return prompt


def dictionary_to_text(dictionary):
    # 按字母顺序输出字典
    sorted_keys = sorted(dictionary.keys())
    lines = []
    for key in sorted_keys:
        lines.append(f"{key}: {dictionary[key]}")
    return "\n".join(lines)

def  select_random_choice(answer,l):
    options = ['A', 'B', 'C', 'D','E','F',"G","H"][:l]
    remaining_options = [opt for opt in options if opt != answer]
    random_choice = random.choice(remaining_options)
    return random_choice

def cacluate_current_time_save_path(temp, args):

    initial_time = datetime.datetime.now()
    month_day = initial_time.strftime("%m%d")
    hour = initial_time.strftime("%H")
    formatted_time = initial_time.strftime("%Y-%m-%d_%H-%M-%S")
    save_directory = f"{args.json_save_dir}"
    os.makedirs(save_directory, exist_ok=True)

    filename = f"{formatted_time}.jsonl"
    save_path = f"{save_directory}/{temp}_{filename}"
    return save_path


def calculate_consistency_MC(data, filename):

    consistent_count = 0
    inconsistent_count = 0
    consistent_count_other = 0
    for item in data:
        # from IPython import embed; embed()
        answer = item['response']
        true_answer  = item['answer']

        if item['question_type_CHorYN']=='CH':
            if  true_answer in answer :
                consistent_count+=1
            else:
                inconsistent_count += 1
        elif item['question_type_CHorYN']=='YN':
            if  true_answer.replace(' ', '').lower() in answer :
                consistent_count+=1
            else:
                inconsistent_count += 1
        elif answer == -2:
            consistent_count_other +=1
        else:
            inconsistent_count += 1
    
    try:
        total = consistent_count + inconsistent_count
        consistent_ratio = consistent_count / total
        inconsistent_ratio = inconsistent_count / total
    except ZeroDivisionError:
        consistent_ratio = 0
        inconsistent_ratio = 0
    logging.info(f"Consistent Count: {consistent_count}")
    logging.info(f"Inconsistent Count: {inconsistent_count}")
    logging.info(f"{filename} Other Count: {consistent_count_other}")
    logging.info(f"{filename} Consistent Ratio: {consistent_ratio:.2%}")
    logging.info(f"{filename} Inconsistent Ratio: {inconsistent_ratio:.2%}")
    return consistent_ratio, inconsistent_ratio


def calculate_consistency_YN(data, filename):
    
    # Initialize counts for confusion matrix
    true_positive = 0  # yes-yes
    true_negative = 0  # no-no
    false_positive = 0  # yes-no
    false_negative = 0  # no-yes
    # unknow_count =0

    for item in data:
        response = item['response'].strip().lower()
        true_answer = item['answer'].strip().lower()

        if 'yes' in response:
            response = 'yes'
        elif 'no' in response:
            response = 'no'
        elif 'Unknown' in response:
            response = 'Unknown'

        if response == 'yes' and true_answer == 'yes':
            true_positive += 1
        elif response == 'no' and true_answer.lower() == 'no':
            true_negative += 1
        elif response == 'yes' and true_answer.lower() == 'no':
            false_positive += 1
        elif response == 'no' and true_answer == 'yes':
            false_negative += 1
        # elif response == 'Unknown':
        #     unknow_count +=1

    # Total count of responses
    total = true_positive + true_negative + false_positive + false_negative

    # Calculate ratios
    try:
        consistent_ratio = (true_positive + true_negative) / total
        inconsistent_ratio = (false_positive + false_negative) / total
    except ZeroDivisionError:
        consistent_ratio = 0
        inconsistent_ratio = 0

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Log the results
    logging.info(f"True Positive (yes-yes): {true_positive}")
    logging.info(f"True Negative (no-no): {true_negative}")
    logging.info(f"False Positive (yes-no): {false_positive}")
    logging.info(f"False Negative (no-yes): {false_negative}")
    logging.info(f"Consistent Count: {true_positive + true_negative}, {filename} Consistent Ratio: {consistent_ratio:.2%}")
    logging.info(f"Inconsistent Count: {false_positive + false_negative},{filename} Inconsistent Ratio: {inconsistent_ratio:.2%}")
    logging.info(f"{filename} Precision: {precision:.2%}, Recall: {recall:.2%}, F1 Score: {f1_score:.2%}")

    return consistent_ratio, inconsistent_ratio


def extract_option(response, question):
    # 提取选项及其描述
    options_pattern = r'\b([A-D]): ([^,]+)(?:,|$)'
    options = re.findall(options_pattern, question)
    
    # 检查响应是否完全匹配某个选项
    for option in options:
        option_letter = option[0]
        option_text = option[1].strip()
        if option_text in response:
            return option_letter

    # 如果响应中没有完全匹配的选项，检查是否有选项字母存在于响应中
    letter_pattern = r'\b([A-D])\b'
    match = re.search(letter_pattern, response)
    if match:
        return match.group(1)

    return response.replace(' ', '').lower()
# def extract_option(response, question):  
#     return response.replace(' ', '')

def find_image(start_path, filename):
    # Walk through the directory
    import os
    for dirpath, dirnames, filenames in os.walk(start_path):
        if filename in filenames:
            # Return the full path to the file
            return os.path.join(dirpath, filename)
    return None  # If the file was not found


def find_truth_by_id(data, target_id):
    for item in data:
        if item['id'] == target_id:
            return item['truth']
    return "ID not found"


def get_all_model_api_result(args, prompt, image):
    chat = Chat_gpt4v(model=args.api_model)
    response_temp = get_eval_plain_use_gpt4v(chat, prompt, image ,temperature=args.tempeature ,max_tokens=2048,fail_limit=3,return_resp=True)
    response_temp = response_temp[0]
    return response_temp 
