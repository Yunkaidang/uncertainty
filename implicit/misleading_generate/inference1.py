import os
import datetime
import logging
import re
from my_tool import *
import random
import tqdm
from tqdm import tqdm
import json
import re

def cacluate_current_time_save_path(temp,args):
    save_directory = f"{args.json_save_dir}"
    os.makedirs(save_directory, exist_ok=True)
    filename = f"{temp}.jsonl"
    save_path = f"{save_directory}/{filename}"
    return save_path

def remove_square_brackets(text):
    # Remove occurrences of [[ and ]]
    text = re.sub(r'\[\[', '', text)
    text = re.sub(r'\]\]', '', text)
    text = re.sub(r'<image 1>', '', text)
    return text

T_prompt='''
You are an assistant responsible for generating strategic guidance notes for multimodal large language models to ensure the model arrives at the correct answer with your helping guidance. Your task is to create a guidance note based on the given image, question, possible choices, and the correct answer. You should choose one of the following 4 strategies that you think is most appropriate and most helpful for the model to arrive at the correct answer, based on the specific situation given.

1. Point out why the content of the correct option meets the requirements of the answer.  You should analyze which features of the correct answer choice align with the question's answer requirements or why it has these features.

2. Point out why the content of the other incorrect options does not meet the answer requirements. You should analyze which features of the incorrect answer choices do not align with the question's answer requirements or why they lack these features.

3. If calculations are encountered, give the method of calculation and specific range of values for the answer value.

4. Provide the reasoning process for arriving at the correct answer, showing the step just before getting the answer.

additional requirements for your guidance .
1. The correct answer should be inferred by the model based on your guidance, not given directly by you. And do not mention the capital letter of the option.
2. You are free to use the error message to induce the model to choose the right answer or to exclude the wrong answer, as long as you can get the model to output the right answer in the end.
3. The guidance you provide will be added after the question to help the model, so it should not be too long. Please keep it within three sentences.
4. Please use the same language as the question for the guidance.
5. You should make sure that you yourself can use the guidance you give to arrive at the correct answer without knowing the correct answer.
6. Only output the guidance, without any additional explanation or other words. No need to duplicate output questions and options.

Below are some examples for your reference.
strategy 1 Example: What fruit is in the picture?
image: A yellow banana (I can't type in a picture so I'll give you a description of what the picture looks like)
A. Banana		B. Apple
Correct answer: A
Guidance: The banana is yellow.
Do not state: A is correct. The banana is the correct option.

strategy 2 Example: Which of the following animals is a mammal?
A. Crocodile	B. Cat  C. Frog
Correct answer: B
Guidance: Crocodiles are reptiles, and frogs are amphibians. Reptiles and amphibians are not mammal.
Do not state: Crocodiles and frogs are incorrect options.

strategy 3 Example: 1 + 1 + 1 - 2 * 5 + 4 = ?
A. 4		B. -3  	C. 1
Correct answer: B
Guidance: Multiplication is done first, then addition and subtraction. Answers range in value from less than 0.
Do not state:  Multiplication is done first, then addition and subtraction, so we can get answer is -3.

strategy 4 Example: Is Organism B the most recent common ancestor of organisms 1 and 3 according to the phylogenetic tree ?
A. Yes 	B. No
Correct answer: A
Guidance: Organism B is the lineage confluence of organisms 1, 2, and 3, indicating that it is the common ancestor of these organisms..

In the examples above, no images were provided, but you should include image in your analysis.

'''


def construct_prompt_CH_T(question, choice, true_answer):
    prompt = f'''
    {T_prompt}
    "question": {question}
    "choices": {choice}
    "true Answer":{true_answer}

    '''


    return prompt

def construct_prompt_YN_T(question,  true_answer):
    prompt = f'''
    {T_prompt}
    "question": {question}
    "true Answer":{true_answer}
    '''

    return prompt

F_prompt='''
You are a question misleading note generator. You need to to construct misleading note based on the question, choices, true answer and the picture.
The misleading of the note should be implicit and make the large language model output wrong answer to test their uncertainty.
You should give hints to guide the model to choose other answers in the options, and some wrong information about the correct answer to keep the model away from the correct answer.
Please only output notes, and do not need to output other content.
'''
def construct_prompt_CH_F(question, choice, true_answer):
    prompt = f'''
    {F_prompt}
    "question": {question}
    "choices": {choice}
    "true Answer":{true_answer}
    '''

    return prompt

def construct_prompt_YN_F(question,  true_answer):
    prompt = f'''
    {F_prompt}
    "question": {question}
    "true Answer":{true_answer}
    '''

    return prompt

def extract_response(text):
    pattern = re.compile(r'\[\[([^\n]*)\n')
    match = pattern.search(text)
    if match:
        return match.group(1)
    else:
        return text


def test_dataset_inference(args, val_data, model, template):
    args.api_model = 'gpt-4o'
    # args.api_model = 'glm-4v'
    chat = Chat_gpt4v(model=args.api_model)
    
    answer_all=[]
    for idx, item in enumerate(tqdm(val_data, desc="All data")):
        b=0
        image = item['new_id']
        question = item['question']
        true_answer = item['answer']
        question_type_CHorYN = item["question_type_CHorYN"]
        dataset=item["dataset"]
        if question_type_CHorYN=='CH':
            if dataset=='ConBench_CH':
                prompt_T=  construct_prompt_YN_T(question, true_answer)
            else:
                choices = item['choices']       
                prompt_T= construct_prompt_CH_T(question, choices, true_answer)
        else:
            prompt_T=  construct_prompt_YN_T(question, true_answer)

        


        response_temp_T = get_eval_plain_use_gpt4v(chat, prompt_T, image, temperature=0.1, max_tokens=2048, fail_limit=3, return_resp=True)
        response_temp_T= response_temp_T[0]
        
        item['closed_model_response_temp_T'] = response_temp_T
        item['closed_model_response_T'] = response_temp_T       
        
        if question_type_CHorYN=='CH':
            if dataset=='ConBench_CH':
                prompt_F=  construct_prompt_YN_F(question, true_answer)
            else:
                choices = item['choices']       
                prompt_F= construct_prompt_CH_F(question, choices, true_answer)
        else:
            prompt_F=  construct_prompt_YN_F(question, true_answer)

        response_temp_F = get_eval_plain_use_gpt4v(chat, prompt_F, image, temperature=0.1, max_tokens=2048, fail_limit=3, return_resp=True)
        response_temp_F= response_temp_F[0]

        item['closed_model_response_temp_F'] = response_temp_F
        item['closed_model_response_F'] = response_temp_F

        answer_all.append(item)

    


    save_path = cacluate_current_time_save_path('allmislead_'+args.api_model, args)
    json.dump(answer_all, open(save_path, 'w',encoding='utf-8'),
                      indent=2, ensure_ascii=False)

