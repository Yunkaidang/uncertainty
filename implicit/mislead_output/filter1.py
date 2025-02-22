import json

# File paths
in_file = '/hpc2hdd/JH_DATA/share/zrao538/PrivateShareGroup/zrao538_NLPGroup/dyk/View/result/upload/all_combined_items.jsonl'
out_file = '/hpc2hdd/JH_DATA/share/zrao538/PrivateShareGroup/zrao538_NLPGroup/dyk/View/result/upload/all_combined_items.jsonl'

import re

def extract_guidanceA_content(text):
    # 使用正则表达式匹配'guidanceA': 到 'guidanceB':之间的内容
    match = re.search(r"'guidanceA':(.*?)'guidanceB':", text, re.DOTALL)
    if match:
        # 去掉左右空格并返回内容
        return match.group(1).strip()
    else:
        return None  # 如果找不到匹配内容则返回None



# Load the list of questions from duplicates_output.json
with open(in_file, 'r', encoding='utf-8') as f:
    in_data = json.load(f)




# Filter out items where "query" contains any of the duplicate questions
filtered_data = []
for item in in_data:
    retain_ls=["question","question_type_CHorYN","choices","answer","response","all_category","question_correction","model","guide","dataset","num","new_id","closed_model_response_T","closed_model_response_F"]
    keys_to_delete = [key for key in item if key not in retain_ls]
    for key in keys_to_delete:
        del item[key]

        filtered_data.append(item)

# Save the filtered data to a new JSONL file

json.dump(filtered_data, open(out_file, 'w',encoding='utf-8'),
                        indent=2, ensure_ascii=False)


