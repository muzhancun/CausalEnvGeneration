import os
import argparse
import numpy as np
import json
import shutil
import copy
import time
import datetime
import asyncio
import re
import copy
import glob 
from openai import OpenAI

import yaml


history_conversation = []

def fetch_gpt4(query):
    print('fetching gpt4 ...')
    client = OpenAI(api_key='empty')
    completion = client.chat.completions.create(
        model="gpt-4o", #gpt-4o-mini
        messages=[query],
        temperature=0.7
    )
    res = completion.choices[0].message.content
    return res

def generate_rule(task, description):
    with open('./prompt/rule_system_prompt.txt', 'r', encoding='utf-8') as file:  
        content = file.read()
    query = {
        "role": "user", "content": 
        content + 
        f'The task is ' + task + ': ' + description
        + f'please generate the score point of it'
        }
    # print(query)

    ans = fetch_gpt4(query)
    print(ans)
    answer = {"role": "assistant", "content": f'{ans}'}
    history_conversation.append(copy.deepcopy(query))
    history_conversation.append(copy.deepcopy(answer))

    return ans

with open('./mcu_task_description.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
for item in data:
    
    task_name = item.get("Task")
    description = item.get("Description")

    ans = generate_rule(task_name, description) 
    save_path = './task_rule_file/'
    rule_file = task_name.replace(' ', '_').lower()
    with open(save_path + f"{rule_file}.txt", 'w') as file:  
        file.write(ans)





