from frame import frame
from utils import TermDataset, load_wiki_dataset
from openai import Client
from multiprocessing import Pool
import json
from datetime import datetime
import os
from tqdm import tqdm
from pathlib import Path
import http.client
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', '--api', type=str, required=True)
    args = parser.parse_args()
    return args


def extract(wikipage, api_key):
    valid = False
    while not valid:
        try:
            conn = http.client.HTTPSConnection("www.dwyu.top")
            instruction_prompt = frame.format(wiki=wikipage.strip())
            payload = json.dumps({
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": instruction_prompt
                    }
                ]
            })
            headers = {
                'Authorization': api_key,
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            }
            conn.request("POST", "/v1/chat/completions", payload, headers)
            response = conn.getresponse()
            response = response.read().decode('utf-8')
            response = json.loads(response)
            content = response['choices'][0]['message']['content']
            valid = True
        except Exception as ex:
            print(ex)
    return content


def main():
    args = parse_args()
    dataset = load_wiki_dataset()
    wdata = TermDataset(dataset)
    
    responses_dir = str(Path(__file__).parent / 'responses')
    os.makedirs(responses_dir, mode=0o777, exist_ok=True)
    start = 0 if len(os.listdir(responses_dir)) == 0 else sorted([int(x[:-5]) for x in os.listdir(responses_dir)])[-1] + 1
    
    for i in tqdm(range(start, len(dataset))):
        valid = False
        while not valid:
            title = wdata.getterm(i)
            answer = extract(wdata[i], api_key=args.api_key)
            answer = answer.replace('```json', '').replace('```', '')
            try:
                answer = json.loads(answer)
                answer = {
                    'title': title,
                    'relationships': answer['relationships'],
                    'properties': answer['properties']
                }
                valid = True
            except:
                print(i, title)
        
        with open(os.path.join(responses_dir, f"{i}.json"), 'w') as f:
            json.dump(answer, f, indent=4)


if __name__ == '__main__':
    main()