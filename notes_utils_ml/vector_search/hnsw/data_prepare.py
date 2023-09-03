import json
import pandas as pd

file = r"C:\Users\Jejune\Desktop\WebQA.v1.0\me_test.ann.json"

data_orig = json.load(open(file))
data = []
questions, answers = [], []
for q_id in data_orig:
    question = data_orig[q_id]['question']
    answer = data_orig[q_id]['evidences'][f"{q_id}#00"]['answer'][0]
    data.append({'query': question, 'answer': answer})

with open("data_qa.jsonl", 'w') as f:
    for item in data:
        f.write(json.dumps(item) + "\n")

