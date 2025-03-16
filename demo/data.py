import json

path = '../dataset/openwebtext-10k.jsonl'

samples = []
with open(path, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        data = json.loads(line.strip())
        samples.append(data)
        print(data)
        break

