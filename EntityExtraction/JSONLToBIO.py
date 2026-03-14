import json

input_file = 'entityFromDoccano.jsonl'
output_file = 'entityFromDoccano.bio'

with open(input_file, 'r', encoding='utf-8') as f_in, \
     open(output_file, 'w', encoding='utf-8') as f_out:
    for line in f_in:
        data = json.loads(line)
        text = data['text']
        entities = data.get('entities', [])
        tags = ['O'] * len(text)
        for ent in entities:
            start, end, label = ent['start_offset'], ent['end_offset'], ent['label']
            tags[start] = f'B-{label}'
            for i in range(start + 1, end):
                tags[i] = f'I-{label}'
        for char, tag in zip(text, tags):
            f_out.write(f'{char} {tag}\n')
        f_out.write('\n')  # 每条数据之间空行