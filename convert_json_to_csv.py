import json
import csv

# Read JSONL and convert to CSV
input_file = 'new datasets/Cyberbullying detection datasets/Cyber_Bully_Data.json'
output_file = 'new datasets/Cyberbully_converted.csv'

with open(input_file, 'r', encoding='utf-8') as f:
    entries = []
    for line in f:
        line = line.strip()
        if line:
            entry = json.loads(line)
            entries.append(entry)

# Write CSV
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['text', 'label'])  # header
    for entry in entries:
        text = entry.get('text', '')
        label = entry.get('label', 'unknown')
        writer.writerow([text, label])

print(f'Converted {len(entries)} entries to {output_file}')