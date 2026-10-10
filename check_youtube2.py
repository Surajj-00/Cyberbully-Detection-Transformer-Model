import csv

f = open('new datasets/archive (1)/youtube_parsed_dataset.csv', encoding='utf-8')
r = csv.reader(f)
header = next(r)
print('Header:', header)
rows = 0
label0 = 0
label1 = 0
for row in r:
    rows += 1
    label = row[9] if len(row) > 9 else 'unknown'
    if label == '0':
        label0 += 1
    elif label == '1':
        label1 += 1
print('Total rows:', rows)
print(f'Label 0: {label0} ({label0/rows*100:.1f}%)')
print(f'Label 1: {label1} ({label1/rows*100:.1f}%)')
f.close()