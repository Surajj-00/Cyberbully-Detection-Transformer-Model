import csv

f = open('new datasets/archive (1)/twitter_parsed_dataset.csv', encoding='utf-8')
r = csv.reader(f)
header = next(r)
print('Header:', header)
rows = 0
labels = {}
for row in r:
    rows += 1
    label = row[3] if len(row) > 3 else 'unknown'
    labels[label] = labels.get(label, 0) + 1
print('Total rows:', rows)
print('Labels:', labels)
f.close()