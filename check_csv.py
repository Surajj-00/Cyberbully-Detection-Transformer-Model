import csv

f = open('new datasets/Cyberbully_converted.csv', encoding='utf-8')
r = csv.reader(f)
header = next(r)
print('Header:', header)
count = 0
labels = {}
for row in r:
    count += 1
    lab = row[1] if len(row) > 1 else 'unknown'
    labels[lab] = labels.get(lab, 0) + 1
print('Total rows:', count)
print('Labels:', labels)
f.close()