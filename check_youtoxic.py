import csv

f = open('new datasets/YouToxic English _Audio Dataset/youtoxic_english_1000.csv', encoding='utf-8')
r = csv.reader(f)
header = next(r)
print('Header:', header)
rows = 0
for row in r:
    rows += 1
print('Total rows:', rows)
f.close()