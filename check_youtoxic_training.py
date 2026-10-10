import csv

f = open('new datasets/YouToxic_training.csv', encoding='utf-8')
r = csv.reader(f)
header = next(r)
print('Header:', header)
toxic = 0
normal = 0
for row in r:
    lab = int(row[1])
    if lab == 1:
        toxic += 1
    else:
        normal += 1
print(f'Total rows: {normal + toxic}')
print(f'Normal (0): {normal} ({normal/(normal+toxic)*100:.1f}%)')
print(f'Toxic (1): {toxic} ({toxic/(normal+toxic)*100:.1f}%)')
f.close()