import csv

f = open('new datasets/youtube_converted.csv', encoding='utf-8')
r = csv.reader(f)
header = next(r)
print('Header:', header)
normal = 0
toxic = 0
for row in r:
    lab = row[1]
    if lab == '0':
        normal += 1
    else:
        toxic += 1
print(f'Total rows: {normal + toxic}')
print(f'Normal (0): {normal} ({normal/(normal+toxic)*100:.1f}%)')
print(f'Toxic (1): {toxic} ({toxic/(normal+toxic)*100:.1f}%)')
f.close()