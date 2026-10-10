import csv

f = open('new datasets/Twitter_converted.csv', encoding='utf-8')
r = csv.reader(f)
header = next(r)
print('Header:', header)
normal = 0
sexism = 0
racism = 0
for row in r:
    lab = row[1]
    if lab == 'normal':
        normal += 1
    elif lab == 'sexism':
        sexism += 1
    elif lab == 'racism':
        racism += 1
print(f'Total rows: {normal + sexism + racism}')
print(f'Normal: {normal} ({normal/(normal+sexism+racism)*100:.1f}%)')
print(f'Sexism: {sexism} ({sexism/(normal+sexism+racism)*100:.1f}%)')
print(f'Racism: {racism} ({racism/(normal+sexism+racism)*100:.1f}%)')
f.close()