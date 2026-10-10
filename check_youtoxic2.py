import csv

f = open('new datasets/YouToxic_converted.csv', encoding='utf-8')
r = csv.reader(f)
header = next(r)
print('Header:', header)
threats = 0
hates = 0
racist = 0
total = 0
for row in r:
    total += 1
    threats += 1 if row[1] == 'True' else 0
    hates += 1 if row[2] == 'True' else 0
    racist += 1 if row[3] == 'True' else 0
print(f'Total rows: {total}')
print(f'Threats: {threats} ({threats/total*100:.1f}%)')
print(f'Hatespeech: {hates} ({hates/total*100:.1f}%)')
print(f'Racist: {racist} ({racist/total*100:.1f}%)')
f.close()