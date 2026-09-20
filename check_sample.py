import pandas as pd

# Check cyberbullying_sample_data.csv
df = pd.read_csv('claude-provided-files/cyberbullying_sample_data.csv')
print('Sample data:')
print(f'Shape: {df.shape}')
print(f'Columns: {df.columns.tolist()}')
print(f'Label distribution:')
print(df['label'].value_counts())