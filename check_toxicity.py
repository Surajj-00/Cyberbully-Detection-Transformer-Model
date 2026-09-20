import pandas as pd

csv_path = "gdrive-files/Automatic Detection of Cyberbullying Behaviour on Social Media Using Hybrid Transformers and Deep Learning Models_DATASETS/toxicity_parsed_dataset.csv"
df = pd.read_csv(csv_path)

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nFirst 3 rows:")
print(df.head(3))

last_col = df.columns[-1]
print(f"\nLast column '{last_col}' value counts:")
print(df[last_col].value_counts())