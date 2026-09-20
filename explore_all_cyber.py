import zipfile
import os
import pandas as pd

# Check cyberbullying_tweets.csv.zip
zip_path = "gdrive-files/Kaggle Dataset-20260723T140242Z-1-001/Kaggle Dataset/cyberbullying_tweets.csv.zip"
extract_to = "gdrive-files/cyberbullying_tweets_Test"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    print("cyberbullying_tweets.zip contents:")
    for name in zip_ref.namelist()[:5]:
        print(" ", name)
    
    os.makedirs(extract_to, exist_ok=True)
    zip_ref.extractall(extract_to)
    
    # Read the CSV
    for root, dirs, files in os.walk(extract_to):
        for f in files:
            if f.endswith('.csv'):
                full_path = os.path.join(root, f)
                df = pd.read_csv(full_path)
                rel_path = os.path.relpath(full_path, extract_to)
                print(f"\n{rel_path}:")
                print(f"  Shape: {df.shape}")
                print(f"  Columns: {df.columns.tolist()}")
                print(f"  Cyberbullying type distribution:")
                if 'cyberbullying_type' in df.columns:
                    print(df['cyberbullying_type'].value_counts())
                elif 'label' in df.columns:
                    print(df['label'].value_counts())
                else:
                    print(df.iloc[:, -1].value_counts())
                print(f"  Sample rows:")
                print(df.head(2).to_string())