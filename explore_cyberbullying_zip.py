import zipfile
import os
import pandas as pd

# Check Cyberbullying Classification datasets
zip_path = "gdrive-files/Kaggle Dataset-20260723T140242Z-1-001/Kaggle Dataset/Cyberbullying Classification datasets ( Explicit Cyberbullying).zip"
extract_to = "gdrive-files/Cyberbullying_Classification_Test"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    print("Cyberbullying Classification zip contents:")
    for name in zip_ref.namelist()[:15]:
        print(" ", name)
    
    # Extract
    os.makedirs(extract_to, exist_ok=True)
    zip_ref.extractall(extract_to)
    
    # List CSV files
    print(f"\nExtracted files. Finding CSVs...")
    csv_count = 0
    for root, dirs, files in os.walk(extract_to):
        for f in files:
            if f.endswith('.csv'):
                csv_count += 1
                full_path = os.path.join(root, f)
                df = pd.read_csv(full_path)
                rel_path = os.path.relpath(full_path, extract_to)
                print(f"\n{rel_path}:")
                print(f"  Shape: {df.shape}")
                print(f"  Columns: {df.columns.tolist()}")
                print(f"  First 2 rows:")
                print(df.head(2).to_string())
    
    print(f"\nTotal CSV files found: {csv_count}")