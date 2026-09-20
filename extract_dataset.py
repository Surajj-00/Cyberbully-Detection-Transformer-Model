import zipfile
import os
import shutil

zip_path = "D:/Developer-Mood/transformers-algo-ml/gdrive-files/Automatic Detection of Cyberbullying Behaviour on Social Media Using Hybrid Transformers and Deep Learning Models_DATASETS.zip"
extract_to = "D:/Developer-Mood/transformers-algo-ml/gdrive-files/Automatic Detection of Cyberbullying Behaviour on Social Media Using Hybrid Transformers and Deep Learning Models_DATASETS"

# Unzip the file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    print("Extracting:", zip_ref.namelist()[:5])
    zip_ref.extractall(extract_to)

# List extracted files
extracted_path = Path(extract_to)
csv_files = list(extracted_path.rglob("*.csv"))
print(f"\nFound {len(csv_files)} CSV files:")
for f in csv_files:
    print(f"  {f.relative_to(extract_to)}")