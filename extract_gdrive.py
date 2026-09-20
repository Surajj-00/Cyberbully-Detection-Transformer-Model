import zipfile
import os

zip_path = "gdrive-files/Automatic Detection of Cyberbullying Behaviour on Social Media Using Hybrid Transformers and Deep Learning Models_DATASETS.zip"
extract_to = "gdrive-files/Automatic Detection of Cyberbullying Behaviour on Social Media Using Hybrid Transformers and Deep Learning Models_DATASETS"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)
    print("Extracted files:")
    for name in zip_ref.namelist()[:20]:
        print(" ", name)
    
    # List CSV files
    csv_files = []
    for root, dirs, files in os.walk(extract_to):
        for f in files:
            if f.endswith('.csv'):
                csv_files.append(os.path.join(root, f))
    
    print(f"\nTotal CSV files: {len(csv_files)}")
    for f in csv_files:
        print(" ", os.path.relpath(f, extract_to))