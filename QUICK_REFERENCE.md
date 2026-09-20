# Quick Reference Guide

## Files Overview

| File | Purpose |
|------|---------|
| `kaggle_transformers_train.py` | Main training script - downloads data and trains model |
| `inference.py` | Use trained model to make predictions |
| `SETUP_INSTRUCTIONS.md` | Detailed setup and troubleshooting |
| `QUICK_REFERENCE.md` | This file - quick commands |

---

## Quick Start (5 minutes)

```bash
# 1. Install dependencies
pip install kaggle datasets transformers torch scikit-learn pandas accelerate --break-system-packages

# 2. Setup Kaggle API (one-time)
# - Download from https://www.kaggle.com/settings/account
# - Save to ~/.kaggle/kaggle.json
# - chmod 600 ~/.kaggle/kaggle.json

# 3. Train model
python kaggle_transformers_train.py

# 4. Make predictions
python inference.py --text "I love this product!"
```

---

## Training Script Options

### Adjust Training Parameters

Edit these variables in `kaggle_transformers_train.py`:

```python
# Change model
model_name = "roberta-base"  # or bert-base-uncased, albert-base-v2

# Change training epochs
num_train_epochs = 5  # default is 3

# Change batch size (smaller = slower but less memory)
per_device_train_batch_size = 8  # default is 16

# Change learning rate
learning_rate = 1e-5  # default is 2e-5 (smaller = slower learning)

# Change max sequence length
max_length = 512  # default is 256
```

### Override Column Detection

If auto-detection fails:

```python
# In prepare_training_data() function, change:
text_col = "post"        # your text column name
label_col = "sentiment"  # your label column name
```

---

## Inference Usage

### Command Line

```bash
# Single prediction
python inference.py --text "Love this!"

# Interactive mode
python inference.py --interactive

# Custom model path
python inference.py --text "Text here" --model ./my_model

# Confidence threshold
python inference.py --text "Maybe..." --threshold 0.8
```

### Python Code

```python
from inference import SocialMediaClassifier

# Load model
classifier = SocialMediaClassifier(model_dir="./social_media_model")

# Single prediction
result = classifier.predict_single("Great product!")
print(result)  # {'label': 'POSITIVE', 'score': 0.95}

# Batch predictions
texts = ["Love it!", "Hate it!", "It's okay"]
results = classifier.predict_batch(texts)

# With confidence threshold
result = classifier.predict_with_confidence("Maybe...", threshold=0.7)
```

---

## Model Selection Guide

| Model | Best For | Speed | Memory | Accuracy |
|-------|----------|-------|--------|----------|
| **distilbert-base-uncased** | Default choice | ⚡⚡⚡ | Low | ⭐⭐⭐ |
| albert-base-v2 | Very fast inference | ⚡⚡⚡⚡ | Very Low | ⭐⭐ |
| bert-base-uncased | Balanced | ⚡⚡ | Medium | ⭐⭐⭐ |
| roberta-base | Better accuracy | ⚡ | Medium-High | ⭐⭐⭐⭐ |
| xlnet-base-cased | Best accuracy | 🐢 | High | ⭐⭐⭐⭐⭐ |

```python
# Change in train_model() function:
train_model(tokenized_dataset, num_labels, model_name="roberta-base")
```

---

## Common Use Cases

### 1. Sentiment Analysis (Social Media Posts)

```python
text = "This product is amazing! Best purchase ever! 🎉"
result = classifier.predict_single(text)
# Expected: {'label': 'POSITIVE', 'score': 0.98}
```

### 2. Batch Classification (CSV File)

```python
import pandas as pd
from inference import SocialMediaClassifier

classifier = SocialMediaClassifier()

# Load data
df = pd.read_csv('social_posts.csv')

# Predict
predictions = classifier.predict_batch(df['text'].tolist())

# Add to dataframe
df['sentiment'] = [p['label'] for p in predictions]
df['confidence'] = [p['score'] for p in predictions]

# Save
df.to_csv('predictions.csv', index=False)
print("✅ Saved to predictions.csv")
```

### 3. Filter by Confidence

```python
results = classifier.predict_batch(texts)

# Keep only high-confidence predictions
high_confidence = [
    (text, result) 
    for text, result in zip(texts, results)
    if result['score'] > 0.8
]

print(f"High confidence predictions: {len(high_confidence)}")
```

### 4. Save Predictions to Database

```python
from inference import SocialMediaClassifier
import sqlite3
from datetime import datetime

classifier = SocialMediaClassifier()

# Connect to SQLite database
conn = sqlite3.connect('predictions.db')
cursor = conn.cursor()

# Create table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY,
        text TEXT,
        label TEXT,
        confidence REAL,
        timestamp DATETIME
    )
''')

# Add predictions
for text in social_media_texts:
    result = classifier.predict_single(text)
    cursor.execute('''
        INSERT INTO predictions (text, label, confidence, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (text, result['label'], result['score'], datetime.now()))

conn.commit()
conn.close()
print("✅ Saved to predictions.db")
```

---

## Troubleshooting

### Problem: Out of Memory Error

```python
# Solution: Reduce batch size and max length
per_device_train_batch_size = 8   # was 16
per_device_eval_batch_size = 8    # was 16
max_length = 128                  # was 256
```

### Problem: Model not using GPU

```bash
# Check GPU status
python -c "import torch; print(torch.cuda.is_available())"

# If False, install correct CUDA:
# https://pytorch.org/get-started/locally/
```

### Problem: Kaggle API not found

```bash
# Verify setup
kaggle datasets list

# If error, reconfigure:
# Download from https://www.kaggle.com/settings/account
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Problem: Column detection failed

```python
# Find available columns:
import pandas as pd
df = pd.read_csv('./social_media_data/YOUR_FILE.csv')
print(df.columns.tolist())
print(df.head())

# Then update in prepare_training_data():
text_col = "actual_column_name"
label_col = "actual_label_name"
```

---

## Performance Optimization

### For Faster Training

```python
# Use smaller model
model_name = "distilbert-base-uncased"  # smaller model

# Use lower precision
# Add to training_args:
fp16=True  # Use 16-bit floating point
```

### For Better Accuracy

```python
# Use larger model
model_name = "roberta-base"

# More training
num_train_epochs = 5

# Better learning rate
learning_rate = 1e-5

# More warmup
warmup_steps = 500
```

### For Inference Speed

```python
# Use smaller model
classifier = SocialMediaClassifier(model_dir="./social_media_model")

# Batch processing is faster than individual
results = classifier.predict_batch(texts)  # ✅ Better
# vs
results = [classifier.predict_single(t) for t in texts]  # ❌ Slower
```

---

## Exporting the Model

### To Hugging Face Hub

```bash
pip install huggingface-hub
huggingface-cli login

python -c "
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained('./social_media_model')
tokenizer = AutoTokenizer.from_pretrained('./social_media_model')

model.push_to_hub('your-username/social-media-classifier')
tokenizer.push_to_hub('your-username/social-media-classifier')
"
```

### To ONNX (for production)

```bash
pip install optimum onnxruntime

python -c "
from optimum.onnxruntime import AutoOptimizationConfig, ORTModelForSequenceClassification
from transformers import AutoTokenizer

model_id = './social_media_model'
optimization_config = AutoOptimizationConfig.O3()

model = ORTModelForSequenceClassification.from_pretrained(
    model_id,
    export=True,
    optimization_config=optimization_config
)
model.save_pretrained('./social_media_model_onnx')
"
```

---

## Next Steps

- ✅ Run training: `python kaggle_transformers_train.py`
- ✅ Test predictions: `python inference.py`
- ✅ Integrate into your app
- ✅ Monitor performance metrics
- ✅ Fine-tune with more data as needed
