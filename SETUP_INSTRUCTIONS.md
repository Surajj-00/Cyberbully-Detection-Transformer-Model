# Setup Instructions for Kaggle Transformers Training

## Step 1: Install Dependencies

```bash
pip install kaggle datasets transformers torch scikit-learn pandas accelerate --break-system-packages
```

## Step 2: Configure Kaggle API

### 2a. Go to Kaggle Account Settings
- Visit https://www.kaggle.com/settings/account
- Click "Create New API Token"
- This downloads `kaggle.json`

### 2b. Set Up Kaggle Configuration

**On Linux/Mac:**
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**On Windows:**
```
Create folder: C:\Users\<YourUsername>\.kaggle
Move kaggle.json there
```

## Step 3: Run the Training Script

```bash
python kaggle_transformers_train.py
```

The script will:
1. ✅ Download the dataset from Kaggle
2. 📊 Explore the dataset structure
3. 🔧 Automatically detect text and label columns
4. 📈 Prepare and tokenize data
5. 🚀 Fine-tune a DistilBERT model
6. 💾 Save the trained model to `./social_media_model`

## Important: Column Name Detection

The script auto-detects columns, but if it fails, edit these lines in the script:

```python
text_col = 'your_text_column_name'      # e.g., 'post', 'tweet', 'content'
label_col = 'your_label_column_name'    # e.g., 'sentiment', 'category', 'label'
```

## Using the Trained Model

### Make Predictions
```python
from transformers import pipeline

classifier = pipeline("text-classification", model="./social_media_model")
result = classifier("Great product!")
print(result)
```

### Load for Fine-Tuning Again
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("./social_media_model")
tokenizer = AutoTokenizer.from_pretrained("./social_media_model")
```

### Push to Hugging Face Hub (Optional)
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

## Troubleshooting

### Kaggle API Error
- Verify `~/.kaggle/kaggle.json` exists and has correct permissions
- Run: `kaggle datasets list` to test

### Out of Memory Error
- Reduce `per_device_train_batch_size` from 16 to 8
- Reduce `max_length` from 256 to 128

### Dataset Column Mismatch
- Run: `python -c "import pandas as pd; df = pd.read_csv('./social_media_data/YOUR_FILE.csv'); print(df.columns)"`
- Update `text_col` and `label_col` in the script

### GPU Not Being Used
- Check: `python -c "import torch; print(torch.cuda.is_available())"`
- Install correct CUDA drivers for your GPU

## Model Options

Replace `"distilbert-base-uncased"` with other models:

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| distilbert-base-uncased | 268MB | Fast | Good |
| bert-base-uncased | 440MB | Medium | Better |
| roberta-base | 498MB | Medium | Better |
| albert-base-v2 | 45MB | Very Fast | Good |
| xlnet-base-cased | 340MB | Slower | Best |

Example:
```python
train_model(tokenized_dataset, num_labels, model_name="roberta-base")
```

## Expected Output

```
==============================================================
🤖 Kaggle Social Media Dataset + Transformers Training
==============================================================
📥 Downloading dataset from Kaggle...
✅ Dataset downloaded to ./social_media_data

📊 Exploring dataset...
Dataset shape: (10000, 5)

🔧 Preparing data for training...
Text column: 'post'
Label column: 'sentiment'
Train set: 8000 samples
Test set: 2000 samples

🔤 Tokenizing dataset...
✅ Tokenization complete

🚀 Training transformer model...
[Training progress...]

📈 Final Evaluation:
  eval_loss: 0.1234
  eval_accuracy: 0.9456
  eval_f1: 0.9412
  
✅ Model saved to ./social_media_model
```



---------------------------------------------------------

# Run py code in virtual environment

python -m venv .venv
>> .\.venv\Scripts\activate


--------------------------------------------------------
