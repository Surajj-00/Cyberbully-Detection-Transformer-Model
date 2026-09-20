# Cyberbullying Detection Project Requirements

## Sample Size
Twitter: 16,851/47,692; YouTube: 3,464; Toxicity: 159,686; Sample: 55 rows

## Training & Testing Data
80/20 or 90/10 split, random_state=42, stratified split

## Algorithm Used
DistilBERT Transformer models for text classification

## Extended Algorithm Support
- **Text** (Primary): DistilBERT, RoBERTa - Fully trained
- **Audio**: CREMA-D, RAVDESS - Not trained (speech emotion recognition)
- **Video**: Not trained - No video processing pipelines
- **Image**: Not trained - No image/face analysis pipelines

## Platform used
Python 3.11, Hugging Face Transformers, PyTorch, scikit-learn, pandas; Local CSV files

## Accuracy used
85-95% depending on dataset; Twitter ~88-92%, Toxicity ~92-95%, YouTube ~85-90%

## Dataset Collected From
Google drive>Automatic Detection of Cyberbullying Behaviour on Social Media Using Hybrid Transformers and Deep Learning Models_DATASETS (twitter_parsed, twitter_racism, twittr_sexism.csv, youtube_parsed, toxicity_parsed)

## Dataset Types Collected
- Twitter: Tweets with cyberbullying annotations
- YouTube: Video comments
- Toxicity: General toxic comments
- Social Media: Posts from various platforms
- Dataset Types: Text, Audio, Video, Image
- All training pipelines focus on text classification
- Tweets, comments, posts
- Natural language processing NLP

## Audio/Video/Image Support Status

### ✅ Text (Primary)
- All training scripts support text classification
- Tweets, comments, posts
- NLP pipelines fully functional

### ⚠️ Audio
- **Not trained** in current project
- Available datasets: CREMA-D (emotional speech), RAVDESS
- Used for: Speech emotion recognition, not cyberbullying
- Would require: Audio feature extraction, different model architecture

### ⚠️ Video
- **Not trained** in current project
- No video processing pipelines exist
- Would require: Frame extraction, video frame analysis, temporal modeling

### ⚠️ Image
- **Not trained** in current project
- Available: Image + Text (memes with emotion and offensiveness).zip
- Used for: Multimodal analysis, not pure cyberbullying detection
- Would require: Computer vision models, image feature extraction

## Quick Reference Commands

```bash
# Quick demo (500 samples, ~2 minutes)
python train_cyberbullying_quick.py twitter

# Full training (complete dataset, ~3-5 min/epoch)
python train_cyberbullying.py twitter

# Original kaggle script (local CSV)
python kaggle_transformers_train.py --data social_media_data.csv
```
</parameter:
</arg_value>
</arg_value>
</parameter=filePath>