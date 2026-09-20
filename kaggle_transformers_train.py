"""
Train a transformer model on social media dataset for cyberbullying detection.

This script supports local CSV datasets without requiring Kaggle API.
Works with the gdrive social media datasets extracted from:
  Automatic Detection of Cyberbullying Behaviour on Social Media Using Hybrid Transformers and Deep Learning Models_DATASETS
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

# ========================================
# 1. LOAD LOCAL DATASET
# ========================================
def load_local_dataset(csv_path="./social_media_data.csv"):
    """Load a local CSV dataset for training."""
    print(f"Loading local dataset from: {csv_path}")
    
    data_path = Path(csv_path)
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} examples")
    
    return df


# ========================================
# 2. EXPLORE DATA
# ========================================
def explore_data(df):
    """Load and explore the dataset structure."""
    print("\nExploring dataset...")
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumn names:\n{df.columns.tolist()}")
    print(f"\nFirst 3 rows:\n{df.head(3)}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    return df


# ========================================
# 3. PREPARE DATA FOR TRAINING
# ========================================
def prepare_training_data(df):
    """
    Prepare data for transformer training.
    Auto-detects text and label columns.
    Adjust column names based on your dataset structure.
    """
    print("\nPreparing data for training...")
    
    # Try to auto-detect text and label columns
    text_col = None
    label_col = None
    
    # Common text column names (case-insensitive matching)
    text_variations = ['text', 'content', 'post', 'tweet', 'message', 'description', 'comment', 'Text']
    label_variations = ['label', 'sentiment', 'category', 'class', 'target', 'oh_label', 'label_id']
    
    # Check column names (case-insensitive)
    df_columns_lower = [c.lower() for c in df.columns]
    
    # Find text column
    for col in df.columns:
        if col in text_variations or col.lower() in text_variations:
            text_col = col
            break
    
    # Find label column
    for col in df.columns:
        if col in label_variations or col.lower() in [v.lower() for v in label_variations]:
            label_col = col
            break
    
    # Fallback: if no match, use first text-like and last column
    if not text_col:
        text_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    if not label_col:
        label_col = df.columns[-1]
    
    print(f"Detected text column: '{text_col}'")
    print(f"Detected label column: '{label_col}'")
    
    # Remove missing values
    df = df.dropna(subset=[text_col, label_col])
    print(f"Removed NaN values. New shape: {df.shape}")
    
    # Encode labels if they're strings
    if df[label_col].dtype == 'object':
        label_encoder = LabelEncoder()
        df['encoded_label'] = label_encoder.fit_transform(df[label_col])
        print(f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
        num_labels = len(label_encoder.classes_)
    else:
        # Numerical labels - check if binary or multi-class
        unique_labels = df[label_col].nunique()
        if unique_labels <= 2:
            df['encoded_label'] = df[label_col]
            num_labels = 2
        else:
            label_encoder = LabelEncoder()
            df['encoded_label'] = label_encoder.fit_transform(df[label_col])
            num_labels = len(label_encoder.classes_)
            print(f"Multi-class classification: {num_labels} classes")
            print(f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    # Create train/test split
    from sklearn.model_selection import train_test_split
    
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['encoded_label']
    )
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    # Create HF datasets
    train_dataset = Dataset.from_dict({
        'text': train_df[text_col].tolist(),
        'label': train_df['encoded_label'].tolist()
    })
    
    test_dataset = Dataset.from_dict({
        'text': test_df[text_col].tolist(),
        'label': test_df['encoded_label'].tolist()
    })
    
    return train_dataset, test_dataset, num_labels


# ========================================
# 4. TOKENIZE DATASET
# ========================================
def tokenize_dataset(dataset, model_name="distilbert-base-uncased"):
    """Tokenize the dataset."""
    print("\nTokenizing dataset...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,
            padding="max_length"
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    
    print("Tokenization complete")
    return tokenized_dataset, tokenizer


# ========================================
# 5. COMPUTE METRICS
# ========================================
def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted", zero_division=0),
        "precision": precision_score(labels, predictions, average="weighted", zero_division=0),
        "recall": recall_score(labels, predictions, average="weighted", zero_division=0),
    }


# ========================================
# 6. TRAIN MODEL
# ========================================
def train_model(tokenized_dataset, num_labels, model_name="distilbert-base-uncased"):
    """Fine-tune transformer model."""
    print("\nTraining transformer model...")
    
    # Load pretrained model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Simplified training arguments
    training_args = TrainingArguments(
        output_dir="./social_media_model",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=False,
        seed=42,
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Create trainer (without tokenizer argument for newer transformers)
    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    trainer.train()
    
    # Evaluate
    print("\nFinal Evaluation:")
    metrics = trainer.evaluate()
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save model
    trainer.save_model("./social_media_model")
    tokenizer.save_pretrained("./social_media_model")
    print("\nModel saved to ./social_media_model")
    
    return trainer, model


# ========================================
# 7. MAIN PIPELINE
# ========================================
def main():
    """Run the complete pipeline."""
    print("="*60)
    print("Social Media Text Classification Training")
    print("="*60)
    
    import argparse
    parser = argparse.ArgumentParser(description="Train transformer model")
    parser.add_argument(
        "--data",
        type=str,
        default="./social_media_data.csv",
        help="Path to local CSV dataset (default: ./social_media_data.csv)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="distilbert-base-uncased",
        help="Pretrained model name"
    )
    
    args = parser.parse_args()
    
    try:
        # Step 1: Load local dataset
        print(f"Loading dataset: {args.data}")
        df = load_local_dataset(args.data)
        
        # Step 2: Explore data
        df = explore_data(df)
        
        # Step 3: Prepare data
        train_dataset, test_dataset, num_labels = prepare_training_data(df)
        
        # Create DatasetDict
        from datasets import DatasetDict
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })
        
        # Step 4: Tokenize
        tokenized_dataset, tokenizer = tokenize_dataset(
            dataset_dict,
            model_name=args.model
        )
        
        # Step 5: Train
        trainer, model = train_model(
            tokenized_dataset,
            num_labels,
            model_name=args.model
        )
        
        print("\n" + "="*60)
        print("Training completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()