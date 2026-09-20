"""
Train cyberbullying detection model using local gdrive datasets.

Supports: twitter_parsed_dataset.csv, youtube_parsed_dataset.csv, toxicity_parsed_dataset.csv
All datasets have 'Text' column and 'oh_label' (binary: 0=clean, 1=bullying/toxic).
No Kaggle API required - uses local CSV files only.
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
# 1. CONFIGURATION
# ========================================
DATASET_CONFIGS = {
    "twitter": {
        "path": "twitter_parsed_dataset.csv",
        "text_col": "Text",
        "label_col": "oh_label",
        "label_names": {0: "clean", 1: "bullying"},
    },
    "youtube": {
        "path": "youtube_parsed_dataset.csv",
        "text_col": "Text",
        "label_col": "oh_label",
        "label_names": {0: "clean", 1: "bullying"},
    },
    "toxicity": {
        "path": "toxicity_parsed_dataset.csv",
        "text_col": "Text",
        "label_col": "oh_label",
        "label_names": {0: "clean", 1: "toxic"},
    },
}

# ========================================
# 2. LOAD LOCAL DATASET
# ========================================
def load_dataset(dataset_name, data_dir="gdrive-files/Automatic Detection of Cyberbullying Behaviour on Social Media Using Hybrid Transformers and Deep Learning Models_DATASETS"):
    """Load a specific cyberbullying dataset."""
    config = DATASET_CONFIGS[dataset_name]
    csv_path = os.path.join(data_dir, config["path"])
    
    print(f"Loading {dataset_name} dataset...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} examples")
    print(f"Columns: {df.columns.tolist()}")
    
    return df, config


# ========================================
# 3. EXPLORE DATA
# ========================================
def explore_data(df, config):
    """Explore dataset structure."""
    print(f"\nDistribution of {config['label_col']}:")
    dist = df[config['label_col']].value_counts()
    for label_id, count in dist.items():
        label_name = config['label_names'].get(label_id, f"class_{label_id}")
        pct = count / len(df) * 100
        print(f"  {label_name}: {label_id} ({count} samples, {pct:.1f}%)")
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Text column: '{config['text_col']}'")
    print(f"Label column: '{config['label_col']}'")
    
    return df


# ========================================
# 4. PREPARE DATA FOR TRAINING
# ========================================
def prepare_training_data(df, config):
    """Prepare data with column auto-detection fallback."""
    print("\nPreparing data for training...")
    
    # Use configured columns, fallback to auto-detection
    text_col = config["text_col"]
    label_col = config["label_col"]
    
    # Verify columns exist
    if text_col not in df.columns:
        # Auto-detect
        for col in df.columns:
            if col.lower() in ['text', 'content', 'post', 'tweet', 'message']:
                text_col = col
                break
    
    if label_col not in df.columns:
        # Auto-detect - use last column
        label_col = df.columns[-1]
    
    print(f"Using text column: '{text_col}'")
    print(f"Using label column: '{label_col}'")
    
    # Remove missing values
    df = df.dropna(subset=[text_col, label_col])
    print(f"Removed NaN values. New shape: {df.shape}")
    
    # Encode labels (already binary 0/1, but ensure numeric)
    if df[label_col].dtype == 'object':
        label_encoder = LabelEncoder()
        df['encoded_label'] = label_encoder.fit_transform(df[label_col])
    else:
        # Already numeric (0/1), ensure it's int
        df['encoded_label'] = df[label_col].astype(int)
    
    num_labels = 2  # Binary classification
    
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
# 5. TOKENIZE DATASET
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
# 6. COMPUTE METRICS
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
# 7. TRAIN MODEL
# ========================================
def train_model(tokenized_dataset, num_labels, model_name="distilbert-base-uncased", output_dir="./cyberbullying_model"):
    """Fine-tune transformer model for cyberbullying detection."""
    print("\nTraining cyberbullying detection model...")
    
    # Load pretrained model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments - optimized for quicker training
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,  # Reduced from 3 for quicker training
        per_device_train_batch_size=32,  # Increased batch size
        per_device_eval_batch_size=32,
        warmup_steps=5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        load_best_model_at_end=False,
        seed=42,
    )
    
    # Create trainer
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
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nModel saved to {output_dir}")
    
    return trainer, model


# ========================================
# 8. MAIN PIPELINE - TRAIN ON ALL DATASETS
# ========================================
def train_on_all_datasets():
    """Train model on each available dataset."""
    print("Cyberbullying Detection: Training on All Datasets")
    
    for dataset_name in DATASET_CONFIGS.keys():
        print(f"\nTraining on {dataset_name} dataset")
        
        try:
            # Load dataset
            df, config = load_dataset(dataset_name)
            
            # Explore data
            df = explore_data(df, config)
            
            # Prepare data
            train_dataset, test_dataset, num_labels = prepare_training_data(df, config)
            
            # Create DatasetDict
            from datasets import DatasetDict
            dataset_dict = DatasetDict({
                'train': train_dataset,
                'test': test_dataset
            })
            
            # Tokenize
            tokenized_dataset, tokenizer = tokenize_dataset(
                dataset_dict,
                model_name="distilbert-base-uncased"
            )
            
            # Train model
            output_dir = f"./cyberbullying_model_{dataset_name}"
            trainer, model = train_model(
                tokenized_dataset,
                num_labels,
                model_name="distilbert-base-uncased",
                output_dir=output_dir
            )
            
            print(f"{dataset_name} training completed!\n")
            
        except Exception as e:
            print(f"Error training on {dataset_name}: {e}\n")
    
    print("All dataset training completed!")


# ========================================
# 9. MAIN PIPELINE - SINGLE DATASET
# ========================================
def train_single_dataset(dataset_name):
    """Train model on a single specified dataset."""
    print(f"Training on {dataset_name} dataset")
    
    try:
        # Load dataset
        df, config = load_dataset(dataset_name)
        
        # Explore data
        df = explore_data(df, config)
        
        # Prepare data
        train_dataset, test_dataset, num_labels = prepare_training_data(df, config)
        
        # Create DatasetDict
        from datasets import DatasetDict
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })
        
        # Tokenize
        tokenized_dataset, tokenizer = tokenize_dataset(
            dataset_dict,
            model_name="distilbert-base-uncased"
        )
        
        # Train model
        output_dir = f"./cyberbullying_model_{dataset_name}"
        trainer, model = train_model(
            tokenized_dataset,
            num_labels,
            model_name="distilbert-base-uncased",
            output_dir=output_dir
        )
        
        print(f"{dataset_name} training completed!")
        print(f"Model saved to: {output_dir}\n")
        
    except Exception as e:
        print(f"Error: {e}\n")


# ========================================
# 10. MAIN ENTRY POINT
# ========================================
if __name__ == "__main__":
    import sys
    
    print("Cyberbullying Detection Model Training")
    print("Datasets available:")
    for name in DATASET_CONFIGS.keys():
        print(f"  - {name:10s} ({DATASET_CONFIGS[name]['path']})")
    
    print("\nUsage:")
    print("  python train_cyberbullying.py            # Train on all datasets")
    print("  python train_cyberbullying.py twitter    # Train on Twitter only")
    print("  python train_cyberbullying.py youtube    # Train on YouTube only")
    print("  python train_cyberbullying.py toxicity   # Train on Toxicity only")
    print()
    
    if len(sys.argv) > 1:
        dataset_arg = sys.argv[1].lower()
        if dataset_arg in DATASET_CONFIGS:
            train_single_dataset(dataset_arg)
        else:
            print(f"Unknown dataset: {dataset_arg}")
            print("Available:", list(DATASET_CONFIGS.keys()))
    else:
        # Train on all datasets by default
        train_on_all_datasets()