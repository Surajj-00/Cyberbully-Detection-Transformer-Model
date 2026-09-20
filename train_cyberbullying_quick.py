"""
Quick cyberbullying detection demo - uses small subsets for fast training.

Uses reduced samples from gdrive datasets for quick demonstration (~500 samples each).
Complete datasets available in train_cyberbullying.py.
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
# 1. CONFIGURATION - QUICK DEMO SAMPLES
# ========================================
DATASET_CONFIGS = {
    "twitter": {
        "path": "twitter_parsed_dataset.csv",
        "text_col": "Text",
        "label_col": "oh_label",
        "label_names": {0: "clean", 1: "bullying"},
        "sample_size": 500,  # Small sample for quick demo
    },
    "youtube": {
        "path": "youtube_parsed_dataset.csv",
        "text_col": "Text",
        "label_col": "oh_label",
        "label_names": {0: "clean", 1: "bullying"},
        "sample_size": 500,
    },
    "toxicity": {
        "path": "toxicity_parsed_dataset.csv",
        "text_col": "Text",
        "label_col": "oh_label",
        "label_names": {0: "clean", 1: "toxic"},
        "sample_size": 500,
    },
}

# ========================================
# 2. LOAD DATASET WITH SAMPLE
# ========================================
def load_dataset_quick(dataset_name, data_dir="gdrive-files/Automatic Detection of Cyberbullying Behaviour on Social Media Using Hybrid Transformers and Deep Learning Models_DATASETS"):
    """Load a specific cyberbullying dataset with reduced sample."""
    config = DATASET_CONFIGS[dataset_name]
    csv_path = os.path.join(data_dir, config["path"])
    
    print(f"Loading {dataset_name} dataset (sample={config['sample_size']})...")
    df = pd.read_csv(csv_path)
    
    # Take sample if dataset is larger than sample_size
    if len(df) > config["sample_size"]:
        df = df.sample(n=config["sample_size"], random_state=42)
        print(f"Using sample of {config['sample_size']} from {len(df) + (len(df) - config['sample_size'])} total")
    
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
    
    text_col = config["text_col"]
    label_col = config["label_col"]
    
    if text_col not in df.columns:
        for col in df.columns:
            if col.lower() in ['text', 'content', 'post', 'tweet', 'message']:
                text_col = col
                break
    
    if label_col not in df.columns:
        label_col = df.columns[-1]
    
    print(f"Using text column: '{text_col}'")
    print(f"Using label column: '{label_col}'")
    
    df = df.dropna(subset=[text_col, label_col])
    print(f"Removed NaN values. New shape: {df.shape}")
    
    if df[label_col].dtype == 'object':
        label_encoder = LabelEncoder()
        df['encoded_label'] = label_encoder.fit_transform(df[label_col])
    else:
        df['encoded_label'] = df[label_col].astype(int)
    
    num_labels = 2
    
    from sklearn.model_selection import train_test_split
    
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['encoded_label']
    )
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
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
# 7. TRAIN MODEL (QUICK - 1 EPOCH)
# ========================================
def train_model_quick(tokenized_dataset, num_labels, model_name="distilbert-base-uncased", output_dir="./cyberbullying_model_quick"):
    """Quick training - 1 epoch for demonstration."""
    print("\nTraining cyberbullying detection model (1 epoch)...")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,  # Single epoch for quick demo
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=False,
        seed=42,
    )
    
    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    print("\nFinal Evaluation:")
    metrics = trainer.evaluate()
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nModel saved to {output_dir}")
    
    return trainer, model


# ========================================
# 8. MAIN PIPELINE - QUICK DEMO
# ========================================
def quick_demo(dataset_name):
    """Run quick training demo on specified dataset."""
    print(f"="*60)
    print(f"Quick Cyberbullying Demo: {dataset_name.upper()}")
    print("="*60)
    
    try:
        df, config = load_dataset_quick(dataset_name)
        df = explore_data(df, config)
        train_dataset, test_dataset, num_labels = prepare_training_data(df, config)
        
        from datasets import DatasetDict
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })
        
        tokenized_dataset, tokenizer = tokenize_dataset(dataset_dict)
        trainer, model = train_model_quick(tokenized_dataset, num_labels)
        
        print(f"\n✅ {dataset_name} quick demo completed!")
        print(f"Model saved to: {output_dir}\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}\n")


# ========================================
# 9. MAIN PIPELINE - TRAIN ALL QUICK
# ========================================
def quick_demo_all():
    """Run quick demo on all datasets."""
    print("Quick Cyberbullying Demo: All Datasets")
    
    for dataset_name in DATASET_CONFIGS.keys():
        print(f"\n{dataset_name} dataset")
        quick_demo(dataset_name)
    
    print("\nAll quick demos completed!")


# ========================================
# 10. MAIN ENTRY POINT
# ========================================
if __name__ == "__main__":
    import sys
    
    print("Quick Cyberbullying Detection Demo")
    print("Datasets available (with sample sizes):")
    for name, config in DATASET_CONFIGS.items():
        total = config['sample_size'] if config['sample_size'] else 'full'
        print(f"  - {name:10s} (sample: {total})")
    
    print("\nUsage:")
    print("  python train_cyberbullying_quick.py            # Quick demo on all datasets")
    print("  python train_cyberbullying_quick.py twitter    # Twitter quick demo")
    print("  python train_cyberbullying_quick.py youtube    # YouTube quick demo")
    print("  python train_cyberbullying_quick.py toxicity   # Toxicity quick demo")
    print()
    
    if len(sys.argv) > 1:
        dataset_arg = sys.argv[1].lower()
        if dataset_arg in DATASET_CONFIGS:
            quick_demo(dataset_arg)
        else:
            print(f"Unknown dataset: {dataset_arg}")
            print("Available:", list(DATASET_CONFIGS.keys()))
    else:
        quick_demo_all()