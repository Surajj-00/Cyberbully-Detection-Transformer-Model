"""
CYBERBULLYING DETECTION: Fine-tuned Transformer Model

This script trains a specialized model for cyberbullying detection using:
  - Fine-tuned RoBERTa or DistilBERT
  - Multi-class labels (harassment, hate_speech, threats, etc.)
  - Class weighting for imbalanced data
  - Optimized for this specific task

Install dependencies:
    pip install transformers datasets torch scikit-learn pandas accelerate --break-system-packages

Usage:
    python cyberbullying_detection.py --data_path ./cyberbullying_data.csv
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
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import warnings
import argparse
import json

warnings.filterwarnings("ignore")


# ========================================
# 1. LOAD CYBERBULLYING DATA
# ========================================
def load_cyberbullying_data(data_path):
    """Load cyberbullying dataset from CSV."""
    print(f"📂 Loading cyberbullying data from: {data_path}")
    
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} examples\n")
    
    return df


# ========================================
# 2. EXPLORE DATA
# ========================================
def explore_cyberbullying_data(df):
    """Analyze cyberbullying dataset."""
    print("📊 DATASET ANALYSIS")
    print("="*70)
    
    # Basic info
    print(f"Total examples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst 3 examples:")
    print(df.head(3))
    
    # Label distribution
    if 'label' in df.columns or 'is_bullying' in df.columns:
        label_col = 'label' if 'label' in df.columns else 'is_bullying'
        print(f"\n🏷️ Label Distribution ({label_col}):")
        dist = df[label_col].value_counts()
        print(dist)
        print(f"\nPercentages:")
        print((dist / len(df) * 100).round(2))
    
    # Text statistics
    if 'text' in df.columns:
        print(f"\n📝 Text Statistics:")
        lengths = df['text'].str.len()
        print(f"  Min length: {lengths.min()}")
        print(f"  Max length: {lengths.max()}")
        print(f"  Avg length: {lengths.mean():.0f}")
        print(f"  Median length: {lengths.median():.0f}")
    
    print("\n" + "="*70 + "\n")


# ========================================
# 3. PREPARE CYBERBULLYING DATA
# ========================================
def prepare_cyberbullying_data(
    df,
    text_col="text",
    label_col="label",
    test_size=0.15,
    val_size=0.15,
):
    """
    Prepare cyberbullying data for training.
    
    Supports labels:
      - Binary: is_bullying (0/1) or bullying/not_bullying
      - Multi-class: harassment, hate_speech, threats, etc.
    """
    print("🔧 PREPARING DATA")
    print("="*70)
    
    # Validate columns
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Columns {text_col}, {label_col} not found. Available: {df.columns.tolist()}")
    
    # Remove missing values
    initial_size = len(df)
    df = df.dropna(subset=[text_col, label_col])
    removed = initial_size - len(df)
    
    if removed > 0:
        print(f"Removed {removed} rows with missing values")
    
    # Encode labels
    if df[label_col].dtype == 'object':
        label_encoder = LabelEncoder()
        df['encoded_label'] = label_encoder.fit_transform(df[label_col])
        
        label_mapping = dict(zip(
            label_encoder.classes_,
            label_encoder.transform(label_encoder.classes_)
        ))
        
        print(f"\n🏷️ Label Encoding:")
        for class_name, class_id in sorted(label_mapping.items(), key=lambda x: x[1]):
            count = (df['encoded_label'] == class_id).sum()
            pct = count / len(df) * 100
            print(f"  {class_name}: {class_id} ({count} samples, {pct:.1f}%)")
        
        num_labels = len(label_encoder.classes_)
    else:
        df['encoded_label'] = df[label_col]
        num_labels = df[label_col].nunique()
        print(f"Number of classes: {num_labels}")
    
    # Calculate class weights for imbalanced data
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(df['encoded_label']),
        y=df['encoded_label']
    )
    
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    print(f"\n⚖️ Class Weights (for imbalanced data):")
    for class_id, weight in sorted(class_weight_dict.items()):
        print(f"  Class {class_id}: {weight:.3f}")
    
    # Train/val/test split
    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=df['encoded_label']
    )
    
    # Second split: train vs val
    adjusted_val_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_size,
        random_state=42,
        stratify=train_val_df['encoded_label']
    )
    
    print(f"\n📊 Data Split:")
    print(f"  Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    # Create HF datasets
    train_dataset = Dataset.from_dict({
        'text': train_df[text_col].tolist(),
        'label': train_df['encoded_label'].tolist()
    })
    
    val_dataset = Dataset.from_dict({
        'text': val_df[text_col].tolist(),
        'label': val_df['encoded_label'].tolist()
    })
    
    test_dataset = Dataset.from_dict({
        'text': test_df[text_col].tolist(),
        'label': test_df['encoded_label'].tolist()
    })
    
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    })
    
    print("\n" + "="*70 + "\n")
    
    return dataset_dict, num_labels, class_weight_dict


# ========================================
# 4. TOKENIZE DATASET
# ========================================
def tokenize_dataset(dataset, model_name="distilbert-base-uncased", max_length=256):
    """Tokenize texts for transformer."""
    print(f"🔤 Tokenizing dataset (max_length={max_length})...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    
    print("✅ Tokenization complete\n")
    return tokenized_dataset, tokenizer


# ========================================
# 5. CUSTOM TRAINER WITH CLASS WEIGHTS
# ========================================
class CyberbullyingTrainer(Trainer):
    """Custom trainer that uses class weights for imbalanced data."""
    
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute weighted loss for imbalanced data."""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.class_weights:
            import torch
            loss_fn = torch.nn.CrossEntropyLoss(
                weight=torch.tensor(
                    list(self.class_weights.values()),
                    dtype=torch.float32,
                    device=model.device
                )
            )
            loss = loss_fn(logits, labels)
        else:
            loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss


# ========================================
# 6. COMPUTE METRICS
# ========================================
def compute_metrics(eval_pred):
    """Compute detailed metrics for cyberbullying detection."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, predictions, average="weighted", zero_division=0),
        "precision": precision_score(labels, predictions, average="weighted", zero_division=0),
        "recall": recall_score(labels, predictions, average="weighted", zero_division=0),
    }


# ========================================
# 7. TRAIN MODEL
# ========================================
def train_cyberbullying_model(
    tokenized_dataset,
    num_labels,
    class_weights,
    model_name="distilbert-base-uncased",
    output_dir="./cyberbullying_model",
    num_epochs=3,
    batch_size=16,
    learning_rate=2e-5,
):
    """Train fine-tuned model for cyberbullying detection."""
    print("🚀 TRAINING CYBERBULLYING DETECTION MODEL")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Output: {output_dir}\n")
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        save_total_limit=2,
        seed=42,
    )
    
    # Create trainer with class weights
    trainer = CyberbullyingTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['val'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )
    
    # Train
    trainer.train()
    
    print("\n" + "="*70)
    print("📈 VALIDATION RESULTS")
    print("="*70)
    val_metrics = trainer.evaluate()
    for metric, value in val_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Test on test set
    print("\n" + "="*70)
    print("🧪 TEST SET RESULTS")
    print("="*70)
    test_results = trainer.evaluate(eval_dataset=tokenized_dataset['test'])
    for metric, value in test_results.items():
        print(f"{metric}: {value:.4f}")
    
    # Detailed classification report
    print("\n" + "="*70)
    print("📊 DETAILED CLASSIFICATION REPORT")
    print("="*70)
    predictions = trainer.predict(tokenized_dataset['test'])
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    true_labels = predictions.label_ids
    
    print(classification_report(true_labels, pred_labels))
    
    # Save model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save metrics
    metrics_file = Path(output_dir) / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump({
            'validation': val_metrics,
            'test': test_results
        }, f, indent=2)
    
    print(f"\n✅ Model saved to {output_dir}")
    print(f"✅ Metrics saved to {metrics_file}\n")
    
    return trainer


# ========================================
# 8. ARGUMENT PARSER
# ========================================
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train cyberbullying detection model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python cyberbullying_detection.py --data_path ./cyberbullying_data.csv
  
  # With custom model and parameters
  python cyberbullying_detection.py \\
    --data_path ./data.csv \\
    --model roberta-base \\
    --epochs 5 \\
    --batch_size 32 \\
    --learning_rate 1e-5
  
  # High accuracy training
  python cyberbullying_detection.py \\
    --data_path ./data.csv \\
    --model roberta-base \\
    --epochs 10 \\
    --batch_size 64 \\
    --max_length 512
        """)
    
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to cyberbullying dataset (CSV)"
    )
    
    parser.add_argument(
        "--text_col",
        type=str,
        default="text",
        help="Name of text column (default: text)"
    )
    
    parser.add_argument(
        "--label_col",
        type=str,
        default="label",
        help="Name of label column (default: label)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="distilbert-base-uncased",
        help="Pretrained model (default: distilbert-base-uncased)",
        choices=[
            "distilbert-base-uncased",
            "bert-base-uncased",
            "roberta-base",
            "albert-base-v2",
        ]
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size (default: 16)"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum sequence length (default: 256)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./cyberbullying_model",
        help="Output directory (default: ./cyberbullying_model)"
    )
    
    return parser.parse_args()


# ========================================
# 9. MAIN PIPELINE
# ========================================
if __name__ == "__main__":
    import torch
    
    args = parse_arguments()
    
    # Check GPU
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    
    print("="*70)
    print("🎯 CYBERBULLYING DETECTION: TRAINING PIPELINE")
    print("="*70 + "\n")
    
    try:
        # Load data
        df = load_cyberbullying_data(args.data_path)
        
        # Explore
        explore_cyberbullying_data(df)
        
        # Prepare
        dataset, num_labels, class_weights = prepare_cyberbullying_data(
            df,
            text_col=args.text_col,
            label_col=args.label_col,
        )
        
        # Tokenize
        tokenized_dataset, tokenizer = tokenize_dataset(
            dataset,
            model_name=args.model,
            max_length=args.max_length
        )
        
        # Train
        trainer = train_cyberbullying_model(
            tokenized_dataset,
            num_labels,
            class_weights,
            model_name=args.model,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        
        print("\n" + "="*70)
        print("✨ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\n📌 Use for inference:")
        print(f"   python cyberbullying_inference.py --model {args.output_dir}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
