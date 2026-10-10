import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
import os

warnings.filterwarnings("ignore")

# Load the YouToxic training data
df = pd.read_csv('new datasets/YouToxic_training.csv')
print(f"Loaded {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")
print(f"First few rows:")
print(df.head())

# Prepare data
text_col = 'Text'
label_col = 'oh_label'

if label_col not in df.columns:
    label_col = df.columns[-1]

print(f"\nUsing text column: '{text_col}'")
print(f"Using label column: '{label_col}'")

# Encode labels
if df[label_col].dtype == 'object':
    label_encoder = LabelEncoder()
    df['encoded_label'] = label_encoder.fit_transform(df[label_col])
else:
    df['encoded_label'] = df[label_col].astype(int)

print(f"Label distribution: {df['encoded_label'].value_counts().to_dict()}")

num_labels = 2

# Train/test split
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

dataset_dict = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

# Tokenize
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=256,
        padding="max_length"
    )

tokenized_dataset = dataset_dict.map(
    tokenize_function,
    batched=True,
    remove_columns=['text']
)

print("Tokenization complete")

# Train model
output_dir = "./cyberbullying_model_youtoxic"

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=2,  # 2 epochs for quicker test
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=False,
    seed=42,
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    data_collator=data_collator,
)

print("\nStarting training...")
trainer.train()

print("\nFinal Evaluation:")
metrics = trainer.evaluate()
for metric, value in metrics.items():
    print(f"  {metric}: {value:.4f}")

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"\nModel saved to {output_dir}")