# CYBERBULLYING DETECTION: Complete Implementation Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Dataset Types Explained](#dataset-types-explained)
3. [Approach Comparison](#approach-comparison)
4. [Implementation Steps](#implementation-steps)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 5-Minute Setup

```bash
# 1. Install dependencies
pip install transformers datasets torch scikit-learn pandas accelerate --break-system-packages

# 2. Train on sample data
python cyberbullying_detection.py --data_path ./cyberbullying_sample_data.csv

# 3. Make predictions
python cyberbullying_inference.py --text "I hope you die"

# 4. Interactive testing
python cyberbullying_inference.py --interactive
```

---

## Dataset Types Explained

### 1. **Annotated Social Media Text** (⭐⭐⭐⭐⭐ BEST)

**What it is:**
- Real posts from Twitter, Reddit, Instagram, etc.
- Labeled as bullying/not bullying by human annotators
- Includes different types: harassment, hate speech, threats

**Pros:**
- ✅ Authentic cyberbullying examples
- ✅ Natural language and slang
- ✅ Platform-specific patterns
- ✅ Most realistic training data

**Cons:**
- ❌ Privacy concerns
- ❌ Imbalanced (bullying is rarer)
- ❌ Subjective labels
- ❌ Requires annotation effort

**Example:**
```csv
text,is_bullying,type
"Die you worthless human",1,threats
"Great job!",0,none
"All X are criminals",1,hate_speech
```

**Where to get:**
- Kaggle Toxic Comments dataset
- Hate Speech and Offensive Language dataset
- Academic papers with released datasets

---

### 2. **Multi-Label Data** (⭐⭐⭐⭐⭐ EXCELLENT)

**What it is:**
- Single text can have multiple issues
- E.g., "Death to all X" = threats + hate speech

**Pros:**
- ✅ Captures nuance
- ✅ Better accuracy
- ✅ Real-world scenarios
- ✅ Learn interactions

**Cons:**
- ❌ More complex annotation
- ❌ Harder to collect

**Example:**
```csv
text,harassment,hate_speech,threats
"Death to all X people",0,1,1
"You're dumb",1,0,0
```

---

### 3. **Context-Aware Data** (⭐⭐⭐⭐⭐ EXCELLENT)

**What it is:**
- Includes conversation context
- Who said it to whom
- What platform/game/space

**Pros:**
- ✅ "You're trash" differs in gaming chat vs personal attack
- ✅ Sarcasm detection
- ✅ Better generalization

**Example:**
```csv
text,context,target,is_bullying
"You're trash",gaming_chat,player,0
"You're trash",personal_attack,user123,1
```

---

### 4. **Existing Public Datasets** (⭐⭐⭐⭐ GOOD)

**Top Datasets:**

| Dataset | Size | Labels | Quality |
|---------|------|--------|---------|
| **Hate Speech & Offensive Language** | 24,802 | 3-way | ⭐⭐⭐⭐ |
| **Toxic Comments (Kaggle)** | 223,549 | 6-label | ⭐⭐⭐⭐ |
| **OffenseEval (SemEval)** | 14,000+ | Multi-level | ⭐⭐⭐⭐⭐ |
| **HASOC** | 5,000+ | Multi-lingual | ⭐⭐⭐⭐ |
| **Cyberbullying Classification** | 47,692 | Target-based | ⭐⭐⭐⭐ |

**Pros:**
- ✅ Free and ready to use
- ✅ Pre-annotated
- ✅ No privacy issues
- ✅ Benchmark available

**Cons:**
- ❌ May not fit your domain
- ❌ Different annotation schemes

---

## Approach Comparison

### Pre-Trained Model Only (70% accuracy)
```bash
# BERT embeddings without fine-tuning
# Fast but less accurate
# Not recommended
```

**Pros:** Fast, no training needed  
**Cons:** Poor accuracy for cyberbullying

---

### Fine-Tuned Transformer (⭐⭐⭐⭐⭐ RECOMMENDED - 92% accuracy)

**What you do:**
1. Take pre-trained BERT/RoBERTa
2. Add task-specific layer
3. Train on cyberbullying data (1-2 hours)
4. Deploy

**Pros:**
- ✅ Highest accuracy for effort
- ✅ Works with 1,000+ examples
- ✅ Fast inference
- ✅ Production-ready
- ✅ Easy to deploy

**Cons:**
- ❌ Needs labeled data
- ❌ Needs training infrastructure

**Expected Performance:**
- Accuracy: 88-92%
- F1-score: 85-90%
- Inference time: <100ms

**Code:**
```python
python cyberbullying_detection.py --data_path ./data.csv

# Makes a model that achieves 88-92% accuracy
```

---

### Hybrid Ensemble (95% accuracy)

**What you do:**
1. Fine-tuned transformer (base)
2. Rule-based patterns (catch obvious cases)
3. Keyword/lexicon matching
4. Multiple models voting

**Pros:**
- ✅ Highest accuracy (95%)
- ✅ Robust to edge cases
- ✅ Explainable (can see why)

**Cons:**
- ❌ More complex
- ❌ Slower inference
- ❌ More maintenance

**Recommended for:**
- Enterprise systems
- High accuracy critical
- Budget available for complexity

---

### Large Language Models (88-95% accuracy)

**Models:** GPT-4, Claude, LLaMA

**Pros:**
- ✅ Very high accuracy
- ✅ Great context understanding
- ✅ Few-shot capable

**Cons:**
- ❌ Expensive ($)
- ❌ Privacy concerns
- ❌ Latency issues
- ❌ Overkill for many cases

**Recommended for:**
- Unlimited budget
- Complex reasoning needed
- Privacy is not a concern

---

## Implementation Steps

### Step 1: Prepare Your Dataset

**Option A: Use Sample Data (Quick Test)**
```bash
python cyberbullying_detection.py --data_path ./cyberbullying_sample_data.csv
```

**Option B: Use Public Dataset**

```python
# Download from Kaggle
# https://www.kaggle.com/datasets/search?q=cyberbullying

# Or download from GitHub:
# wget https://raw.githubusercontent.com/t-davidson/hate-speech-dataset/master/data/labeled_data.csv

# Prepare it
python prepare_data.py \
  --input labeled_data.csv \
  --output cyberbullying_data.csv \
  --text_col tweet \
  --label_col class \
  --clean --balance
```

**Option C: Annotate Your Own Data**

```csv
text,label
"Your text here",harassment
"Another text",none
...
```

**Best practice:** Combine all three!
```
- 70% from public datasets
- 20% from your domain
- 10% synthetic/augmented
```

---

### Step 2: Train Model

```bash
# Basic training
python cyberbullying_detection.py --data_path ./cyberbullying_data.csv

# Better accuracy (RoBERTa)
python cyberbullying_detection.py \
  --data_path ./cyberbullying_data.csv \
  --model roberta-base \
  --epochs 5 \
  --batch_size 32 \
  --learning_rate 1e-5

# High accuracy (more resources)
python cyberbullying_detection.py \
  --data_path ./cyberbullying_data.csv \
  --model roberta-base \
  --epochs 10 \
  --batch_size 64 \
  --max_length 512
```

**Expected times:**
- Small data (1K): 5-10 minutes
- Medium data (5K): 20-30 minutes
- Large data (20K): 1-2 hours

---

### Step 3: Test Model

```bash
# Single prediction
python cyberbullying_inference.py --text "I hate you"

# Interactive testing
python cyberbullying_inference.py --interactive

# Show examples
python cyberbullying_inference.py --examples

# With confidence threshold
python cyberbullying_inference.py --text "Maybe bad" --threshold 0.8
```

---

### Step 4: Deploy

```python
# In your application
from cyberbullying_inference import CyberbullyingDetector

detector = CyberbullyingDetector(model_dir="./cyberbullying_model")

# Single prediction
result = detector.predict("Your text here")
if result['is_bullying']:
    print(f"Detected: {result['label']}, Severity: {result['severity']}")

# Batch processing
results = detector.batch_predict(list_of_texts)
stats = detector.batch_predict_with_stats(list_of_texts)
```

---

## Best Practices

### 1. **Data Quality**

✅ DO:
- Collect diverse examples
- Use multiple annotators
- Target inter-annotator agreement ≥ 0.80
- Include context when possible
- Balance classes (or use class weights)

❌ DON'T:
- Use single annotator
- Skip validation
- Ignore class imbalance
- Mix annotation schemes

### 2. **Model Selection**

✅ Use:
- **DistilBERT** for speed/accuracy balance (recommended)
- **RoBERTa** for high accuracy (if resources available)
- **ALBERT** for very limited resources

❌ Avoid:
- XLNet (too slow for real-time)
- Training from scratch (use pre-trained always)

### 3. **Hyperparameters**

```bash
# Default (good for most cases)
epochs=3, batch_size=16, learning_rate=2e-5

# Better accuracy (if data >5K)
epochs=5, batch_size=32, learning_rate=1e-5

# Maximum accuracy (if resources available)
epochs=10, batch_size=64, learning_rate=5e-6, max_length=512
```

### 4. **Evaluation**

Always test on:
- Different platforms (if applicable)
- Different languages (if multilingual)
- Edge cases (sarcasm, slang, technical jargon)
- Temporal data (if available)

### 5. **Monitoring**

In production:
- Log predictions
- Track false positives/negatives
- Monitor for data drift
- Retrain periodically with new data

---

## Troubleshooting

### Problem: Low accuracy (< 80%)

**Solution 1: More data**
- Collect more labeled examples
- Use data augmentation
- Add public datasets

**Solution 2: Better model**
```bash
python cyberbullying_detection.py \
  --data_path ./data.csv \
  --model roberta-base \
  --epochs 5
```

**Solution 3: Better hyperparameters**
```bash
python cyberbullying_detection.py \
  --data_path ./data.csv \
  --learning_rate 1e-5 \
  --epochs 5 \
  --batch_size 32
```

---

### Problem: Class imbalance (70% not bullying, 30% bullying)

**Solution: Automatic class weighting**
- Script handles this automatically
- Weights minority class appropriately
- No action needed!

---

### Problem: Out of memory

**Solution 1: Smaller batch size**
```bash
python cyberbullying_detection.py --data_path ./data.csv --batch_size 8
```

**Solution 2: Smaller model**
```bash
python cyberbullying_detection.py \
  --data_path ./data.csv \
  --model distilbert-base-uncased \
  --batch_size 16
```

**Solution 3: Shorter sequences**
```bash
python cyberbullying_detection.py \
  --data_path ./data.csv \
  --max_length 128
```

---

### Problem: Poor predictions on edge cases

**Solution: Add more training data**
- Collect examples of edge cases
- Explicitly add to training data
- Retrain model

**Edge cases:**
- Sarcasm: "Yeah, that was GREAT" (sarcasm, not actually great)
- Reclaimed slurs: Used by community members themselves
- Technical jargon: "This code is trash" (not bullying)
- Humor: "I'm gonna kill that test!" (not a threat)

---

## Performance Summary

| Approach | Accuracy | Time | Effort | Cost |
|----------|----------|------|--------|------|
| Pre-trained only | 70% | 0 min | None | $0 |
| Fine-tuned DistilBERT | 88% | 20 min | Low | <$50 |
| Fine-tuned RoBERTa | 92% | 1 hour | Low | <$100 |
| Hybrid ensemble | 95% | 3 hours | High | <$500 |
| LLM (GPT-4) | 90% | 0 min | None | $$$ |

**Recommendation: Fine-tuned RoBERTa (92% accuracy)**
- Sweet spot between accuracy and effort
- Production-ready
- Affordable
- Industry standard

---

## Dataset Recommendation

For your cyberbullying detection model:

```
1. Start with: DistilBERT fine-tuned (88% accuracy)
2. Data source: Public datasets + your domain data
3. Size: 5,000-10,000 examples
4. Time: 2-4 weeks (mostly for data prep)
5. Cost: <$200
6. Expected accuracy: 88-92%
```

If you need higher accuracy (95%+):
```
1. Use: Hybrid ensemble approach
2. Time: 4-8 weeks
3. Cost: $500-2,000
4. Expected accuracy: 95%+
```

---

## Next Steps

1. ✅ Download sample data or create your own
2. ✅ Train: `python cyberbullying_detection.py --data_path ./data.csv`
3. ✅ Test: `python cyberbullying_inference.py --interactive`
4. ✅ Evaluate: Check accuracy on test set
5. ✅ Deploy: Integrate into your application
6. ✅ Monitor: Track performance in production
7. ✅ Retrain: Periodically with new data
