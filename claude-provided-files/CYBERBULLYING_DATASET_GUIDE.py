"""
CYBERBULLYING DETECTION: Dataset Types & Model Approaches

This guide covers:
1. Best dataset types for cyberbullying detection
2. Pre-trained models vs training from scratch
3. Transformers for this task
4. Hybrid approaches
5. Practical recommendations
"""

# ========================================
# 1. DATASET TYPES FOR CYBERBULLYING
# ========================================

DATASET_COMPARISON = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ DATASET TYPE COMPARISON FOR CYBERBULLYING DETECTION                        │
├─────────────────────────────────────────────────────────────────────────────┤

1. ANNOTATED SOCIAL MEDIA TEXT (⭐⭐⭐⭐⭐ BEST)
   ─────────────────────────────────────────────
   Source: Twitter, Instagram, Reddit, Facebook comments
   Format: [text, label (bullying/not_bullying)]
   Characteristics:
     ✅ Real-world examples of cyberbullying
     ✅ Natural language and slang
     ✅ Contextual relevance to social media
     ✅ Multiple bullying types (harassment, doxing, hate speech)
     ✅ Emotional triggers preserved
   
   Challenges:
     ❌ Privacy concerns (need anonymization)
     ❌ Imbalanced classes (bullying is less frequent)
     ❌ Subjective labels (need multiple annotators)
     ❌ Offensive language present
   
   Example Structure:
     text,is_bullying,bullying_type
     "You're so ugly and stupid",1,harassment
     "Great job on the project!",0,none
     "Die in a fire you n*****",1,hate_speech


2. MULTI-LABEL CYBERBULLYING DATA (⭐⭐⭐⭐⭐ VERY GOOD)
   ──────────────────────────────────────────────────
   Categories: harassment, doxing, hate_speech, threats, exclusion
   Format: [text, label1, label2, ...]
   Advantages:
     ✅ Captures multiple simultaneous issues
     ✅ More nuanced understanding
     ✅ Better for real-world scenarios
     ✅ Helps model learn subtleties
   
   Example:
     text,harassment,hate_speech,threats
     "Death threats to all X group members",1,1,1
     "You're dumb",1,0,0


3. CONTEXT-AWARE DATA (⭐⭐⭐⭐⭐ EXCELLENT)
   ──────────────────────────────────────
   Includes: Target, Context, Reply chain
   Format: [text, target_person, context, label]
   Why it's better:
     ✅ "You're trash" differs with/without context
     ✅ Game trash talk ≠ personal harassment
     ✅ Model learns context dependency
     ✅ More robust predictions
   
   Example:
     text,target,context,is_bullying
     "You're trash in gaming",player,gaming_chat,0
     "You're trash as a person",specific_user,personal_attack,1


4. MULTILINGUAL DATA (⭐⭐⭐⭐ GOOD)
   ──────────────────────────────
   Languages: English, Spanish, French, etc.
   Advantages:
     ✅ Global cyberbullying patterns
     ✅ Cross-lingual understanding
     ✅ Better generalization
   
   Limitation:
     ❌ More complex to collect
     ❌ Language-specific nuances


5. TEMPORAL DATA (⭐⭐⭐ GOOD)
   ────────────────────────
   Includes: Timestamps, User history
   Useful for:
     ✅ Understanding escalation patterns
     ✅ Detecting coordinated attacks
     ✅ User behavior patterns
   
   Example:
     text,timestamp,user_id,is_bullying
     "I hate you",2024-01-15 10:30,user123,1
     "Die",2024-01-15 10:31,user456,1


6. SYNTHETIC/AUGMENTED DATA (⭐⭐⭐ HELPFUL)
   ─────────────────────────────────────
   Created using:
     - Paraphrasing real examples
     - Back-translation
     - Synonym replacement
   Advantages:
     ✅ Increases training data
     ✅ Reduces class imbalance
     ✅ Helps model generalize
   
   Limitation:
     ❌ May not capture real-world patterns
     ❌ Can introduce artificial patterns


7. EXISTING PUBLIC DATASETS (⭐⭐⭐⭐ GOOD)
   ─────────────────────────────────────
   Examples:
     - Hate Speech and Offensive Language Dataset
     - Cyberbullying Classification Dataset
     - OffenseEval datasets
     - SemEval shared tasks
   Advantages:
     ✅ Readily available
     ✅ Pre-annotated
     ✅ Benchmarked
     ✅ No privacy issues
   
   Limitation:
     ❌ May not be domain-specific
     ❌ Size varies


8. COMBINATION/ENSEMBLE DATA (⭐⭐⭐⭐⭐ BEST PRACTICE)
   ────────────────────────────────────────────────
   Mix of: Public datasets + Custom annotated data
   Strategy:
     ✅ Use public data for pretraining
     ✅ Fine-tune on domain-specific data
     ✅ Add custom edge cases
     ✅ Balance classes artificially
   
   Example:
     - 70% from public datasets
     - 20% from your own annotated data
     - 10% synthetic/augmented
"""

print(DATASET_COMPARISON)

# ========================================
# 2. MODEL APPROACH COMPARISON
# ========================================

MODEL_APPROACH_COMPARISON = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ MODEL APPROACH COMPARISON FOR CYBERBULLYING DETECTION                       │
├─────────────────────────────────────────────────────────────────────────────┤

APPROACH 1: PRE-TRAINED MODELS ONLY
────────────────────────────────────
Models: BERT, RoBERTa, DistilBERT (without fine-tuning)
How it works: Use embeddings directly for similarity
Accuracy: 60-70%
Training time: None
Data needed: None (or just for testing)

Pros:
  ✅ No training required
  ✅ Works immediately
  ✅ No private data needed
  ✅ Fast deployment

Cons:
  ❌ Lower accuracy for cyberbullying
  ❌ Misses domain-specific patterns
  ❌ Can't learn new offensive terms
  ❌ Struggles with slang/abbreviations
  ❌ Not recommended for production

Example scenario: "yeet u trash" - may not understand as bullying


APPROACH 2: TRANSFORMERS FINE-TUNED (⭐⭐⭐⭐⭐ BEST)
────────────────────────────────────────────────
Models: BERT, RoBERTa, DistilBERT (with fine-tuning)
How it works: Fine-tune pre-trained model on cyberbullying data
Accuracy: 85-92%
Training time: 30 min - 2 hours (depending on data size)
Data needed: 1000-10000 labeled examples

Pros:
  ✅ Highest accuracy (85-92%)
  ✅ Learns cyberbullying-specific patterns
  ✅ Understands slang & abbreviations
  ✅ Good with limited data
  ✅ Domain transfer learning
  ✅ Fast inference
  ✅ Production-ready

Cons:
  ❌ Needs labeled data
  ❌ Needs training infrastructure
  ❌ Hyperparameter tuning required

Example scenario: "yeet u trash" - understands slang context
Recommended: YES, this is the best approach


APPROACH 3: HYBRID ENSEMBLE (⭐⭐⭐⭐⭐ BEST FOR ROBUSTNESS)
──────────────────────────────────────────────────────────
Combines:
  1. Fine-tuned Transformer (85-92% accuracy)
  2. Rule-based patterns (catch obvious cases)
  3. Lexicon approach (offensive words list)
  4. Multiple models voting

Accuracy: 90-95%
Training time: 1-3 hours
Data needed: 1000-10000 examples + word lists

Pros:
  ✅ Highest accuracy (90-95%)
  ✅ Catches edge cases
  ✅ Robust to adversarial inputs
  ✅ Explainable predictions
  ✅ Handles multiple bullying types
  ✅ Production-grade

Cons:
  ❌ More complex
  ❌ Higher maintenance
  ❌ Slower inference
  ❌ Requires domain expertise

Example: Combines transformer + keyword detection + context rules


APPROACH 4: TRADITIONAL ML (Not recommended)
──────────────────────────────────────────────
Models: SVM, Random Forest, Naive Bayes
Accuracy: 70-80%
Training time: Minutes
Data needed: 100+

Pros:
  ✅ Fast to train
  ✅ Works with small data
  ✅ Simple

Cons:
  ❌ Lower accuracy than transformers
  ❌ Poor at understanding context
  ❌ Requires manual feature engineering
  ❌ Struggles with slang
  ❌ Not recommended for this task


APPROACH 5: LARGE LANGUAGE MODELS (GPT-4, etc)
────────────────────────────────────────────────
Models: GPT-4, Claude, LLaMA
Accuracy: 88-95%
Cost: High ($)
Training time: None (pre-trained)

Pros:
  ✅ Very high accuracy
  ✅ Great context understanding
  ✅ Few-shot learning capability
  ✅ Handles complex cases

Cons:
  ❌ Expensive API costs
  ❌ Privacy concerns (data to cloud)
  ❌ Latency for real-time use
  ❌ Overkill for many cases
  ❌ Harder to deploy locally

Recommended for: Enterprise with budget, privacy-agnostic
"""

print(MODEL_APPROACH_COMPARISON)

# ========================================
# 3. RECOMMENDATION MATRIX
# ========================================

RECOMMENDATION = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ BEST APPROACH FOR YOUR USE CASE                                             │
├─────────────────────────────────────────────────────────────────────────────┤

YOUR SITUATION                          → RECOMMENDED APPROACH
──────────────────────────────────────────────────────────────────────────────

Quick prototype, no data               → Pre-trained BERT (70% accuracy)
                                          • Fast to start
                                          • 0 labeling effort

Production use, 1000+ labeled examples → Fine-tuned Transformer (92% accuracy)
                                          ⭐ BEST CHOICE
                                          • Highest accuracy for effort
                                          • Reasonable data requirement
                                          • Good speed/accuracy tradeoff

Production use, 5000+ examples + time  → Hybrid Ensemble (95% accuracy)
                                          • Maximum accuracy
                                          • More complex but worth it
                                          • Handles edge cases

Enterprise, unlimited budget           → LLM API + Fine-tuned Model
                                          • Best overall performance
                                          • High cost
                                          • Privacy tradeoff

Limited compute, need speed            → DistilBERT fine-tuned (88% accuracy)
                                          • Fast inference
                                          • Lower resource requirement
                                          • Slightly lower accuracy


FOR CYBERBULLYING SPECIFICALLY:
═══════════════════════════════════════════════════════════════════════════════

✅ RECOMMENDED STACK:

1. MODEL: RoBERTa-base or DistilBERT
2. DATASET: Combination of:
   - 70% from public datasets (Hate Speech, OffenseEval, etc.)
   - 20% custom annotated from your domain
   - 10% synthetic augmented data

3. APPROACH: Fine-tuned Transformer with:
   - Multi-class labels (harassment, hate_speech, threats, etc.)
   - Class weighting for imbalanced data
   - Data augmentation for minority classes
   - Threshold tuning for precision/recall tradeoff

4. VALIDATION: Test on:
   - Different platforms (Twitter, Reddit, Discord, etc.)
   - Different languages if needed
   - Edge cases (sarcasm, reclaimed slurs, etc.)

Expected Performance: 88-92% accuracy, 85-90% F1-score


❌ AVOID:

1. Training from scratch on small data (<1000 examples)
2. Using only pre-trained models without fine-tuning
3. Single-label classification (too simplistic)
4. Ignoring class imbalance
5. Not testing on diverse platforms/languages


📊 DATA REQUIREMENTS BY APPROACH:

Pre-trained only:        0 examples needed (70% accuracy)
Fine-tuned transformer:  1,000-5,000 examples (92% accuracy) ⭐ SWEET SPOT
Hybrid ensemble:         5,000+ examples (95% accuracy)
Training from scratch:   50,000+ examples (requires GPUs)
"""

print(RECOMMENDATION)

# ========================================
# 4. PUBLIC DATASETS FOR CYBERBULLYING
# ========================================

PUBLIC_DATASETS = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ PUBLIC DATASETS YOU CAN USE                                                 │
├─────────────────────────────────────────────────────────────────────────────┤

1. HATE SPEECH AND OFFENSIVE LANGUAGE DATASET
   ────────────────────────────────────────────
   Source: https://github.com/t-davidson/hate-speech-dataset
   Size: 24,802 tweets
   Labels: hate speech, offensive language, neither
   Quality: ⭐⭐⭐⭐
   Download: CSV format directly
   
   Good for: Initial training, understanding offensive content


2. CYBERBULLYING CLASSIFICATION DATASET
   ──────────────────────────────────────
   Source: Kaggle - Search "cyberbullying classification"
   Size: 47,692 tweets
   Labels: age, gender, religion, ethnicity, other (not cyberbullying)
   Quality: ⭐⭐⭐⭐
   
   Good for: Target-specific bullying detection


3. OFFENSIVE LANGUAGE IDENTIFICATION (OffenseEval)
   ────────────────────────────────────────────────
   Source: SemEval shared task (OffenseEval 2019, 2020, 2021)
   Size: ~14,000 tweets per year
   Labels: offensive/not offensive, targeted/untargeted, type
   Quality: ⭐⭐⭐⭐⭐
   
   Good for: Competitive benchmark, multi-level classification


4. TOXIC COMMENT CLASSIFICATION (Kaggle)
   ─────────────────────────────────────
   Source: Kaggle competition
   Size: 223,549 comments
   Labels: toxic, severe_toxic, obscene, threat, insult, identity_hate
   Quality: ⭐⭐⭐⭐
   
   Good for: Multi-label cyberbullying, production training


5. TWITTER ABUSE DATASET
   ──────────────────────
   Source: Various researchers
   Size: 100,000+
   Labels: abuse/not abuse
   Quality: ⭐⭐⭐
   
   Good for: Platform-specific training


6. HASOC (Hate Speech and Offensive Content)
   ─────────────────────────────────────────
   Source: Shared task (English, Hindi, German)
   Size: 5,000+ per language
   Labels: hate speech, offensive, profane, none
   Quality: ⭐⭐⭐⭐⭐
   Multilingual: Yes
   
   Good for: Multilingual approaches


7. REDDIT r/BanHammer
   ──────────────────
   Source: Reddit moderation data
   Size: ~100,000 comments
   Labels: Removed for various violations
   Quality: ⭐⭐⭐
   
   Good for: Real-world examples, context-aware


8. YOUTUBE COMMENTS DATASET
   ──────────────────────────
   Source: Various researchers
   Size: 50,000+
   Labels: spam, abusive, clean
   Quality: ⭐⭐⭐
   
   Good for: Platform diversity


COMBINED DATASET STRATEGY:
═════════════════════════════════════════════════════════════════════════════

1. Download multiple datasets (1, 2, 3, 4, 6)
2. Standardize labels to: [text, is_bullying, bullying_type]
3. Remove duplicates
4. Balance classes (upsample minority)
5. Split: 70% train, 15% val, 15% test
6. Train fine-tuned transformer
7. Evaluate on each dataset separately
8. Fine-tune on your domain-specific data (if available)

Expected result: 88-92% on diverse platforms
"""

print(PUBLIC_DATASETS)

# ========================================
# 5. DATASET CHARACTERISTICS
# ========================================

IDEAL_DATASET = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ IDEAL CYBERBULLYING DATASET CHARACTERISTICS                                 │
├─────────────────────────────────────────────────────────────────────────────┤

SIZE:
  • Minimum: 1,000 examples
  • Recommended: 5,000-10,000 examples
  • Optimal: 20,000+ examples
  
  Rule of thumb: 10x more examples for +1% accuracy improvement

BALANCE:
  • Bullying: 40-50%
  • Non-bullying: 50-60%
  
  If imbalanced: Use class weights or oversampling
  Example: 30% bullying / 70% non-bullying
    → Apply class weight of 2.33 to bullying class

LABELS (Multi-class better than binary):
  Labels to include:
    • harassment/personal attacks
    • hate speech (group-targeted)
    • threats/threats of violence
    • doxing/sharing private info
    • exclusion/ostracism
    • cyberstalking
    • none (clean content)

CONTEXT:
  Better: Include conversation context
    ❌ "You're trash" (ambiguous)
    ✅ "You're trash" [in: gaming chat] (not bullying)
    ✅ "You're trash" [target: specific user] (bullying)

DIVERSITY:
  • Multiple platforms (Twitter, Reddit, Discord, etc.)
  • Multiple languages if possible
  • Different user demographics
  • Various writing styles (formal, slang, abbreviations)

QUALITY:
  • Inter-annotator agreement ≥ 0.8 (Cohen's kappa)
  • Multiple annotators per example
  • Clear annotation guidelines
  • Cases resolved by expert if disagreement

METADATA (optional but helpful):
  • User ID
  • Timestamp
  • Platform
  • Target information
  • Conversation chain
  • User history


EXAMPLE PERFECT CYBERBULLYING DATASET FORMAT:
═════════════════════════════════════════════════════════════════════════════

text,is_bullying,bullying_type,platform,severity
"You suck at this game",0,none,gaming_chat,0
"I hope you die",1,threat,twitter,3
"You're such an idiot",1,harassment,reddit,2
"Great analysis!",0,none,twitter,0
"Kill yourself",1,threat,discord,3
"Your music is boring",0,none,youtube,0
"Nobody likes you because you're X",1,hate_speech,twitter,3
"Let's exclude them from the group",1,exclusion,discord,2


MINIMUM VIABLE DATASET:
══════════════════════════════════════════════════════════════════════════════

You can start with:
  • 1,000 labeled examples
  • Binary labels (bullying / not bullying)
  • From 1-2 public sources
  • No metadata required

This gives: ~85% accuracy

Then improve by:
  • Adding multi-class labels
  • Adding context
  • Adding more examples
  • Adding multiple annotators
  • Balancing classes

This gives: ~92% accuracy
"""

print(IDEAL_DATASET)

# ========================================
# 6. HOW TO CREATE YOUR CYBERBULLYING DATASET
# ========================================

DATASET_CREATION = """
┌─────────────────────────────────────────────────────────────────────────────┐
│ HOW TO CREATE YOUR OWN CYBERBULLYING DATASET                                │
├─────────────────────────────────────────────────────────────────────────────┤

STEP 1: COLLECT RAW DATA
────────────────────────
Option A: Use Public APIs
  • Twitter API (academic research track)
  • Reddit API (praw library)
  • YouTube API (public comments)
  
Option B: Use Existing Research Data
  • Academic datasets (GitHub)
  • Kaggle datasets
  • HuggingFace datasets
  
Option C: Scrape (Ethically)
  • Respect Terms of Service
  • Anonymize before storing
  • Get IRB approval if academic

Recommended: Start with existing datasets + add your domain data


STEP 2: ANONYMIZE & CLEAN
─────────────────────────
# Pseudonymize personally identifiable information
import re

def anonymize(text):
    # Remove URLs
    text = re.sub(r'http\S+', '[URL]', text)
    # Remove @mentions (keeping count)
    text = re.sub(r'@\w+', '@user', text)
    # Remove emails
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL]', text)
    # Remove phone numbers
    text = re.sub(r'\+?1?\d{9,15}', '[PHONE]', text)
    return text

# IMPORTANT: Store original and anonymized separately for records
# Follow GDPR/CCPA if applicable


STEP 3: ANNOTATION GUIDELINES
──────────────────────────────
Create clear guidelines for annotators:

HARASSMENT:
  Definition: Direct personal attacks
  Examples:
    ✅ "You're stupid and ugly"
    ✅ "I hate you so much"
    ❌ "This code is bad" (criticism, not bullying)
  
HATE SPEECH:
  Definition: Attacks targeting group identity
  Examples:
    ✅ "All X people are criminals"
    ✅ "Death to [group]"
    ❌ "I don't like this restaurant" (not group-based)
  
THREATS:
  Definition: Implied or explicit threats of harm
  Examples:
    ✅ "I know where you live"
    ✅ "Wait till I find you"
    ❌ "That was close!" (not a threat)
  
EXCLUSION:
  Definition: Intentional exclusion/ostracism
  Examples:
    ✅ "Nobody likes you, leave the group"
    ✅ "We're blocking you from everything"


STEP 4: RECRUIT & TRAIN ANNOTATORS
────────────────────────────────────
• Need: 2-3 annotators per example
• Training: Show them guidelines + examples
• Validation: Check first 50 for consistency
• Payment: Reasonable compensation ($0.10-0.50 per item)
• Tools: Use Label Studio, Amazon MTurk, or custom tool


STEP 5: CALCULATE INTER-ANNOTATOR AGREEMENT
──────────────────────────────────────────────
from sklearn.metrics import cohen_kappa_score

# Calculate agreement
agreement = cohen_kappa_score(annotator1, annotator2)

# Interpretation:
# 0.81-1.00: Excellent agreement ✅
# 0.61-0.80: Substantial agreement ✅
# 0.41-0.60: Moderate agreement (resolve disagreements)
# 0.01-0.40: Fair to poor agreement (revise guidelines)

Target: ≥ 0.80 (Excellent agreement)


STEP 6: RESOLVE DISAGREEMENTS
──────────────────────────────
For cases where annotators disagree:
  • Option A: Take majority vote
  • Option B: Expert review (better)
  • Option C: Exclude ambiguous cases (stricter)


STEP 7: FINAL DATASET
─────────────────────
Output format:
  text,is_bullying,bullying_type,confidence,annotator_count
  "You're dumb",1,harassment,0.95,3
  "Great job!",0,none,0.98,3
  "...",... ,... ,... ,...


TIMELINE & EFFORT:
═════════════════════════════════════════════════════════════════════════════

1,000 examples:    2-4 weeks (1-2 hours annotation each)
5,000 examples:    2-3 months (active recruitment)
10,000 examples:   3-6 months

Effort breakdown:
  - Collection: 20%
  - Annotation: 50%
  - Validation: 20%
  - Quality control: 10%


COST ESTIMATION:
════════════════════════════════════════════════════════════════════════════

MTurk rates (~$0.20 per item):
  1,000 items:   $200
  5,000 items:   $1,000
  10,000 items:  $2,000

Expert annotation (~$0.50 per item):
  1,000 items:   $500
  5,000 items:   $2,500


FASTER ALTERNATIVE: USE EXISTING DATASETS
═════════════════════════════════════════════════════════════════════════════

Most practical approach:
1. Download 2-3 public datasets (weeks, not months)
2. Combine and preprocess them
3. Add 500-1000 custom examples for your domain
4. Train fine-tuned model

Time investment: 2-4 weeks
Cost: $0 (free public datasets)
Accuracy: 88-92%

Recommended for most projects!
"""

print(DATASET_CREATION)

print("\n" + "="*80)
print("SUMMARY & FINAL RECOMMENDATION")
print("="*80 + "\n")

SUMMARY = """
FOR CYBERBULLYING DETECTION:

🏆 BEST APPROACH:
   Fine-tuned Transformer (RoBERTa or DistilBERT)
   
📊 BEST DATASET:
   Combination of:
   - 70% Hate Speech & Offensive Language Dataset
   - 20% Toxic Comments Classification Dataset  
   - 10% Your domain-specific annotated data

🎯 EXPECTED PERFORMANCE:
   - Accuracy: 88-92%
   - F1-score: 85-90%
   - Precision: 86-91%
   - Recall: 84-89%

⏱️ TIME TO PRODUCTION:
   - Setup: 1 hour
   - Data collection: 1 week
   - Training: 1-2 hours
   - Validation: 1 week
   - Total: ~3 weeks

💰 COST:
   - Development: $0 (free tools & datasets)
   - Infrastructure: <$100 (cloud GPU for training)
   - Total: <$150

✅ WHEN TO USE THIS:
   - Production cyberbullying detection
   - Moderate accuracy requirements
   - Limited budget/resources
   - Single language okay

❌ WHEN TO USE HYBRID:
   - Enterprise deployment
   - Multiple platforms
   - High accuracy critical (95%+)
   - Edge cases important

❌ WHEN TO USE LLM:
   - Unlimited budget
   - Complex reasoning needed
   - Privacy not a concern
   - Real-time speed not critical
"""

print(SUMMARY)
