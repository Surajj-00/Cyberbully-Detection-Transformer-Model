"""
Use the trained social media model for inference/predictions.

After running kaggle_transformers_train.py, use this to make predictions
on new social media posts/text.
"""

import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer


class SocialMediaClassifier:
    """Wrapper for the trained social media classifier."""
    
    def __init__(self, model_dir="./social_media_model"):
        """Load the model and tokenizer."""
        print(f"Loading model from {model_dir}...")
        
        self.device = 0 if torch.cuda.is_available() else -1
        
        self.classifier = pipeline(
            "text-classification",
            model=model_dir,
            device=self.device,
            top_k=None  # Return scores for all labels
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        
        print("✅ Model loaded successfully")
        print(f"Using device: {'GPU' if self.device == 0 else 'CPU'}")
    
    def predict_single(self, text):
        """Predict label for a single text."""
        result = self.classifier(text)
        return result[0] if isinstance(result, list) else result
    
    def predict_batch(self, texts):
        """Predict labels for multiple texts."""
        results = self.classifier(texts)
        return results
    
    def predict_with_confidence(self, text, threshold=0.5):
        """Predict with confidence threshold."""
        predictions = self.classifier(text)
        
        if isinstance(predictions, list):
            predictions = predictions[0] if predictions else {}
        
        confidence = predictions.get('score', 0)
        label = predictions.get('label', 'UNKNOWN')
        
        if confidence < threshold:
            return {
                'label': 'LOW_CONFIDENCE',
                'score': confidence,
                'original_label': label,
                'threshold': threshold
            }
        
        return {
            'label': label,
            'score': confidence,
            'threshold': threshold
        }


# ========================================
# EXAMPLES
# ========================================

def main():
    """Run inference examples."""
    print("="*60)
    print("🎯 Social Media Classification Inference")
    print("="*60 + "\n")
    
    # Initialize classifier
    classifier = SocialMediaClassifier(model_dir="./social_media_model")
    
    # -------- Example 1: Single Prediction --------
    print("📌 Example 1: Single Prediction")
    print("-" * 60)
    
    text1 = "I absolutely love this product! Highly recommend!"
    result1 = classifier.predict_single(text1)
    
    print(f"Text: {text1}")
    print(f"Prediction: {result1}\n")
    
    # -------- Example 2: Multiple Predictions --------
    print("📌 Example 2: Batch Predictions")
    print("-" * 60)
    
    texts = [
        "This is terrible, worst experience ever",
        "It's okay, nothing special",
        "Amazing! Best purchase I've made",
        "Disappointed with the quality"
    ]
    
    results = classifier.predict_batch(texts)
    
    for text, result in zip(texts, results):
        print(f"Text: {text}")
        print(f"  → {result}\n")
    
    # -------- Example 3: Confidence Threshold --------
    print("📌 Example 3: Predictions with Confidence Threshold")
    print("-" * 60)
    
    test_texts = [
        "Love it!",
        "Maybe okay...",
        "Fantastic product!"
    ]
    
    for text in test_texts:
        result = classifier.predict_with_confidence(text, threshold=0.7)
        print(f"Text: {text}")
        print(f"  → Label: {result['label']}, Confidence: {result['score']:.4f}\n")
    
    # -------- Example 4: Interactive Mode --------
    print("📌 Example 4: Interactive Prediction Mode")
    print("-" * 60)
    print("Enter text to classify (type 'quit' to exit):\n")
    
    while True:
        user_input = input("Enter text: ").strip()
        
        if user_input.lower() == 'quit':
            print("Exiting...")
            break
        
        if not user_input:
            print("Please enter some text.\n")
            continue
        
        result = classifier.predict_single(user_input)
        print(f"Prediction: {result['label']} (confidence: {result['score']:.4f})\n")


# ========================================
# COMMAND LINE INTERFACE
# ========================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Classify social media text")
    parser.add_argument(
        "--text",
        type=str,
        help="Text to classify"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./social_media_model",
        help="Path to trained model (default: ./social_media_model)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)"
    )
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = SocialMediaClassifier(model_dir=args.model)
    
    # If text provided via command line
    if args.text:
        result = classifier.predict_with_confidence(args.text, threshold=args.threshold)
        print(f"\nText: {args.text}")
        print(f"Prediction: {result}")
    
    # If interactive mode
    elif args.interactive:
        print("\n🎯 Interactive Classification Mode")
        print("Type 'quit' to exit\n")
        
        while True:
            user_input = input("Enter text: ").strip()
            
            if user_input.lower() == 'quit':
                break
            
            if user_input:
                result = classifier.predict_with_confidence(
                    user_input,
                    threshold=args.threshold
                )
                print(f"→ {result['label']} (confidence: {result['score']:.4f})\n")
    
    # Default: run examples
    else:
        main()


# ========================================
# USAGE EXAMPLES
# ========================================

"""
Command line usage:

# Run built-in examples
python inference.py

# Classify single text
python inference.py --text "I love this!"

# Interactive mode
python inference.py --interactive

# Use custom model
python inference.py --text "Hello" --model ./my_custom_model

# With confidence threshold
python inference.py --text "Maybe..." --threshold 0.8

# Batch processing from file
python -c "
import pandas as pd
from inference import SocialMediaClassifier

classifier = SocialMediaClassifier()
df = pd.read_csv('test_data.csv')
predictions = classifier.predict_batch(df['text'].tolist())

df['prediction'] = [p['label'] for p in predictions]
df['confidence'] = [p['score'] for p in predictions]
df.to_csv('predictions.csv', index=False)
print('Predictions saved to predictions.csv')
"
"""
