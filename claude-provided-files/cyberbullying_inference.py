"""
CYBERBULLYING DETECTION: Inference Script

Use trained model to detect cyberbullying in real-time texts.

Usage:
    python cyberbullying_inference.py --model ./cyberbullying_model --text "Text to classify"
"""

import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path
import argparse
import json


class CyberbullyingDetector:
    """Cyberbullying detection model wrapper."""
    
    def __init__(self, model_dir="./cyberbullying_model"):
        """Initialize detector with trained model."""
        print(f"Loading model from {model_dir}...")
        
        self.device = 0 if torch.cuda.is_available() else -1
        
        # Load classifier pipeline
        self.classifier = pipeline(
            "text-classification",
            model=model_dir,
            device=self.device,
            top_k=None  # Return scores for all labels
        )
        
        # Load tokenizer and model for details
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        
        # Load label mapping
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        
        print("✅ Model loaded successfully")
        print(f"Labels: {list(self.id2label.values())}")
        print(f"Using device: {'GPU' if self.device == 0 else 'CPU'}\n")
    
    def predict(self, text, return_probabilities=False):
        """
        Predict if text is cyberbullying.
        
        Returns:
            dict with 'label', 'score', 'is_bullying'
        """
        result = self.classifier(text)
        
        if isinstance(result, list):
            result = result[0] if result else {}
        
        # Convert label name to human-readable
        label_name = result.get('label', 'UNKNOWN')
        score = result.get('score', 0)
        
        # Determine if bullying
        is_bullying = self._is_bullying(label_name)
        
        output = {
            'text': text,
            'label': label_name,
            'score': round(score, 4),
            'is_bullying': is_bullying,
            'severity': self._get_severity(label_name),
        }
        
        if return_probabilities:
            output['all_scores'] = {
                r['label']: round(r['score'], 4) 
                for r in self.classifier(text, top_k=None)
            }
        
        return output
    
    def _is_bullying(self, label_name):
        """Determine if label indicates cyberbullying."""
        bullying_labels = [
            'harassment', 'hate_speech', 'threats', 'doxing',
            'exclusion', 'cyberstalking', 'abusive', 'toxic',
            'offensive', 'spam', 'insult', 'bullying'
        ]
        return any(label.lower() in label_name.lower() for label in bullying_labels)
    
    def _get_severity(self, label_name):
        """Estimate severity of cyberbullying."""
        if 'threat' in label_name.lower():
            return 'CRITICAL'
        elif 'hate' in label_name.lower():
            return 'HIGH'
        elif 'harassment' in label_name.lower():
            return 'MEDIUM'
        elif 'none' in label_name.lower() or 'clean' in label_name.lower():
            return 'NONE'
        else:
            return 'MEDIUM'
    
    def batch_predict(self, texts):
        """Predict for multiple texts."""
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results
    
    def predict_with_threshold(self, text, threshold=0.7):
        """Predict with confidence threshold."""
        result = self.predict(text, return_probabilities=True)
        
        if result['score'] < threshold:
            return {
                **result,
                'is_bullying': None,  # Uncertain
                'confidence': 'LOW',
                'message': f"Confidence {result['score']:.2%} below threshold {threshold:.0%}"
            }
        else:
            result['confidence'] = 'HIGH'
            return result
    
    def batch_predict_with_stats(self, texts):
        """Batch predict with statistics."""
        results = self.batch_predict(texts)
        
        bullying_count = sum(1 for r in results if r['is_bullying'])
        
        stats = {
            'total': len(results),
            'bullying': bullying_count,
            'not_bullying': len(results) - bullying_count,
            'bullying_percentage': round(bullying_count / len(results) * 100, 1),
            'critical_count': sum(1 for r in results if r['severity'] == 'CRITICAL'),
            'high_count': sum(1 for r in results if r['severity'] == 'HIGH'),
        }
        
        return {
            'results': results,
            'stats': stats
        }


# ========================================
# EXAMPLES
# ========================================

def run_examples():
    """Run example predictions."""
    print("="*70)
    print("CYBERBULLYING DETECTION: EXAMPLE PREDICTIONS")
    print("="*70 + "\n")
    
    detector = CyberbullyingDetector(model_dir="./cyberbullying_model")
    
    # Example 1: Single prediction
    print("📌 Example 1: Single Prediction")
    print("-" * 70)
    text = "I hope you die in a fire, you're worthless"
    result = detector.predict(text, return_probabilities=True)
    print(f"Text: {text}")
    print(f"Result: {json.dumps(result, indent=2)}\n")
    
    # Example 2: Multiple predictions
    print("📌 Example 2: Batch Predictions")
    print("-" * 70)
    texts = [
        "You're so stupid and ugly",
        "Great job on the presentation!",
        "I know where you live, watch out",
        "Your code is inefficient",
        "Everyone hates you, loser",
        "Let's grab coffee sometime!",
    ]
    
    results = detector.batch_predict_with_stats(texts)
    
    for i, result in enumerate(results['results'], 1):
        status = "⚠️ BULLYING" if result['is_bullying'] else "✅ CLEAN"
        print(f"{i}. {status} | {result['severity']:8} | {result['label']}")
        print(f"   Text: {result['text'][:60]}...")
        print(f"   Confidence: {result['score']:.2%}\n")
    
    print(f"Summary: {results['stats']}\n")
    
    # Example 3: With threshold
    print("📌 Example 3: Predictions with Threshold (75%)")
    print("-" * 70)
    test_texts = [
        "You're dumb",
        "Maybe you could improve this",
        "Absolute garbage code",
    ]
    
    for text in test_texts:
        result = detector.predict_with_threshold(text, threshold=0.75)
        status = "⚠️" if result['is_bullying'] else "✅"
        print(f"{status} {text}")
        print(f"   Label: {result['label']}, Score: {result['score']:.2%}, Confidence: {result.get('confidence', 'N/A')}")
        if 'message' in result:
            print(f"   Note: {result['message']}")
        print()


# ========================================
# COMMAND LINE INTERFACE
# ========================================

def main():
    parser = argparse.ArgumentParser(
        description="Cyberbullying detection inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prediction
  python cyberbullying_inference.py --text "I hate you"
  
  # With custom model
  python cyberbullying_inference.py --model ./my_model --text "You're stupid"
  
  # Interactive mode
  python cyberbullying_inference.py --interactive
  
  # With threshold
  python cyberbullying_inference.py --text "Maybe bad" --threshold 0.8
  
  # Show examples
  python cyberbullying_inference.py --examples
        """)
    
    parser.add_argument(
        "--text",
        type=str,
        help="Text to classify"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./cyberbullying_model",
        help="Path to model directory"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode"
    )
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Run example predictions"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Confidence threshold (0-1)"
    )
    parser.add_argument(
        "--probabilities",
        action="store_true",
        help="Show all probabilities"
    )
    
    args = parser.parse_args()
    
    # Check model exists
    if not Path(args.model).exists():
        print(f"❌ Model not found at {args.model}")
        print("   Train a model first: python cyberbullying_detection.py --data_path ./data.csv")
        return
    
    # Initialize detector
    detector = CyberbullyingDetector(model_dir=args.model)
    
    # Run examples
    if args.examples:
        run_examples()
        return
    
    # Single prediction
    if args.text:
        result = detector.predict(args.text, return_probabilities=args.probabilities)
        
        print("\n" + "="*70)
        print("CYBERBULLYING DETECTION RESULT")
        print("="*70)
        print(json.dumps(result, indent=2))
        return
    
    # Interactive mode
    if args.interactive:
        print("\n" + "="*70)
        print("CYBERBULLYING DETECTION: INTERACTIVE MODE")
        print("="*70)
        print("Enter text to classify (type 'quit' to exit, 'stats' for batch stats)\n")
        
        while True:
            user_input = input("Enter text: ").strip()
            
            if user_input.lower() == 'quit':
                print("Exiting...")
                break
            
            if not user_input:
                print("Please enter some text.\n")
                continue
            
            result = detector.predict(user_input, return_probabilities=False)
            
            status = "⚠️ CYBERBULLYING DETECTED" if result['is_bullying'] else "✅ CLEAN"
            print(f"\n{status}")
            print(f"Label: {result['label']}")
            print(f"Confidence: {result['score']:.2%}")
            print(f"Severity: {result['severity']}\n")
    
    else:
        # No input provided
        print("No input provided. Use:")
        print("  --text 'Your text here'  (single prediction)")
        print("  --interactive            (interactive mode)")
        print("  --examples              (run examples)")
        print("\nFor help: python cyberbullying_inference.py --help")


if __name__ == "__main__":
    main()
