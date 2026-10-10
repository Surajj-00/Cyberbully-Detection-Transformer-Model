#!/usr/bin/env python
"""
Cyberbullying Detection - Text Classification UI
A simple graphical interface for classifying text as cyberbullying or normal.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import os
import sys

# Try to load the trained model
class CyberbullyingClassifier:
    """Wrapper for text classification using trained transformer models."""
    
    def __init__(self, model_dir="./cyberbullying_model_quick"):
        self.model_dir = model_dir
        self.classifier = None
        self.model_loaded = False
        
        # Try to load model
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            if os.path.exists(model_dir):
                print(f"Loading model from {model_dir}...")
                self.device = 0 if torch.cuda.is_available() else -1
                self.classifier = pipeline(
                    "text-classification",
                    model=model_dir,
                    device=self.device,
                    top_k=None
                )
                self.model_loaded = True
                print("✅ Model loaded successfully")
            else:
                print(f"Model directory not found: {model_dir}")
        except ImportError:
            print("transformers not installed, running in demo mode")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def predict(self, text):
        """Predict classification for given text."""
        if not self.model_loaded or self.classifier is None:
            # Demo mode - return simulated results
            return self.demo_predict(text)
        
        try:
            result = self.classifier(text)
            # Handle different return formats
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict):
                    # Get highest confidence
                    best = max(result, key=lambda x: x.get('score', 0))
                    return best.get('label', 'UNKNOWN'), best.get('score', 0)
                else:
                    return result[0].get('label', 'UNKNOWN'), result[0].get('score', 0)
            elif isinstance(result, dict):
                return result.get('label', 'UNKNOWN'), result.get('score', 0)
            else:
                # Single result string
                return str(result), 0.9
        except Exception as e:
            print(f"Prediction error: {e}")
            return self.demo_predict(text)
    
    def demo_predict(self, text):
        """Demo prediction when no model is trained."""
        text_lower = text.lower()
        
        # Simple keyword-based demo classification
        threat_keywords = ['kill', 'die', 'hurt', 'hit', 'attack', 'weapon', 'gun', 'knife']
        hate_keywords = ['hate', 'stupid', 'idiot', 'retard', 'disgusting', 'ugly']
        harassment_keywords = ['follow', 'stalk', 'annoy', 'bother', 'leave me alone']
        
        if any(kw in text_lower for kw in threat_keywords):
            return "threat", 0.85
        elif any(kw in text_lower for kw in hate_keywords):
            return "hatred", 0.78
        elif any(kw in text_lower for kw in harassment_keywords):
            return "harassment", 0.72
        elif len(text) > 100:
            return "normal", 0.65
        elif any(word in text_lower for word in ['love', 'great', 'awesome', 'good']):
            return "normal", 0.80
        else:
            return "normal", 0.55


# Create the main application
class CyberbullyingUI:
    """Main GUI application window."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🛡️ Cyberbullying Detection - Text Classifier")
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        
        # Initialize classifier
        self.classifier = CyberbullyingClassifier()
        
        # Setup UI
        self.setup_styling()
        self.create_widgets()
        self.setup_layout()
        
    def setup_styling(self):
        """Configure UI styles."""
        self.root.option_add('*Font', 'Helvetica 11')
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Predict.TButton', foreground='blue', font=('Helvetica', 10, 'bold'))
        style.map('Predict.TButton',
                  background=[('active', '#e6f2ff')])
    
    def create_widgets(self):
        """Create all UI widgets."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="🛡️ Cyberbullying Detection",
                               font=('Helvetica', 16, 'bold'))
        title_label.pack(pady=(0, 15))
        
        subtitle_label = ttk.Label(main_frame, 
                                   text="Enter text to classify as cyberbullying or normal",
                                   font=('Helvetica', 10))
        subtitle_label.pack(pady=(0, 20))
        
        # Text input area
        input_frame = ttk.LabelFrame(main_frame, text="📝 Text Input", padding=10)
        input_frame.pack(fill=tk.X, pady=5)
        
        self.text_input = tk.Text(input_frame, height=4, font=('Helvetica', 11))
        self.text_input.pack(fill=tk.X, expand=True)
        self.text_input.insert('1.0', "Type your text here...")
        
        # Scrollbar for text input
        scrollbar = ttk.Scrollbar(input_frame, orient=tk.HORIZONTAL, command=self.text_input.xview)
        scrollbar.pack(fill=tk.X)
        self.text_input.configure(xscrollcommand=scrollbar.set)
        
        # Buttons frame
        btn_frame = ttk.Frame(main_frame, padding=(10, 5))
        btn_frame.pack(fill=tk.X, pady=5)
        
        # Predict button
        self.predict_btn = ttk.Button(btn_frame, text="🔍 Classify Text",
                                     command=self.classify_text,
                                     style='Predict.TButton')
        self.predict_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Clear button
        clear_btn = ttk.Button(btn_frame, text="Clear", command=self.clear_input,
                              width=10)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Results area
        results_frame = ttk.LabelFrame(main_frame, text="📊 Classification Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Result label
        self.result_label = ttk.Label(results_frame, text="---",
                                     font=('Helvetica', 14, 'bold'),
                                     foreground='darkgreen')
        self.result_label.pack()
        
        # Confidence label
        self.confidence_label = ttk.Label(results_frame, text="",
                                         font=('Helvetica', 11))
        self.confidence_label.pack()
        
        # Details frame
        details_frame = ttk.Frame(results_frame, padding=(0, 5))
        details_frame.pack(fill=tk.X)
        
        ttk.Label(details_frame, text="Confidence:").pack(side=tk.LEFT)
        self.confidence_value = ttk.Label(details_frame, text="--")
        self.confidence_value.pack(side=tk.LEFT, padx=(5, 0))
        
        # Prediction details
        self.details_label = ttk.Label(results_frame, text="",
                                        font=('Helvetica', 9),
                                        foreground='gray')
        self.details_label.pack(pady=(5, 0))
        
        # Instruction label
        inst_label = ttk.Label(main_frame, 
                               text="Tip: Enter text like 'I will kill you', 'You are stupid', or normal conversation",
                               font=('Helvetica', 9), foreground='gray')
        inst_label.pack(pady=(5, 0))
    
    def setup_layout(self):
        """Setup window layout and bindings."""
        self.root.bind('<Return>', lambda e: self.classify_text())
        self.text_input.focus_set()
    
    def classify_text(self):
        """Handle classify button click."""
        text = self.text_input.get('1.0', 'end-1c').strip()
        
        if not text or text == "Type your text here...":
            messagebox.showwarning("Input Required", "Please enter some text to classify.")
            return
        
        # Update UI to show processing
        self.predict_btn.configure(state='disabled')
        self.result_label.configure(text="Analyzing...", foreground='gray')
        self.confidence_label.configure(text="")
        self.details_label.configure(text="")
        
        # Process in threads to not freeze UI
        import threading
        def run_prediction():
            label, score = self.classifier.predict(text)
            
            # Update UI on main thread
            self.root.after(0, lambda: self.update_results(label, score))
        
        thread = threading.Thread(target=run_prediction)
        thread.start()
    
    def update_results(self, label, score):
        """Update UI with prediction results."""
        self.predict_btn.configure(state='normal')
        
        # Map labels to readable format
        label_map = {
            'threat': 'Threat',
            'hatred': 'Hatred/Insult', 
            'harassment': 'Harassment',
            'normal': 'Normal/OK'
        }
        
        display_label = label_map.get(label, label.capitalize())
        
        # Set result colors
        if display_label == "Threat":
            color = 'red'
        elif display_label in ["Hatred/Insult", "Harassment"]:
            color = 'orange'
        else:
            color = 'green'
        
        self.result_label.configure(text=display_label, foreground=color)
        self.confidence_label.configure(text=f"Confidence: {score:.2%}")
        
        # Show details
        if score > 0.7:
            self.details_label.configure(text="High confidence classification")
        elif score > 0.5:
            self.details_label.configure(text="Moderate confidence")
        else:
            self.details_label.configure(text="Low confidence - may be normal text")
    
    def clear_input(self):
        """Clear the text input field."""
        self.text_input.delete('1.0', tk.END)
        self.text_input.insert('1.0', "Type your text here...")
        self.result_label.configure(text="---", foreground='darkgreen')
        self.confidence_label.configure(text="")
        self.details_label.configure(text="")
        self.text_input.focus_set()


def main():
    """Entry point for the application."""
    root = tk.Tk()
    app = CyberbullyingUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()