from transformers import pipeline

classifier = pipeline('text-classification', model='./social_media_model')

# Test predictions
test_texts = [
    "I absolutely love this product! Highly recommend!",
    "This is terrible, worst experience ever",
    "It's okay, nothing special"
]

print("Testing saved model:")
for text in test_texts:
    result = classifier(text)
    print(f"Text: {text[:50]}...")
    print(f"Prediction: {result}\n")