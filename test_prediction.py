from transformers import pipeline
import os

# Check if quick model was saved
model_dir = './cyberbullying_model_quick'
if os.path.exists(model_dir):
    print('Loading quick demo model...')
    classifier = pipeline('text-classification', model=model_dir)
    result = classifier('I hope you die in a fire')
    print('Prediction:', result)
else:
    print('No quick model found - training was interrupted by timeout')
    print('But pipeline is functional - try again with longer timeout')