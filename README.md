# 🛡️ Cyberbullying Detection - Web UI

A web-based interface for detecting cyberbullying in text using transformer models.

## 🚀 Quick Start

```bash
# Install dependencies
pip install streamlit transformers torch

# Run the web app
streamlit run streamlit_app.py

# Opens at: http://localhost:8501
```

## 📋 Features

- **🟢 Demo Mode**: Always active - no model download required
- **🔍 Text Classification**: Detects cyberbullying, threats, harassment, hate speech
- **🟢 Green/Red Results**: Visual color-coded output
- **📊 Confidence Scores**: Shows classification confidence percentage
- **📋 Details Expander**: Shows input text, model label, confidence score
- **📚 Example Texts**: Pre-defined cyberbullying and normal examples

## 📋 How It Works

The app uses **keyword detection** to classify text as cyberbullying or normal content. No model training or download required - works immediately on first run.

**Classification Categories:**
- 🟢 **Normal**: Normal, non-bullying content
- 🟢 **Normal**: Normal, non-bullying content  
- 🟠 **Harassment**: Harassment detected
- 🟠 **Hatred**: Hateful content detected
- 🔴 **Threat/Cyberbullying**: Cyberbullying/toxic content detected

## 🚀 Quick Start

```bash
# Install dependencies
pip install streamlit transformers

# Run the web app
streamlit run streamlit_app.py

# Opens at: http://localhost:8501
```

## 📋 Example Classifications

| Input Text | Result |
|------------|--------|
| "I will kill you" | 🔴 Threat/Cyberbullying |
| "You are so stupid" | 🟠 Harassment detected |
| "Hello how are you" | 🟢 Normal, non-bullying content |
| "I love this product" | 🟢 Normal, non-bullying content |

## 📋 Dataset Information

The system uses keyword-based detection (no model download required):

- **Dataset Size**: Keyword-based rules
- **Classes**: Cyberbullying/toxic, Normal
- **Training**: Keyword-based rules (no model training required)
- **Deployment**: Works immediately on Streamlit Cloud

## 📁 Project Structure

```
transformers-algo-ml/
├── streamlit_app.py          # Main web UI application
├── cyberbullying_ui.py       # Tkinter desktop GUI
├── train_cyberbullying.py    # Full dataset training script
├── train_cyberbullying_quick.py  # Quick demo trainer
├── convert_json_to_csv.py    # JSON to CSV converter
├── .gitignore               # Git ignore rules
├── .gitattributes           # Git LFS tracking
├── social_media_data.csv     # Sample test dataset
└── new datasets/             # Dataset CSV files
```

## 📋 Usage

### Quick Start

```bash
# Install dependencies
pip install streamlit transformers

# Run the web app
streamlit run streamlit_app.py

# Opens at: http://localhost:8501

# Enter text and click "Classify"
# View results with color-coded output and confidence scores
```

### Example Usage

```python
# The app can be used via:
# 1. Web interface: streamlit run streamlit_app.py
# 2. Desktop GUI: python cyberbullying_ui.py
# 3. Command line: python streamlit_app.py
```

## 📋 Troubleshooting

| Issue | Solution |
|-------|----------|
| **App won't start** | Ensure `streamlit` is installed (`pip install streamlit`) |
| **Classification not working** | Check internet connection for model downloads |
| **Unexpected results** | Keyword detection may not catch all cases |
| **Performance slow** | Close other applications, reduce system load |

## 📁 Project Structure

```
transformers-algo-ml/
├── streamlit_app.py          # Main web UI application (UPDATED)
├── cyberbullying_ui.py       # Tkinter desktop GUI
├── train_cyberbullying.py    # Full dataset training script
├── train_cyberbullying_quick.py  # Quick demo trainer
├── convert_json_to_csv.py    # JSON to CSV converter
├── .gitignore               # Git ignore rules
├── .gitattributes           # Git LFS tracking
├── social_media_data.csv     # Sample test dataset (55 rows)
├── new datasets/             # Dataset CSV files
└── README.md                # Project documentation
```

## 📋 Installation

```bash
# Install required dependencies
pip install streamlit transformers torch

# Run the web app
streamlit run streamlit_app.py

# Opens at: http://localhost:8501
```

## 📋 Implementation Details

- **Language**: Python 3.x
- **Framework**: Streamlit for web UI
- **Model**: Keyword-based detection (no model download)
- **Classification**: Binary (cyberbullying vs normal)
- **UI**: Streamlit web interface

## 📞 Support

- **Issues**: Check the troubleshooting section above
- **Questions**: Refer to the project documentation
- **Expansion**: Can be expanded with model training later

## 📜 License

This project is for educational and research purposes.

---

*Built with Streamlit & Transformers for cyberbullying detection research*