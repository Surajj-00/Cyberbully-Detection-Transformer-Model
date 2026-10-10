#!/usr/bin/env python
"""
🛡️ Cyberbullying Detection - Web UI
Open this in your browser: streamlit run streamlit_app.py
"""

import streamlit as st
import os
from transformers import pipeline

st.set_page_config(page_title="Cyberbullying Detection", page_icon="🛡️", layout="centered")

# Minimal CSS
st.markdown("""
<style>
.main {padding: 2rem;}
.stButton > button {width: 100%; background-color: #4CAF50; color: white; font-size: 16px; padding: 8px; border-radius: 6px; border: none;}
.stButton > button:hover {background-color: #45a049;}
.result-box {padding: 15px; border-radius: 8px; margin: 10px 0; font-size: 18px; font-weight: bold; text-align: center;}
.threat {background-color: #ffebee; color: #c62829; border-left: 4px solid #c62829;}
.normal {background-color: #e8f5e9; color: #2e7d32; border-left: 4px solid #2e7d32;}
.title {color: #1a237e; font-size: 20px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">Cyberbullying Detection</h1>', unsafe_allow_html=True)
st.markdown('<p style="color: #555;">An intelligent platform that analyzes text to identify cyberbullying and detect harmful language</p>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_dir = "./cyberbullying_model_quick"
    if os.path.exists(model_dir):
        try:
            classifier = pipeline("text-classification", model=model_dir, top_k=None)
            return classifier, True
        except: return None, False
    else: st.warning("Model not found"); return None, False

classifier, model_loaded = load_model()

if not model_loaded:
    st.info("Model not running. Using demo mode.")

# Initialize label and score for details section
label, score = "normal", 0.5
label_lower = label.lower().replace('label_','').replace('label','')

user_text = st.text_area("", height=200, placeholder="Enter or paste text here...")

if st.button("Classify"):
    if classifier:
        result = classifier(user_text)
        inner = result[0] if isinstance(result[0], list) else result
        best = max(inner, key=lambda x: x.get('score', 0))
        label, score = best.get('label', 'normal'), best.get('score', 0)
    else:
        label, score = ('normal', 0.55) if not any(k in user_text.lower() for k in ['kill','die','hate','stupid']) else ('threat', 0.85)

    label_lower = label.lower().replace('label_','').replace('label','')
    if label_lower == "1":
        desc, cls = "Cyberbullying/toxic content", "threat"
    elif label_lower == "0":
        desc, cls = "Normal, non-bullying content", "normal"
    elif 'threat' in label_lower:
        desc, cls = "Potential threat detected", "threat"
    elif 'harass' in label_lower:
        desc, cls = "Harassment detected", "normal"
    elif 'hatred' in label_lower:
        desc, cls = "Hateful content", "normal"
    else:
        desc, cls = "Normal content", "normal"

    st.markdown(f'<div class="result-box {cls}">{desc}</div>', unsafe_allow_html=True)

# Details section
with st.expander("Details"):
    st.write(f"**Input text:** {user_text[:100]}{'...' if len(user_text) > 100 else ''}")
    st.write(f"**Model output label:** {label}")
    st.write(f"**Confidence score:** {score:.4f}")
    
    # Interpretation
    if label_lower == "1":
        st.error("⚠️ This text contains cyberbullying or toxic content.")
    elif 'threat' in label_lower:
        st.error("⚠️ This text contains potential threats or dangerous content.")
    elif 'harass' in label_lower:
        st.warning("⚠️ This text contains harassment or bullying behavior.")
    elif 'hatred' in label_lower:
        st.warning("⚠️ This text contains hateful or prejudicial content.")
    else:
        st.success("✅ This text appears to be normal, non-bullying content.")

# Example texts section
st.markdown("### Example Texts")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**🔴 Cyberbullying Examples:**")
    st.markdown("- I will kill you")
    st.markdown("- You are so stupid")
    st.markdown("- Go die in a hole")

with col2:
    st.markdown("**🟢 Normal Examples:**")
    st.markdown("- Hello how are you")
    st.markdown("- Have a nice day")
    st.markdown("- I love this product")

