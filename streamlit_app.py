import streamlit as st
import torch
import torch.nn as nn
import sentencepiece as spm
from pathlib import Path
import sys
import os

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

from backend.tests.test_infer_bilstm_seq2seq import translate_once, get_model_info, load_model_and_tokenizers

# Page configuration
st.set_page_config(
    page_title="Urdu to Roman-Urdu Translator",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #ff416c, #ff4b2b);
    color: white;
}
.main-header {
    text-align: center;
    padding: 1.5rem 0;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 15px;
    margin-bottom: 2rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
.
.urdu-text {
    font-family: 'Jameel Noori Nastaleeq', 'Noto Nastaliq Urdu', serif;
    font-size: 1.2em;
    direction: rtl;
    text-align: right;
}
.roman-text {
    font-family: 'Courier New', monospace;
    color: #00ffcc;
    background: #132743;
    padding: 1rem;
    border-radius: 10px;
}
.stTextArea textarea {
    font-family: 'Jameel Noori Nastaleeq', 'Noto Nastaliq Urdu', serif !important;
    direction: rtl !important;
    text-align: right !important;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model_info' not in st.session_state:
    st.session_state.model_info = None

# Load model automatically
if not st.session_state.model_loaded:
    with st.spinner("⚙️ Loading model and tokenizers..."):
        try:
            load_model_and_tokenizers()
            st.session_state.model_loaded = True
            st.session_state.model_info = get_model_info()
        except Exception as e:
            st.error(f"Model loading failed: {str(e)}")

def main():
    st.markdown("""
    <div class="main-header">
        <h1> Urdu to Roman-Urdu Translator</h1>
    </div>
    """, unsafe_allow_html=True)


    input_text = st.text_area(
        "Enter Urdu Text",
        height=150,
        placeholder="یہاں اردو متن لکھیں..."
    )

    st.write(f"Characters: {len(input_text)}")

    if st.button(" Translate", use_container_width=True):
        if not st.session_state.model_loaded:
            st.warning("Model not loaded yet. Please wait or reload.")
        else:
            with st.spinner("Translating..."):
                try:
                    result = translate_once(input_text.strip())
                    if result:
                        st.markdown(f'<div class="roman-text">{result}</div>', unsafe_allow_html=True)
                        
                    else:
                        st.warning("Translation returned empty result.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)

    # Sidebar (Model info and reload)
  

    st.markdown("""
    <hr style='margin-top:3rem;'>
    <div style="text-align:center; color:#eee;">
        <a href="https://github.com/maad328/Urdu-To-Roman" target="_blank" style="color:#00ffcc;">GitHub Repo</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
