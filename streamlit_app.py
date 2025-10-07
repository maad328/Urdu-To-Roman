import streamlit as st
import torch
import torch.nn as nn
import sentencepiece as spm
from pathlib import Path
import sys
import os

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

# Import your model and functions
from backend.tests.test_infer_bilstm_seq2seq import translate_once, get_model_info, load_model_and_tokenizers

# Page configuration
st.set_page_config(
    page_title="Roman to Urdu Converter",
    page_icon="🔤",
    layout="centered"
)

# Professional CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;500;600&display=swap');
    
    .stApp {
        background: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    .main-container {
        max-width: 600px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    .header {
        text-align: center;
        margin-bottom: 3rem;
    }
    
    .header h1 {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e293b;
        margin: 0;
    }
    
    .converter-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
    }
    
    .input-section {
        margin-bottom: 2rem;
    }
    
    .input-label {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        color: #374151;
        margin-bottom: 0.75rem;
        display: block;
    }
    
    .urdu-input {
        font-family: 'Noto Nastaliq Urdu', 'Jameel Noori Nastaleeq', serif !important;
        font-size: 1.1rem !important;
        direction: rtl !important;
        text-align: right !important;
        width: 100% !important;
        padding: 1rem !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 8px !important;
        background: #f8fafc !important;
        color: #1e293b !important;
        transition: all 0.2s ease !important;
        min-height: 120px !important;
    }
    
    .urdu-input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
        background: white !important;
    }
    
    .urdu-input::placeholder {
        color: #9ca3af !important;
    }
    
    .translate-btn {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        background: #3b82f6 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }
    
    .translate-btn:hover {
        background: #2563eb !important;
        transform: translateY(-1px) !important;
    }
    
    .translate-btn:disabled {
        background: #9ca3af !important;
        cursor: not-allowed !important;
        transform: none !important;
    }
    
    .result-section {
        margin-top: 2rem;
        padding-top: 2rem;
        border-top: 1px solid #e2e8f0;
    }
    
    .result-label {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        color: #374151;
        margin-bottom: 0.75rem;
        display: block;
    }
    
    .roman-output {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: #1e293b;
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        min-height: 120px;
        line-height: 1.6;
    }
    
    .error-message {
        background: #fef2f2;
        border: 1px solid #fecaca;
        color: #dc2626;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        font-family: 'Inter', sans-serif;
    }
    
    .success-message {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        color: #16a34a;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        font-family: 'Inter', sans-serif;
    }
    
    .load-model-btn {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        background: #10b981 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
        margin-bottom: 1rem;
    }
    
    .load-model-btn:hover {
        background: #059669 !important;
        transform: translateY(-1px) !important;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model_info' not in st.session_state:
    st.session_state.model_info = None
if 'input_text' not in st.session_state:
    st.session_state.input_text = ''
if 'output_text' not in st.session_state:
    st.session_state.output_text = ''
if 'error' not in st.session_state:
    st.session_state.error = ''

def load_model():
    """Load the model and tokenizers"""
    try:
        with st.spinner("Loading model..."):
            load_model_and_tokenizers()
            st.session_state.model_loaded = True
            st.session_state.model_info = get_model_info()
        return True
    except Exception as e:
        st.session_state.error = f"Error loading model: {str(e)}"
        return False

def handle_translate():
    """Handle translation"""
    if not st.session_state.input_text.strip():
        st.session_state.error = 'Please enter some Urdu text to translate'
        return
    
    if not st.session_state.model_loaded:
        st.session_state.error = 'Please load the model first'
        return
    
    try:
        st.session_state.error = ''
        result = translate_once(st.session_state.input_text.strip())
        st.session_state.output_text = result
    except Exception as e:
        st.session_state.error = f"Translation failed: {str(e)}"

def main():
    # Header
    st.markdown("""
    <div class="header">
        <h1>Roman to Urdu Converter</h1>
    </div>
    """, unsafe_allow_html=True)

    # Main converter
    st.markdown('<div class="converter-card">', unsafe_allow_html=True)
    
    # Model loading section
    if not st.session_state.model_loaded:
        st.markdown('<div class="success-message">', unsafe_allow_html=True)
        st.markdown('**Model Status:** Not Loaded', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("Load Model", key="load_model_btn", use_container_width=True):
            if load_model():
                st.success("Model loaded successfully!")
                st.rerun()
    else:
        st.markdown('<div class="success-message">', unsafe_allow_html=True)
        st.markdown('✅ **Model Status:** Ready for Translation', unsafe_allow_html=True)
        if st.session_state.model_info:
            info = st.session_state.model_info
            st.markdown(f"**Parameters:** {info.get('total_parameters', 0):,} | **Device:** {info.get('device', 'N/A')}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Input section
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<span class="input-label">Enter Urdu Text:</span>', unsafe_allow_html=True)
    
    input_text = st.text_area(
        "",
        value=st.session_state.input_text,
        height=120,
        placeholder="یہاں اردو متن لکھیں...",
        key="input_textarea",
        help="Type your Urdu text here"
    )
    
    st.session_state.input_text = input_text
    
    # Translate button
    if st.button("Translate", key="translate_btn", use_container_width=True, 
                disabled=not st.session_state.model_loaded or not input_text.strip()):
        handle_translate()
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Result section
    if st.session_state.output_text:
        st.markdown('<div class="result-section">', unsafe_allow_html=True)
        st.markdown('<span class="result-label">Roman-Urdu Output:</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="roman-output">{st.session_state.output_text}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Error message
    if st.session_state.error:
        st.markdown(f'<div class="error-message">⚠️ {st.session_state.error}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()