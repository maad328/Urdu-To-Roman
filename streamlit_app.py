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
    page_title="Urdu to Roman-Urdu Translator",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme CSS styling to match your design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;500;600&display=swap');
    
    /* Main page background */
    .stApp {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 30%, #e91e63 100%);
        min-height: 100vh;
    }
    
    /* Main container */
    .main-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        color: white;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        font-weight: 400;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* API Status */
    .api-status {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-top: 1rem;
        padding: 0.5rem 1rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 25px;
        backdrop-filter: blur(10px);
        width: fit-content;
        margin-left: auto;
        margin-right: auto;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
        background-color: #4ade80;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Card styling */
    .input-card, .output-card {
        background: rgba(30, 30, 50, 0.95);
        border-radius: 15px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .card-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        font-weight: 600;
        color: white;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
    }
    
    .card-header::before {
        content: '📄';
        margin-right: 8px;
        font-size: 1.1rem;
    }
    
    /* Text areas */
    .urdu-textarea {
        font-family: 'Noto Nastaliq Urdu', 'Jameel Noori Nastaleeq', serif !important;
        font-size: 1.1rem !important;
        direction: rtl !important;
        text-align: right !important;
        background: rgba(40, 40, 60, 0.8) !important;
        border: 2px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        color: white !important;
        transition: all 0.3s ease !important;
        min-height: 120px !important;
    }
    
    .urdu-textarea:focus {
        border-color: #e91e63 !important;
        box-shadow: 0 0 0 3px rgba(233, 30, 99, 0.2) !important;
        background: rgba(50, 50, 70, 0.9) !important;
    }
    
    .urdu-textarea::placeholder {
        color: rgba(255, 255, 255, 0.6) !important;
    }
    
    .roman-text {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: white;
        background: rgba(40, 40, 60, 0.8);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid rgba(255, 255, 255, 0.2);
        line-height: 1.6;
        min-height: 120px;
    }
    
    /* Labels */
    .input-label, .output-label {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        color: white;
        margin-bottom: 0.5rem;
        display: block;
    }
    
    /* Input info */
    .input-info {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 0.5rem;
        font-size: 0.85rem;
        color: rgba(255, 255, 255, 0.7);
        font-family: 'Inter', sans-serif;
    }
    
    /* Sample buttons */
    .sample-buttons {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 1rem;
    }
    
    .sample-btn {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.85rem !important;
        padding: 0.5rem 1rem !important;
        border-radius: 20px !important;
        border: 2px solid rgba(233, 30, 99, 0.3) !important;
        background: rgba(233, 30, 99, 0.1) !important;
        color: white !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
        white-space: nowrap !important;
    }
    
    .sample-btn:hover {
        border-color: #e91e63 !important;
        background: rgba(233, 30, 99, 0.2) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Action buttons */
    .button-container {
        display: flex;
        gap: 1rem;
        margin-top: 1.5rem;
    }
    
    .translate-btn {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        transition: all 0.3s ease !important;
        border: none !important;
        cursor: pointer !important;
        background: linear-gradient(135deg, #e91e63, #ad1457) !important;
        color: white !important;
        flex: 2;
    }
    
    .translate-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(233, 30, 99, 0.4) !important;
    }
    
    .translate-btn:disabled {
        background: rgba(255, 255, 255, 0.2) !important;
        cursor: not-allowed !important;
        transform: none !important;
    }
    
    .clear-btn {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.3s ease !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        background: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        cursor: pointer !important;
        flex: 1;
    }
    
    .clear-btn:hover {
        background: rgba(255, 255, 255, 0.2) !important;
        transform: translateY(-1px) !important;
    }
    
    .copy-btn {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.3s ease !important;
        border: 2px solid rgba(76, 175, 80, 0.3) !important;
        background: rgba(76, 175, 80, 0.1) !important;
        color: white !important;
        cursor: pointer !important;
    }
    
    .copy-btn:hover {
        background: rgba(76, 175, 80, 0.2) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Loading spinner */
    .loading-spinner {
        display: inline-block;
        width: 16px;
        height: 16px;
        border: 2px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        border-top-color: white;
        animation: spin 1s ease-in-out infinite;
        margin-right: 8px;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Error banner */
    .error-banner {
        background: rgba(244, 67, 54, 0.1);
        border: 1px solid rgba(244, 67, 54, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        color: #ffcdd2;
    }
    
    .error-icon {
        font-size: 1.2rem;
        margin-right: 0.5rem;
    }
    
    .error-close {
        background: none;
        border: none;
        font-size: 1.5rem;
        cursor: pointer;
        color: #ffcdd2;
    }
    
    /* Output section styling */
    .output-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(20, 20, 40, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .input-card, .output-card {
            padding: 1.5rem;
        }
        
        .sample-buttons {
            flex-direction: column;
        }
        
        .button-container {
            flex-direction: column;
        }
    }
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

# Sample texts
SAMPLE_TEXTS = [
    "زندگی ایک سفر ہے",
    "یہ ایک خوبصورت دن ہے", 
    "محبت کرنے والے کم نہیں",
    "دل سے نکلے ہیں جو لفظ اثر رکھتے ہیں",
    "کس کی نمود کے لیے شام و سحر ہیں گرم"
]

def load_model():
    """Load the model and tokenizers"""
    try:
        with st.spinner("Loading model and tokenizers..."):
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
        st.session_state.error = 'Please load the model first from the sidebar'
        return
    
    try:
        st.session_state.error = ''
        result = translate_once(st.session_state.input_text.strip())
        st.session_state.output_text = result
    except Exception as e:
        st.session_state.error = f"Translation failed: {str(e)}"

def clear_all():
    """Clear all inputs and outputs"""
    st.session_state.input_text = ''
    st.session_state.output_text = ''
    st.session_state.error = ''

def load_sample_text(text):
    """Load sample text"""
    st.session_state.input_text = text
    st.session_state.error = ''

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔤 Urdu → Roman Urdu</h1>
        <p>Neural Machine Translation using BiLSTM Encoder + LSTM Decoder</p>
        
        <div class="api-status">
            <span class="status-dot"></span>
            <span>API: Connected (cpu)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Model Information")
        
        if not st.session_state.model_loaded:
            if st.button("🚀 Load Model", key="load_model_btn", use_container_width=True):
                if load_model():
                    st.success("Model loaded successfully!")
                    st.rerun()
        else:
            st.success("✅ Model Loaded")
            
            if st.session_state.model_info:
                info = st.session_state.model_info
                st.markdown(f"**Architecture:** {info.get('model_architecture', 'N/A')}")
                st.markdown(f"**Parameters:** {info.get('total_parameters', 0):,}")
                st.markdown(f"**Device:** {info.get('device', 'N/A')}")

    # Input Section
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">Input Text</div>', unsafe_allow_html=True)
    
    st.markdown('<span class="input-label">Enter Urdu Text:</span>', unsafe_allow_html=True)
    input_text = st.text_area(
        "",
        value=st.session_state.input_text,
        height=120,
        placeholder="یہاں اردو متن لکھیں... (Type your Urdu text here)",
        key="input_textarea",
        help="Type your Urdu text here"
    )
    
    st.session_state.input_text = input_text
    
    st.markdown(f"""
    <div class="input-info">
        <span>Characters: {len(input_text)}</span>
        <span>Click Translate or press Ctrl+Enter</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample buttons
    st.markdown('<div class="sample-buttons">', unsafe_allow_html=True)
    for i, sample in enumerate(SAMPLE_TEXTS):
        if st.button(sample, key=f"sample_{i}"):
            load_sample_text(sample)
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Action buttons
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🚀 TRANSLATE", key="translate_btn", use_container_width=True, 
                    disabled=not st.session_state.model_loaded or not input_text.strip()):
            handle_translate()
            st.rerun()
    
    with col2:
        if st.button("CLEAR", key="clear_btn", use_container_width=True):
            clear_all()
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Output Section
    if st.session_state.output_text:
        st.markdown('<div class="output-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Translation Result</div>', unsafe_allow_html=True)
        
        st.markdown('<span class="output-label">Roman-Urdu Output:</span>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f'<div class="roman-text">{st.session_state.output_text}</div>', unsafe_allow_html=True)
        with col2:
            if st.button("📋 Copy", key="copy_btn", use_container_width=True):
                st.write("Copied to clipboard!")
        
        st.download_button(
            label="📥 Download Translation",
            data=st.session_state.output_text,
            file_name="translation.txt",
            mime="text/plain",
            use_container_width=True
        )
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Error banner
    if st.session_state.error:
        st.markdown(f"""
        <div class="error-banner">
            <span>
                <span class="error-icon">⚠️</span>
                {st.session_state.error}
            </span>
            <button class="error-close" onclick="this.parentElement.style.display='none'">×</button>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()