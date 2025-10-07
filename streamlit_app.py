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

# Enhanced CSS styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;500;600&display=swap');
    
    /* Main container */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="white" opacity="0.1"/><circle cx="75" cy="75" r="1" fill="white" opacity="0.1"/><circle cx="50" cy="10" r="0.5" fill="white" opacity="0.1"/><circle cx="10" cy="60" r="0.5" fill="white" opacity="0.1"/><circle cx="90" cy="40" r="0.5" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
        opacity: 0.3;
    }
    
    .main-header h1 {
        font-family: 'Inter', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        font-weight: 400;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        position: relative;
        z-index: 1;
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
        position: relative;
        z-index: 1;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    .status-dot.healthy {
        background-color: #4ade80;
    }
    
    .status-dot.unhealthy {
        background-color: #f87171;
    }
    
    .status-dot.unknown {
        background-color: #fbbf24;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Input/Output sections */
    .input-section, .output-section {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        margin-bottom: 1.5rem;
    }
    
    .section-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
    }
    
    .section-header::before {
        content: '';
        width: 4px;
        height: 24px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 2px;
        margin-right: 12px;
    }
    
    /* Text areas */
    .urdu-textarea {
        font-family: 'Noto Nastaliq Urdu', 'Jameel Noori Nastaleeq', serif !important;
        font-size: 1.2rem !important;
        direction: rtl !important;
        text-align: right !important;
        border: 2px solid #e5e7eb !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .urdu-textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    .roman-text {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: #1f2937;
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        line-height: 1.6;
    }
    
    /* Buttons */
    .translate-btn, .clear-btn, .copy-btn {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.3s ease !important;
        border: none !important;
        cursor: pointer !important;
    }
    
    .translate-btn {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
    }
    
    .translate-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3) !important;
    }
    
    .translate-btn:disabled {
        background: #9ca3af !important;
        cursor: not-allowed !important;
        transform: none !important;
    }
    
    .clear-btn {
        background: #f3f4f6 !important;
        color: #6b7280 !important;
        border: 1px solid #d1d5db !important;
    }
    
    .clear-btn:hover {
        background: #e5e7eb !important;
        transform: translateY(-1px) !important;
    }
    
    .copy-btn {
        background: #10b981 !important;
        color: white !important;
        font-size: 0.9rem !important;
        padding: 0.5rem 1rem !important;
    }
    
    .copy-btn:hover {
        background: #059669 !important;
        transform: translateY(-1px) !important;
    }
    
    /* Sample buttons */
    .sample-btn {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem !important;
        padding: 0.5rem 1rem !important;
        margin: 0.25rem !important;
        border-radius: 20px !important;
        border: 2px solid #e5e7eb !important;
        background: white !important;
        color: #6b7280 !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
    }
    
    .sample-btn:hover {
        border-color: #667eea !important;
        background: #667eea !important;
        color: white !important;
        transform: translateY(-1px) !important;
    }
    
    /* Loading spinner */
    .loading-spinner {
        display: inline-block;
        width: 16px;
        height: 16px;
        border: 2px solid #ffffff40;
        border-radius: 50%;
        border-top-color: #ffffff;
        animation: spin 1s ease-in-out infinite;
        margin-right: 8px;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Error banner */
    .error-banner {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
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
        color: #dc2626;
    }
    
    /* Input info */
    .input-info {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 0.5rem;
        font-size: 0.9rem;
        color: #6b7280;
    }
    
    /* Footer */
    .app-footer {
        text-align: center;
        color: #6b7280;
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 1px solid #e5e7eb;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .main-header p {
            font-size: 1rem;
        }
        
        .input-section, .output-section {
            padding: 1.5rem;
        }
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

# Sample texts
SAMPLE_TEXTS = [
    "کس کی نمود کے لیے شام و سحر ہیں گرم",
    "دل سے نکلے ہیں جو لفظ اثر رکھتے ہیں",
    "محبت کرنے والے کم نہیں",
    "یہ ایک خوبصورت دن ہے",
    "زندگی ایک سفر ہے",
    "اردو زبان بہت خوبصورت ہے",
    "شاعری دل کی آواز ہے"
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
            <span class="status-dot healthy"></span>
            <span>API: Connected</span>
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
        
        st.markdown("### 📝 Quick Test Samples")
        for i, sample in enumerate(SAMPLE_TEXTS):
            if st.button(f"Sample {i+1}", key=f"sample_{i}", use_container_width=True):
                load_sample_text(sample)
                st.rerun()

    # Main content
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📝 Input Text</div>', unsafe_allow_html=True)
        
        st.markdown("**Enter Urdu Text:**")
        input_text = st.text_area(
            "",
            value=st.session_state.input_text,
            height=200,
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
        
        col1_1, col1_2 = st.columns([2, 1])
        with col1_1:
            if st.button("🚀 Translate", key="translate_btn", use_container_width=True, 
                        disabled=not st.session_state.model_loaded or not input_text.strip()):
                handle_translate()
                st.rerun()
        
        with col1_2:
            if st.button("🗑️ Clear", key="clear_btn", use_container_width=True):
                clear_all()
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="output-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📊 Translation Result</div>', unsafe_allow_html=True)
        
        if st.session_state.output_text:
            st.markdown("**Roman-Urdu Output:**")
            
            col2_1, col2_2 = st.columns([3, 1])
            with col2_1:
                st.markdown(f'<div class="roman-text">{st.session_state.output_text}</div>', unsafe_allow_html=True)
            with col2_2:
                if st.button("📋 Copy", key="copy_btn", use_container_width=True):
                    st.write("Copied to clipboard!")
            
            st.code(st.session_state.output_text, language=None)
            
            st.download_button(
                label="📥 Download Translation",
                data=st.session_state.output_text,
                file_name="translation.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.info("Enter Urdu text and click Translate to see the result")
        
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

    # Footer
    st.markdown("""
    <div class="app-footer">
        <p>
            Built with ❤️ using React + TypeScript + FastAPI + PyTorch<br>
            Assignment: Neural Machine Translation (BiLSTM Encoder + LSTM Decoder)
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()