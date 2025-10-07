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

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .urdu-text {
        font-family: 'Jameel Noori Nastaleeq', 'Noto Nastaliq Urdu', serif;
        font-size: 1.2em;
        direction: rtl;
        text-align: right;
    }
    .roman-text {
        font-family: 'Arial', sans-serif;
        font-size: 1.1em;
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model_info' not in st.session_state:
    st.session_state.model_info = None

# Sample texts
SAMPLE_TEXTS = [
    "کس کی نمود کے لیے شام و سحر ہیں گرم",
    "دل سے نکلے ہیں جو لفظ اثر رکھتے ہیں",
    "محبت کرنے والے کم نہیں",
    "یہ ایک خوبصورت دن ہے",
    "زندگی ایک سفر ہے"
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
        st.error(f"Error loading model: {str(e)}")
        return False

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔤 Urdu to Roman-Urdu Translator</h1>
        <p>Neural Machine Translation using BiLSTM Encoder + LSTM Decoder</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        
        if not st.session_state.model_loaded:
            if st.button("🚀 Load Model"):
                if load_model():
                    st.success("Model loaded successfully!")
                    st.rerun()
        else:
            st.success("✅ Model Loaded")
            
            if st.session_state.model_info:
                info = st.session_state.model_info
                st.write(f"**Architecture:** {info.get('model_architecture', 'N/A')}")
                st.write(f"**Parameters:** {info.get('total_parameters', 0):,}")
                st.write(f"**Device:** {info.get('device', 'N/A')}")
        
        st.header("📝 Quick Samples")
        for i, sample in enumerate(SAMPLE_TEXTS):
            if st.button(f"Sample {i+1}", key=f"sample_{i}"):
                st.session_state.input_text = sample

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📝 Input (Urdu)")
        
        input_text = st.text_area(
            "Enter Urdu text:",
            value=st.session_state.get('input_text', ''),
            height=200,
            placeholder="یہاں اردو متن لکھیں...",
            help="Type your Urdu text here"
        )
        
        st.write(f"Characters: {len(input_text)}")
        translate_btn = st.button("🚀 Translate", type="primary", disabled=not st.session_state.model_loaded)

    with col2:
        st.header("📊 Output (Roman-Urdu)")
        
        if translate_btn and input_text.strip():
            if st.session_state.model_loaded:
                with st.spinner("Translating..."):
                    try:
                        result = translate_once(input_text.strip())
                        
                        if result:
                            st.markdown(f'<div class="roman-text">{result}</div>', unsafe_allow_html=True)
                            st.code(result, language=None)
                            
                            st.download_button(
                                label="📥 Download Translation",
                                data=result,
                                file_name="translation.txt",
                                mime="text/plain"
                            )
                        else:
                            st.warning("Translation returned empty result")
                            
                    except Exception as e:
                        st.error(f"Translation error: {str(e)}")
            else:
                st.warning("Please load the model first from the sidebar")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        Built with ❤️ using Streamlit + PyTorch<br>
        Repository: <a href="https://github.com/maad328/Urdu-To-Roman">GitHub</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()