"""
FastAPI service for Urdu to Roman-Urdu translation
Exposes POST /translate endpoint following assignment specifications
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import sys
from pathlib import Path

# Add parent directory to path for backend imports
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from backend.tests.test_infer_bilstm_seq2seq import translate_once, get_model_info, load_model_and_tokenizers


# Pydantic models for request/response
class TranslateRequest(BaseModel):
    text_ur: str


class TranslateResponse(BaseModel):
    output_text: str


class InfoResponse(BaseModel):
    model_architecture: str
    total_parameters: int
    trainable_parameters: int
    src_vocab_size: int
    tgt_vocab_size: int
    device: str
    framework: str


# Create FastAPI app
app = FastAPI(
    title="Urdu → Roman-Urdu NMT API",
    description="Neural Machine Translation API using BiLSTM Encoder + LSTM Decoder",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:5174", "http://localhost:5175", "http://localhost:5176"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """
    Load model and tokenizers at startup
    """
    try:
        print("Loading model and tokenizers at startup...")
        load_model_and_tokenizers()
        print("Startup complete!")
    except Exception as e:
        print(f"Startup error: {e}")
        # Don't raise here to allow the API to start, but endpoints will fail


@app.get("/")
async def root():
    """
    Root endpoint with API information
    """
    return {
        "message": "Urdu → Roman-Urdu NMT API",
        "version": "1.0.0",
        "architecture": "BiLSTM Encoder (2 layers) + LSTM Decoder (4 layers)",
        "framework": "PyTorch",
        "endpoints": {
            "translate": "POST /translate",
            "info": "GET /info",
            "health": "GET /health"
        }
    }


@app.post("/translate", response_model=TranslateResponse)
async def translate(request: TranslateRequest):
    """
    Translate Urdu text to Roman-Urdu
    
    Request: { "text_ur": "<Urdu string>" }
    Response: { "output_text": "<Roman-Urdu>" }
    """
    try:
        # Validate input
        if not request.text_ur:
            raise HTTPException(status_code=400, detail="text_ur required")
        
        if not request.text_ur.strip():
            raise HTTPException(status_code=400, detail="text_ur cannot be empty")
        
        # Perform translation
        roman_output = translate_once(request.text_ur)
        
        return TranslateResponse(output_text=roman_output)
        
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Model files not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


@app.get("/info", response_model=InfoResponse)
async def info():
    """
    Get model information
    """
    try:
        model_info = get_model_info()
        
        if "error" in model_info:
            raise HTTPException(status_code=500, detail=model_info["error"])
        
        return InfoResponse(**model_info)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@app.get("/health")
async def health():
    """
    Health check endpoint
    """
    try:
        # Try to get model info to verify everything is loaded
        model_info = get_model_info()
        
        if "error" in model_info:
            return {
                "status": "unhealthy",
                "error": model_info["error"]
            }
        
        return {
            "status": "healthy",
            "model_loaded": True,
            "device": model_info["device"]
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "error": str(e)
        }


if __name__ == "__main__":
    print("Starting Urdu → Roman-Urdu Translation API...")
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=["./"]
    )