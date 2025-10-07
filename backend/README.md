# Backend

Python backend for Urdu to Roman-Urdu translation using PyTorch BiLSTM+LSTM seq2seq model.

## Structure
```
backend/
  api/                    # FastAPI routers and main server entry
    app.py               # Main FastAPI application
  models/                 # PyTorch model definitions
    seq2seq.py          # Model architecture definitions
    seq2seq_simple.py   # Simplified model definitions  
    best_bilstm_seq2seq3 (7).pth  # Trained model checkpoint
  tokenizers/             # SentencePiece tokenizer files
    urdu_char.model     # Urdu character-level tokenizer
    roman_char.model    # Roman character-level tokenizer
  tests/                  # Testing and inference modules
    test_infer_bilstm_seq2seq.py  # Inference and testing
  utils/                  # Helper utilities
    decode.py           # Decoding utilities
    device.py           # Device management
  requirements.txt        # Python dependencies
```

## Setup

```bash
cd backend
pip install -r requirements.txt
```

## Usage

### Start API Server
```bash
cd backend
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Run Tests
```bash
cd backend
python tests/test_infer_bilstm_seq2seq.py
```

### API Endpoints

- `POST /translate` - Translate Urdu text to Roman-Urdu
- `GET /info` - Get model information
- `GET /health` - Health check

## Model Details

- **Architecture**: BiLSTM Encoder (2 layers) + LSTM Decoder (4 layers)
- **Parameters**: 39.9M trainable parameters
- **Tokenization**: Character-level SentencePiece
- **Vocab Sizes**: Urdu (51), Roman (32)