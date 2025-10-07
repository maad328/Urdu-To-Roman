"""
Test file to load model/tokenizers and perform ONE translation pass
Loads the trained BiLSTM encoder + LSTM decoder model for inference
Updated with proper model architecture matching training
"""
import os
import torch
import torch.nn as nn
import sentencepiece as spm
from pathlib import Path

# ---------------- CONFIG (match your training) ----------------
EMBED_DIM = 256
HIDDEN_DIM = 512
ENC_LAYERS = 2
DEC_LAYERS = 4
DROPOUT = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables for loaded model and tokenizers
_model = None
_src_tokenizer = None
_tgt_tokenizer = None
_device = None
_bos_id = None
_eos_id = None

# ---------------- MODELS ----------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout, pad_id_src):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id_src)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            dropout=dropout, bidirectional=True, batch_first=True
        )

    def forward(self, src_ids):
        # src_ids: [B, T]
        emb = self.embedding(src_ids)              # [B, T, E]
        outputs, (h, c) = self.lstm(emb)           # h,c: [2*num_layers, B, H]
        # Take last layer's forward/backward states and concat along hidden dim -> [B, 2H]
        h_cat = torch.cat((h[-2], h[-1]), dim=1).unsqueeze(0)   # [1, B, 2H]
        c_cat = torch.cat((c[-2], c[-1]), dim=1).unsqueeze(0)   # [1, B, 2H]
        return outputs, (h_cat, c_cat)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_2x, num_layers, dropout, pad_id_tgt):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id_tgt)
        self.lstm = nn.LSTM(
            embed_dim, hidden_2x, num_layers=num_layers,
            dropout=dropout, batch_first=True
        )
        self.fc_out = nn.Linear(hidden_2x, vocab_size)

    def forward(self, dec_in, hidden, cell):
        # dec_in: [B, 1]
        emb = self.embedding(dec_in)               # [B, 1, E]
        out, (h, c) = self.lstm(emb, (hidden, cell))   # out: [B, 1, 2H]
        logits = self.fc_out(out)                  # [B, 1, V]
        return logits, (h, c)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, dec_layers):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.dec_layers = dec_layers

    def forward(self, src, tgt_in):
        # Used during training; not needed for greedy test, but kept for shape parity
        _, (h, c) = self.encoder(src)          # [1, B, 2H]
        h = h.repeat(self.dec_layers, 1, 1)    # [L, B, 2H]
        c = c.repeat(self.dec_layers, 1, 1)
        # One-shot teacher-forced forward (not decoding loop)
        logits, _ = self.decoder(tgt_in, h, c)
        return logits


def load_model_and_tokenizers():
    """
    Load the trained model and tokenizers once at startup
    """
    global _model, _src_tokenizer, _tgt_tokenizer, _device, _bos_id, _eos_id
    
    if _model is not None:
        return _model, _src_tokenizer, _tgt_tokenizer, _device, _bos_id, _eos_id
    
    try:
        # Get paths - Using organized folder structure
        backend_dir = Path(__file__).parent.parent  # backend directory
        project_root = backend_dir.parent  # Go up to actual project root (urdu_ghazals_rekhta)
        
        # Use properly organized model and tokenizers in backend subdirs
        model_path = backend_dir / "models" / "best_bilstm_seq2seq3 (7).pth"
        src_tokenizer_path = backend_dir / "tokenizers" / "urdu_char.model"
        tgt_tokenizer_path = backend_dir / "tokenizers" / "roman_char.model"
        
        # Check if files exist
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
        if not src_tokenizer_path.exists():
            raise FileNotFoundError(f"Source tokenizer not found at {src_tokenizer_path}")
        if not tgt_tokenizer_path.exists():
            raise FileNotFoundError(f"Target tokenizer not found at {tgt_tokenizer_path}")
        
        print("Loading tokenizers...")
        print(f"Source tokenizer: {src_tokenizer_path}")
        print(f"Target tokenizer: {tgt_tokenizer_path}")
        
        # Load tokenizers
        _src_tokenizer = spm.SentencePieceProcessor()
        _src_tokenizer.load(str(src_tokenizer_path))
        
        _tgt_tokenizer = spm.SentencePieceProcessor()
        _tgt_tokenizer.load(str(tgt_tokenizer_path))
        
        print(f"Source tokenizer vocab size: {_src_tokenizer.vocab_size()}")
        print(f"Target tokenizer vocab size: {_tgt_tokenizer.vocab_size()}")
        
        # Get special token IDs
        _bos_id = _tgt_tokenizer.piece_to_id("<s>")
        _eos_id = _tgt_tokenizer.piece_to_id("</s>")
        pad_id_src = _src_tokenizer.piece_to_id("<pad>")
        pad_id_tgt = _tgt_tokenizer.piece_to_id("<pad>")
        
        # Set device
        _device = DEVICE
        print(f"Using {_device} device")
        
        # Build model
        print("Loading model...")
        enc = Encoder(
            _src_tokenizer.get_piece_size(), EMBED_DIM, HIDDEN_DIM, ENC_LAYERS, DROPOUT, pad_id_src
        ).to(_device)
        dec = Decoder(
            _tgt_tokenizer.get_piece_size(), EMBED_DIM, HIDDEN_DIM*2, DEC_LAYERS, DROPOUT, pad_id_tgt
        ).to(_device)
        _model = Seq2Seq(enc, dec, DEC_LAYERS).to(_device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=_device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            _model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            _model.load_state_dict(checkpoint['state_dict'])
        else:
            _model.load_state_dict(checkpoint)
        
        _model.eval()
        
        print(f"Model loaded successfully on {_device}")
        
        return _model, _src_tokenizer, _tgt_tokenizer, _device, _bos_id, _eos_id
        
    except Exception as e:
        print(f"Error loading model and tokenizers: {e}")
        raise


# ---------------- GREEDY DECODING ----------------
@torch.no_grad()
def greedy_decode(model, ur_sp, ro_sp, text, bos_id, eos_id, max_len=180):
    """
    Greedy decoding function for translation
    """
    # Encode source to IDs
    src_ids = ur_sp.encode(text, out_type=int)
    src = torch.tensor(src_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)  # [1, T]
    
    # Run encoder
    _, (h, c) = model.encoder(src)       # [1, 1, 2H]
    h = h.repeat(DEC_LAYERS, 1, 1)       # [L, 1, 2H]
    c = c.repeat(DEC_LAYERS, 1, 1)

    # Start with BOS
    cur = torch.tensor([[bos_id]], dtype=torch.long, device=DEVICE)  # [1,1]
    out_ids = []

    for _ in range(max_len):
        logits, (h, c) = model.decoder(cur, h, c)  # logits: [1,1,V]
        nxt = torch.argmax(logits[:, -1, :], dim=-1)  # [1]
        tid = nxt.item()
        if tid == eos_id:
            break
        out_ids.append(tid)
        cur = nxt.view(1, 1)

    # Decode target pieces to string
    hyp = ro_sp.decode(out_ids)
    # Clean possible special tokens
    for tok in ["<s>", "</s>", "<pad>"]:
        hyp = hyp.replace(tok, "")
    return hyp.strip()


def translate_once(text_ur):
    """
    Translate a single Urdu text to Roman-Urdu using loaded model with greedy decoding
    
    Args:
        text_ur (str): Input Urdu text
        
    Returns:
        str: Roman-Urdu translation
    """
    try:
        # Load model if not already loaded
        model, src_tokenizer, tgt_tokenizer, device, bos_id, eos_id = load_model_and_tokenizers()
        
        if not text_ur or not text_ur.strip():
            return ""
        
        # Use greedy decoding
        result = greedy_decode(model, src_tokenizer, tgt_tokenizer, text_ur.strip(), bos_id, eos_id, max_len=180)
        
        print(f"Input: {text_ur}")
        print(f"Output: {result}")
        
        return result
        
    except Exception as e:
        print(f"Translation error: {e}")
        import traceback
        traceback.print_exc()
        return ""


def get_model_info():
    """
    Get information about the loaded model
    """
    try:
        model, src_tokenizer, tgt_tokenizer, device, bos_id, eos_id = load_model_and_tokenizers()
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            "model_architecture": "BiLSTM Encoder (2 layers) + LSTM Decoder (4 layers)",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "src_vocab_size": src_tokenizer.vocab_size(),
            "tgt_vocab_size": tgt_tokenizer.vocab_size(),
            "device": str(device),
            "framework": "PyTorch",
            "bos_id": bos_id,
            "eos_id": eos_id
        }
    except Exception as e:
        return {"error": str(e)}


# Test function for debugging
if __name__ == "__main__":
    # Test translation
    test_text = "یہ ایک ٹیسٹ ہے"
    print(f"Input: {test_text}")
    
    try:
        result = translate_once(test_text)
        print(f"Output: {result}")
        
        info = get_model_info()
        print(f"Model Info: {info}")
        
    except Exception as e:
        print(f"Error: {e}")