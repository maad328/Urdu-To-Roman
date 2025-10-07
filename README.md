### Prerequisites

- Python 3.8+
- Node.js 16+
- CUDA-capable GPU (optional, for faster training)

### 1. Clone Repository

```bash
git clone  https://github.com/maad328/Urdu-To-Roman.git
cd urdu_ghazals_rekhta
```

### 2. Backend Setup

```bash
cd backend
pip install -r requirements.txt
```

### 3. Frontend Setup

```bash
cd frontend
npm install
```

### 4. Run the Application

**Start Backend API:**

```bash
cd backend
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

**Start Frontend (in new terminal):**

```bash
cd frontend
npm run dev
```

**Access the Application:**

- Frontend: http://localhost:5173
- API Documentation: http://localhost:8000/docs
- API Health Check: http://localhost:8000/health

## API Endpoints

### Translation

```http
POST /translate
Content-Type: application/json

{
  "text_ur": "کس کی نمود کے لیے شام و سحر ہیں گرم"
}
```

**Response:**

```json
{
  "output_text": "kis ki namood ke liye shaam o sahar hain garam"
}
```

**Response:**

```json
{
  "model_architecture": "BiLSTM Encoder (2 layers) + LSTM Decoder (4 layers)",
  "total_parameters": 39900000,
  "trainable_parameters": 39900000,
  "src_vocab_size": 51,
  "tgt_vocab_size": 32,
  "device": "cuda:0",
  "framework": "PyTorch"
}
```

## Project-Structure

urdu_ghazals_rekhta/
├── backend/ # PyTorch FastAPI backend
│ ├── api/ # FastAPI application
│ │ └── app.py # Main API server with endpoints
│ ├── models/ # Model definitions & checkpoints
│ │ └── best_bilstm_seq2seq3 (7).pth # Trained model weights
│ ├── tokenizers/ # SentencePiece tokenizers
│ │ ├── urdu_char.model # Urdu character-level tokenizer
│ │ └── roman_char.model # Roman character-level tokenizer
│ ├── tests/ # Testing & inference modules
│ │ └── test_infer_bilstm_seq2seq.py # Model inference testing
│ └── requirements.txt # Python dependencies
├── frontend/ # React TypeScript frontend
│ ├── src/ # Source code
│ │ ├── App.tsx # Main React component
│ │ ├── App.css # Application styling
│ │ ├── main.tsx # Application entry point
│ │ └── index.css # Global styles
│ ├── public/ # Static assets
│ │ └── vite.svg # Vite logo
│ ├── package.json # NPM dependencies and scripts
│ ├── vite.config.ts # Vite build configuration
│ └── tsconfig.json # TypeScript configuration
├── notebooks/ # Jupyter notebooks
│ └── project1_updated.ipynb # Training and experimentation notebook
├── requirements.txt # Root project dependencies
├── README.md # Project documentation
└── .gitignore # Git ignore rules

### Health Check

```http
GET /health
```

## Model Training

The model was trained using the following configuration:

### Training Parameters

- **Learning Rate**: 5e-4 (optimized)
- **Batch Size**: 16
- **Epochs**: 15
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Evaluation Metrics**: BLEU, CER, Token Accuracy

### Training Process

```python
# Key training configuration from your code
learning_rate = 5e-4
batch_size = 16
num_epochs = 15
embed_dim = 256
hidden_dim = 512
enc_layers = 2
dec_layers = 4
dropout = 0.3
```

### Dataset

- **Source**: Curated collection of Urdu ghazals from 30+ renowned poets
- **Languages**: Urdu (source) → Roman-Urdu (target)
- **Size**: Thousands of verse pairs
- **Poets Included**: Mirza Ghalib, Allama Iqbal, Faiz Ahmad Faiz, and more

## Performance Metrics

The model achieves the following performance:

- **BLEU Score**: Optimized for Urdu-to-Roman translation
- **Character Error Rate (CER)**: Minimized through character-level tokenization
- **Token Accuracy**: High accuracy on validation set
- **Inference Speed**: Sub-second translation times

## Development

### Running Tests

```bash
cd backend
python tests/test_infer_bilstm_seq2seq.py
```

### Building Frontend

```bash
cd frontend
npm run build
```

## Technical Details

### Tokenization Strategy

- **Character-level SentencePiece**: Reduces vocabulary size while maintaining linguistic accuracy
- **Special Tokens**: `<s>`, `</s>`, `<pad>` for proper sequence handling
- **Bidirectional Processing**: Captures context from both directions

### Model Architecture Details

The system follows a sequence-to-sequence architecture where Urdu text flows through character-level tokenization, then into a BiLSTM encoder that captures bidirectional context, followed by an LSTM decoder that generates Roman-Urdu output.

**Model Specifications:**

- **Encoder**: 2-layer BiLSTM with 512 hidden units that processes input in both forward and backward directions
- **Decoder**: 4-layer LSTM with 1024 hidden units (double the encoder size) for robust generation
- **Embedding Dimension**: 256-dimensional vector representations for each character
- **Dropout**: 0.3 regularization to prevent overfitting
- **Total Parameters**: Approximately 39.9 million trainable parameters
- **Vocabulary Sizes**: Urdu (51 characters), Roman-Urdu (32 characters)

### Greedy Decoding

The system uses greedy decoding for fast, deterministic translation:

```python


## 📚 Dataset Information

The dataset contains Urdu ghazals from renowned poets including:

- **Mirza Ghalib** - Master of Urdu poetry
- **Allama Iqbal** - National poet of Pakistan
- **Faiz Ahmad Faiz** - Progressive poet
- **Ahmad Faraz** - Modern Urdu poet
- **Parveen Shakir** - Contemporary female poet
- **And 25+ more poets**

Each poet's collection includes:
- Original Urdu text
- Hindi transliteration
- English transliteration (Roman-Urdu)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Rekhta Foundation** for providing access to Urdu poetry collections
- **PyTorch Team** for the excellent deep learning framework
- **FastAPI Team** for the modern Python web framework
- **React Team** for the powerful frontend library
- **SentencePiece** for efficient tokenization

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Contact: [maad29996@gmail.com]

---

**Built with ❤️ for preserving and promoting Urdu literature through technology**
```
