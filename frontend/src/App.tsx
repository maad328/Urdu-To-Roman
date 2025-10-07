import { useState, useEffect } from 'react'
import axios from 'axios'
import './App.css'

// Types


interface HealthResponse {
  status: string
  model_loaded: boolean
  device?: string
  error?: string
}

const API_BASE_URL = 'http://127.0.0.1:8000'

// Sample texts for quick testing
const SAMPLE_TEXTS = [
  "کس کی نمود کے لیے شام و سحر ہیں گرم",
  "دل سے نکلے ہیں جو لفظ اثر رکھتے ہیں",
  "محبت کرنے والے کم نہیں",
  "یہ ایک خوبصورت دن ہے",
  "زندگی ایک سفر ہے"
]

function App() {
  const [inputText, setInputText] = useState('')
  const [outputText, setOutputText] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const [apiHealth, setApiHealth] = useState<HealthResponse | null>(null)

  // Check API health on component mount
  useEffect(() => {
    checkApiHealth()
    
  }, [])

  const checkApiHealth = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/health`)
      setApiHealth(response.data)
    } catch (err) {
      setApiHealth({
        status: 'error',
        model_loaded: false,
        error: 'Cannot connect to API'
      })
    }
  }

  

  const handleTranslate = async () => {
    if (!inputText.trim()) {
      setError('Please enter some Urdu text to translate')
      return
    }

    setIsLoading(true)
    setError('')
    setOutputText('')

    try {
      const response = await axios.post(`${API_BASE_URL}/translate`, {
        text_ur: inputText.trim()
      })

      setOutputText(response.data.output_text)
    } catch (err: any) {
      if (err.response?.data?.detail) {
        setError(err.response.data.detail)
      } else {
        setError('Translation failed. Please check if the API server is running.')
      }
    } finally {
      setIsLoading(false)
    }
  }

  const handleCopyOutput = async () => {
    if (outputText) {
      try {
        await navigator.clipboard.writeText(outputText)
        // You could add a toast notification here
      } catch (err) {
        console.error('Failed to copy text:', err)
      }
    }
  }

  const clearAll = () => {
    setInputText('')
    setOutputText('')
    setError('')
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      handleTranslate()
    }
  }

  const loadSampleText = (text: string) => {
    setInputText(text)
    setError('')
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>🔤 Urdu → Roman Urdu</h1>
        <p>Neural Machine Translation using BiLSTM Encoder + LSTM Decoder</p>
        
        {/* API Status */}
        <div className="api-status">
          <span className={`status-dot ${apiHealth?.status || 'unknown'}`}></span>
          <span>
            API: {apiHealth?.status === 'healthy' ? 'Connected' : 
                   apiHealth?.status === 'unhealthy' ? 'Unhealthy' : 'Disconnected'}
            {apiHealth?.device && ` (${apiHealth.device})`}
          </span>
        </div>
      </header>

      <main className="main-content">
        {/* Input Section */}
        <div className="input-section">
          <h2>📝 Input Text</h2>
          <div className="input-container">
            <label htmlFor="urdu-input">Enter Urdu Text:</label>
            <textarea
              id="urdu-input"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder="یہاں اردو متن لکھیں... (Type your Urdu text here)"
              rows={5}
              className="urdu-textarea"
              dir="rtl"
              lang="ur"
            />
            <div className="input-info">
              <span>Characters: {inputText.length}</span>
              <span>Click Translate or press Ctrl+Enter</span>
            </div>
          </div>
          
          {/* Sample Texts */}
          <div className="sample-texts">
            <label>Quick Test Samples:</label>
            <div className="sample-buttons">
              {SAMPLE_TEXTS.map((text, index) => (
                <button
                  key={index}
                  onClick={() => loadSampleText(text)}
                  className="sample-btn"
                  title={`Load: ${text}`}
                >
                  {text.length > 30 ? text.substring(0, 30) + '...' : text}
                </button>
              ))}
            </div>
          </div>
          
          <div className="button-group">
            <button
              onClick={handleTranslate}
              disabled={isLoading || !inputText.trim() || apiHealth?.status !== 'healthy'}
              className="translate-btn"
            >
              {isLoading ? (
                <>
                  <div className="loading-spinner"></div>
                  Translating...
                </>
              ) : (
                <>
                  🚀 Translate
                </>
              )}
            </button>
            <button
              onClick={clearAll}
              disabled={!inputText && !outputText}
              className="clear-btn"
            >
              🗑️ Clear
            </button>
          </div>
        </div>

        {/* Error Banner */}
        {error && (
          <div className="error-banner">
            <span className="error-icon">⚠️</span>
            <span>{error}</span>
            <button onClick={() => setError('')} className="error-close">×</button>
          </div>
        )}

        {/* Output Section */}
        {outputText && (
          <div className="output-section">
            <h2>📊 Translation Result</h2>
            <div className="output-container">
              <div className="output-header">
                <label>Roman-Urdu Output:</label>
                <button
                  onClick={handleCopyOutput}
                  className="copy-btn"
                  title="Copy to clipboard"
                >
                  📋 Copy
                </button>
              </div>
              <div className="output-text">
                {outputText}
              </div>
            </div>
          </div>
        )}

       
      </main>

      <footer className="app-footer">
        <p>
          Built with React + TypeScript + FastAPI + PyTorch
          <br />
          Assignment: Neural Machine Translation (BiLSTM Encoder + LSTM Decoder)
        </p>
      </footer>
    </div>
  )
}

export default App