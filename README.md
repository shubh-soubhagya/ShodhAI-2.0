# ShodhAI - AI-Powered Research Assistant

ShodhAI is a comprehensive AI-powered research and document analysis platform that helps researchers, students, and professionals process, analyze, and extract insights from various types of documents and content.

---

## 🌟 Features

1. **YouTube Transcript Analyzer & QnA Bot**: Turn any YouTube video into an interactive learning experience. Extract transcripts and ask questions about video content through our AI assistant.

2. **Website URL Analyzer**: Analyze content from any website URL using our AI engine. Summarize, extract insights, detect sentiment, and interact with page content through intelligent QnA—all in one click.

3. **RAG-Based Chatbot for PDFs**: Upload any PDF and instantly generate an AI chatbot that understands your document. Ask questions and get accurate, context-aware answers in seconds.

4. **Document Similarity Check**: An advanced tool that detects content overlap, paraphrasing, and plagiarism across academic or professional documents. Upload files and receive detailed similarity reports in seconds with precision accuracy.

5. **High-Precision OCR Text Extractor**: Extract text from images with exceptional accuracy. Copy the extracted content or download it instantly in your preferred format for easy use.

Each feature is designed to be fast, accurate, and easy to use—turning digital resources into dynamic, interactive tools.

---

## ⚙️ Prerequisites
- Python 3.8 or higher  
- Groq API Key 
- Ollama installed locally  
- `all-MiniLM-L6-v2` model saved at `ShodhAI-2.0/models`  
- Git

## 🛠️ Installation Steps

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/ShodhAI-2.0.git
cd ShodhAI-2.0
```

2. **Create Environment File**
   Create a `.env` file in the root directory and add your Groq API key:
```
GROQ_API_KEY = "your_groq_api_key_here"
```
   You can generate your Free API Key from [Groq Console](https://console.groq.com/keys)

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Install Embedding Model**
   Create the directory `ShodhAI-2.0/models` and run the script to download and save the model:
``` bash
from sentence_transformers import SentenceTransformer

# Define your custom path
custom_path = "ShodhAI-2.0/models/all-MiniLM-L6-v2"

# Load and save the model
model = SentenceTransformer('all-MiniLM-L6-v2')
model.save(custom_path)

print(f"Model saved to: {custom_path}")
```
   
5. **Install Ollama**
   Download and install Ollama from [ollama.ai](https://ollama.ai) and pull the required model:
```bash
ollama pull gemma2:2b
```

6. **Run the Application**
```bash
python app.py
```
The application will be available at `http://localhost:5000`

---

## 📁 Project Structure
```
ShodhAI-2.0/
├── app.py                 # Main application file
├── .env                   # Groq API key env 
├── requirements.txt       # Project dependencies
├── templates/             # Web templates
├── static/                # Static assets
├── models/                # Embedding model
├── uploads/               # File uploads
├── temp_files/            # Temporary files
├── transcript_extractor/  # Transcript processing
├── transcriptQA/          # Transcript Q&A
├── python_scripts/        # Utility scripts
├── plag/                  # Plagiarism detection
└── content/               # Content storage
```
---

## ⚠️ Common Issues & Solutions

1. **Ollama Connection Error**
- Ensure Ollama is running locally
- Check if the model is properly pulled (`gemma2:2b`)
- Verify the base URL in the code

2. **File Upload Issues**
- Check file size limits
- Verify file extensions
- Ensure proper permissions

3. **API Key Issues**
- Make sure the key is correctly set in `.env`
- Confirm your key is active at [Groq Console](https://console.groq.com/keys)
- Ensure proper environment setup

4. **Embedding Model Errors**
- Confirm the directory `models/all-MiniLM-L6-v2` exists
- Ensure correct path in code (no trailing slashes)
- If model fails to load, re-run the embedding model installation script

---

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments
- LangChain for the AI framework
- Groq for the LLM API
- HuggingFace for the Embedding Model
- All other open-source contributors

## 📞 Support
For support, please open an issue in the GitHub repository or contact the maintainers.

---
Made with ❤️ by the ShodhAI Team 
