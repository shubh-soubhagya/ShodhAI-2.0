# ShodhAI - AI-Powered Research Assistant

ShodhAI is a comprehensive AI-powered research and document analysis platform that helps researchers, students, and professionals process, analyze, and extract insights from various types of documents and content.

## 📸 Main Interface
![ShodhAI Main Interface](screenshots/main_interface.png)
*ShodhAI's comprehensive dashboard with all tools and features*

## 📸 Adding Screenshots
To add screenshots to the project:

1. Create a `screenshots` directory in the project root:
   ```bash
   mkdir screenshots
   ```

2. Add your screenshots to the `screenshots` directory:
   - `main_interface.png` - Main dashboard screenshot
   - `pdf_processing.png` - PDF processing interface
   - `ocr_tool.png` - OCR tool interface
   - `doc_similarity.png` - Document similarity interface
   - `youtube_bot.png` - YouTube transcript bot interface
   - `web_bot.png` - Web content bot interface

3. Ensure screenshots are:
   - High quality (recommended resolution: 1920x1080)
   - In PNG format
   - Properly cropped to show relevant features
   - Less than 1MB in size

## 🌟 Features

### 1. PDF Processing & RAG
- Upload and process PDF documents
- Extract text and create embeddings
- Ask questions about PDF content
- Get AI-powered responses based on document context

### 2. OCR (Optical Character Recognition)
- Convert images to text
- Support for PNG, JPG, JPEG formats
- Automatic text correction and formatting
- Export to TXT, PDF, or DOCX formats

### 3. Document Similarity Analysis
- Compare multiple PDF documents
- Generate similarity scores
- Download detailed comparison reports
- Visualize document relationships

### 4. YouTube Transcript Bot
- Extract transcripts from YouTube videos
- Ask questions about video content
- Save and process transcripts
- Get AI-powered responses

### 5. Web Content Bot
- Extract content from web pages
- Ask questions about web content
- Maintain interaction history
- Download processed content

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Groq API key
- Ollama installed locally
- Git

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/ShodhAI-2.0.git
   cd ShodhAI-2.0
   ```

2. **Create Environment File**
   - Create a `.env` file in the root directory
   - Add your Groq API key:
     ```
     GROQ_API_KEY=your_groq_api_key_here
     ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Ollama**
   - Download and install Ollama from [ollama.ai](https://ollama.ai)
   - Pull the required model:
     ```bash
     ollama pull gemma2:2b
     ```

5. **Run the Application**
   ```bash
   python app.py
   ```
   The application will be available at `http://localhost:5000`

## 📁 Project Structure
```
ShodhAI-2.0/
├── app.py                 # Main application file
├── requirements.txt       # Project dependencies
├── templates/            # Web templates
├── static/              # Static assets
├── uploads/             # File uploads
├── temp_files/          # Temporary files
├── transcript_extractor/ # Transcript processing
├── transcriptQA/        # Transcript Q&A
├── python_scripts/      # Utility scripts
├── plag/               # Plagiarism detection
└── content/            # Content storage
```

## 🛠️ Technical Stack
- **Backend**: Flask (Python)
- **AI/ML**: 
  - LangChain
  - Groq
  - HuggingFace
  - FAISS
  - Sentence Transformers
- **Document Processing**:
  - PyPDF2
  - python-docx
  - FPDF
  - pytesseract (OCR)
- **Other Dependencies**:
  - OpenCV
  - PaddlePaddle
  - scikit-learn
  - pandas
  - numpy

## 📸 Demo Screenshots

### PDF Processing
![PDF Processing](screenshots/pdf_processing.png)
*Upload and process PDF documents with AI-powered Q&A*

### OCR Tool
![OCR Tool](screenshots/ocr_tool.png)
*Convert images to text with automatic correction*

### Document Similarity
![Document Similarity](screenshots/doc_similarity.png)
*Compare and analyze document similarities*

### YouTube Transcript Bot
![YouTube Bot](screenshots/youtube_bot.png)
*Extract and analyze YouTube video transcripts*

### Web Content Bot
![Web Bot](screenshots/web_bot.png)
*Process and analyze web content*

## 🔒 Security Features
- File size limits (16MB max)
- Secure filename handling
- Allowed file type restrictions
- Environment variable management

## ⚠️ Common Issues & Solutions

1. **Ollama Connection Error**
   - Ensure Ollama is running locally
   - Check if the model is properly installed
   - Verify the base URL in the code

2. **File Upload Issues**
   - Check file size limits
   - Verify file extensions
   - Ensure proper permissions

3. **API Key Issues**
   - Verify GROQ_API_KEY in .env file
   - Check API key validity
   - Ensure proper environment setup

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments
- LangChain for the AI framework
- Groq for the LLM API
- HuggingFace for the models
- All other open-source contributors

## 📞 Support
For support, please open an issue in the GitHub repository or contact the maintainers.

---
Made with ❤️ by the ShodhAI Team 