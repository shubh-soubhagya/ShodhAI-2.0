import os
import re
import io
import csv
import json
import time
import groq
import base64
import shutil
import logging
import requests
import tempfile
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from fpdf import FPDF
from docx import Document
from concurrent.futures import ThreadPoolExecutor
from groq import Groq
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from transcript_extractor.extract_transcript import get_transcript, get_transcript_with_timestamps
from transcriptQA.groqllm import ask_groq_yt
from python_scripts.ocr import extract_text
from python_scripts.spelling_corrections import correct_spelling
from python_scripts.spacings import add_space_after_punctuation
from python_scripts.groqllm import clean_text
from plag.cosine_similarity import cosine_similarity_count, cosine_similarity_tfidf
from plag.jaccard_similarity import jaccard_similarity
from plag.lcs import lcs
from plag.lsh import lsh_similarity
from plag.n_gram_similarity import n_gram_similarity



# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logging.getLogger("langchain").setLevel(logging.ERROR)
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize Flask app
app = Flask(__name__, template_folder="templates")
app.config['UPLOAD_FOLDER'] = 'uploads'
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

agent_executor = None
current_pdf_path = None
retriever_global = None

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_pdf(pdf_path):
    """Load the uploaded PDF and create the agent_executor."""
    global agent_executor

    try:
        logger.info(f"Processing PDF: {pdf_path}")

        # Step 1: Load PDF
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        logger.info(f"✅ Loaded {len(docs)} document chunks successfully.")

        # Step 2: Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        documents = text_splitter.split_documents(docs)
        logger.info(f"✅ Split into {len(documents)} chunks.")

        # Step 3: Load embeddings from local model
        embeddings_model = HuggingFaceEmbeddings(
            model_name=r"C:\Users\hp\Desktop\ps_sol\models\all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        # Step 4: Create FAISS vector DB
        vectordb = FAISS.from_documents(documents, embeddings_model)
        base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
        faiss_path = os.path.join('faiss_indexes', f"faiss_index_{base_filename}")
        os.makedirs('faiss_indexes', exist_ok=True)
        vectordb.save_local(faiss_path)
        logger.info(f"✅ Created and saved FAISS DB at {faiss_path}")

        retriever = vectordb.as_retriever()

        global retriever_global
        retriever_global = retriever

        # Step 5: Create retriever tool
        pdf_tool = create_retriever_tool(
            retriever,
            "pdf_search",
            "Search information inside the PDF"
        )
        tools = [pdf_tool]

        # Step 6: Load Ollama LLM (Gemma 2B)
        llm = ChatOllama(
            model="gemma2:2b",
            base_url="http://localhost:11434",
            temperature=0.7
        )

        # Step 7: Create prompt
        prompt = ChatPromptTemplate.from_template(
            """
            Answer the questions based on the provided PDF context only.
            Provide accurate and detailed responses strictly from the PDF content.
            <context>
            {context}
            <context>
            Questions: {input}
            {agent_scratchpad}
            """
        )

        # Step 8: Create agent and executor
        agent = create_openai_tools_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

        return True

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
        return False



@app.route('/')
def home():
    return render_template('index.html')





###################### OCR TOOL #######################

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
TEMP_FOLDER = os.path.join(BASE_DIR, 'temp_files')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file_image(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    """
    Process the image through OCR pipeline
    """
    try:
        # Extract text from image
        text = extract_text(image_path)
        
        # Apply corrections
        corrected_text = correct_spelling(text)
        
        # Add proper spacing
        spaced_text = add_space_after_punctuation(corrected_text)
        
        # Clean and format text
        final_text = clean_text(spaced_text)
        
        return final_text
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

def create_text_file(text, output_path):
    """Save text to a file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)

def create_pdf_file(text, output_path):
    """Convert text to PDF"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=text)
    pdf.output(output_path)

def create_docx_file(text, output_path):
    """Convert text to DOCX"""
    doc = Document()
    doc.add_paragraph(text)
    doc.save(output_path)

@app.route('/ocr', methods=['GET', 'POST'])
def ocr():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'image' not in request.files:
            return 'No file part', 400
        
        file = request.files['image']
        
        # If user doesn't select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return 'No selected file', 400
        
        if file and allowed_file_image(file.filename):
            # Secure the filename
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            try:
                # Save the file temporarily
                file.save(filepath)
                
                # Process the image
                extracted_text = process_image(filepath)
                
                # Clean up the uploaded file
                os.remove(filepath)
                
                if extracted_text is None:
                    return 'Error processing image', 500
                
                return extracted_text
                
            except Exception as e:
                # Clean up on error
                if os.path.exists(filepath):
                    os.remove(filepath)
                return f'Error: {str(e)}', 500
                
        return 'Invalid file type', 400
        
    return render_template('ocr.html')

@app.route('/ocr/download', methods=['POST'])
def download():
    text = request.form.get('text')
    format_type = request.form.get('format')
    
    if not text:
        return 'No text provided', 400
    
    if not format_type:
        return 'No format specified', 400
    
    filename = f'extracted_text.{format_type}'
    output_path = os.path.join(TEMP_FOLDER, filename)
    
    try:
        # Create file based on requested format
        if format_type == 'txt':
            create_text_file(text, output_path)
        elif format_type == 'pdf':
            create_pdf_file(text, output_path)
        elif format_type == 'docx':
            create_docx_file(text, output_path)
        else:
            return 'Invalid format type', 400
        
        # Send file to user
        return_data = send_file(
            output_path,
            as_attachment=True,
            download_name=filename
        )
        
        # Clean up after sending
        os.remove(output_path)
        
        return return_data
        
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        return f'Error creating file: {str(e)}', 500

@app.errorhandler(413)
def too_large(e):
    return 'File is too large', 413



########################## DOCSIMILARITY ##################################

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

# Preprocessing Function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Extract text from PDF
def read_pdf_text(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ''
        for page in reader.pages:
            text += page.extract_text() or ''
        return preprocess_text(text)
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return ""

# Similarity functions
similarity_functions = {
    "Cosine_TFIDF": cosine_similarity_tfidf,
    "Cosine_Count": cosine_similarity_count,
    "Jaccard": jaccard_similarity,
    "LCS": lcs,
    "LSH": lsh_similarity,
    "NGram": n_gram_similarity
}

# Compute similarity for a pair
def compare_pair(i, j, file_names, texts):
    row = {
        "Doc 1": file_names[i],
        "Doc 2": file_names[j]
    }
    scores = []
    for name, func in similarity_functions.items():
        try:
            score = round(func(texts[i], texts[j]) * 100, 2)
        except Exception as e:
            print(f"Error computing {name} for {file_names[i]} and {file_names[j]}: {e}")
            score = 0.0
        row[name] = score
        scores.append(score)
    row["Average Similarity (%)"] = round(np.mean(scores), 2)
    return row

# Save uploaded files and process them
def process_uploaded_pdfs(files):
    # Create a temporary directory to store the uploaded PDFs
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    
    try:
        # Save the files to the temporary directory
        for file in files:
            if file.filename.lower().endswith('.pdf'):
                file_path = os.path.join(temp_dir, secure_filename(file.filename))
                file.save(file_path)
                file_paths.append(file_path)
        
        # Process the saved files
        results = compare_pdfs(file_paths)
        return results
    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)

# Main comparison function
def compare_pdfs(pdf_files):
    file_names = [os.path.basename(p) for p in pdf_files]

    # Load all PDFs concurrently
    with ThreadPoolExecutor(max_workers=10) as executor:
        texts = list(executor.map(read_pdf_text, pdf_files))

    results = []

    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(len(pdf_files)):
            for j in range(i + 1, len(pdf_files)):
                futures.append(executor.submit(compare_pair, i, j, file_names, texts))
        for future in futures:
            results.append(future.result())

    return results

# Route for the main page
@app.route('/docsim')
def docsim():
    return render_template('docsim.html')

# API endpoint for processing PDFs
@app.route('/docsim/api/process', methods=['POST'])
def process_pdfs():
    if 'files[]' not in request.files:
        return jsonify({"error": "No files provided"}), 400
    
    files = request.files.getlist('files[]')
    if not files or len(files) < 2:
        return jsonify({"error": "Please upload at least 2 PDF files"}), 400
    
    # Filter for PDF files only
    pdf_files = [f for f in files if f.filename.lower().endswith('.pdf')]
    if len(pdf_files) < 2:
        return jsonify({"error": "Please upload at least 2 PDF files"}), 400
    
    try:
        results = process_uploaded_pdfs(pdf_files)
        return jsonify({
            "results": results, 
            "fileCount": len(pdf_files),
            "fileNames": [f.filename for f in pdf_files]
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API endpoint for downloading results as CSV
@app.route('/docsim/api/download/csv', methods=['POST'])
def download_csv():
    try:
        data = request.json
        results = data.get('results', [])
        
        if not results:
            return jsonify({"error": "No results to download"}), 400
        
        # Create a CSV in memory
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
        # Create a temporary file to save the CSV
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        with open(temp_file.name, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        
        return send_file(temp_file.name, 
                         mimetype='text/csv',
                         as_attachment=True, 
                         download_name=f'similarity_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Generate HTML report content
def generate_html_report(results, file_names):    
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    table_rows = ""
    for result in results:
        table_rows += f"""
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">{result["Doc 1"]}</td>
            <td style="border: 1px solid #ddd; padding: 8px;">{result["Doc 2"]}</td>
            <td style="border: 1px solid #ddd; padding: 8px;">{result["Cosine_TFIDF"]}%</td>
            <td style="border: 1px solid #ddd; padding: 8px;">{result["Cosine_Count"]}%</td>
            <td style="border: 1px solid #ddd; padding: 8px;">{result["Jaccard"]}%</td>
            <td style="border: 1px solid #ddd; padding: 8px;">{result["LCS"]}%</td>
            <td style="border: 1px solid #ddd; padding: 8px;">{result["LSH"]}%</td>
            <td style="border: 1px solid #ddd; padding: 8px;">{result["NGram"]}%</td>
            <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">{result["Average Similarity (%)"]}%</td>
        </tr>
        """
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Document Similarity Report - {report_date}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #2563eb; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th {{ background-color: #f1f5f9; text-align: left; padding: 12px; border: 1px solid #ddd; }}
            td {{ padding: 8px; border: 1px solid #ddd; }}
            tr:nth-child(even) {{ background-color: #f9fafb; }}
            .summary {{ margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>Document Similarity Analysis Report</h1>
        <div class="summary">
            <p><strong>Generated on:</strong> {report_date}</p>
            <p><strong>Files analyzed:</strong> {len(file_names)}</p>
            <p><strong>File names:</strong> {', '.join(file_names)}</p>
        </div>
        
        <h2>Similarity Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Doc 1</th>
                    <th>Doc 2</th>
                    <th>Cosine_TFIDF (%)</th>
                    <th>Cosine_Count (%)</th>
                    <th>Jaccard (%)</th>
                    <th>LCS (%)</th>
                    <th>LSH (%)</th>
                    <th>NGram (%)</th>
                    <th>Average (%)</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        
        <h2>Metrics Explanation</h2>
        <table>
            <thead>
                <tr>
                    <th style="width: 25%;">Metric</th>
                    <th style="width: 75%;">Explanation</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Cosine TF-IDF</strong></td>
                    <td>Measures similarity by treating documents as vectors, with words weighted by their importance. Higher scores indicate documents share important terms, not just common words.</td>
                </tr>
                <tr>
                    <td><strong>Cosine Count</strong></td>
                    <td>Similar to Cosine TF-IDF but uses raw word counts instead of weighted values. Measures how similar the word distributions are between documents.</td>
                </tr>
                <tr>
                    <td><strong>Jaccard</strong></td>
                    <td>Compares the shared words between documents to the total unique words in both. Focuses on word overlap regardless of frequency or order.</td>
                </tr>
                <tr>
                    <td><strong>LCS</strong></td>
                    <td>Finds the longest sequence of words that appear in the same order in both documents. Good for detecting large blocks of identical text.</td>
                </tr>
                <tr>
                    <td><strong>LSH</strong></td>
                    <td>Uses hashing to quickly identify similar document segments. Effective for detecting partial matches and document sections that have been copied.</td>
                </tr>
                <tr>
                    <td><strong>NGram</strong></td>
                    <td>Compares sequences of consecutive words (typically 3-5 words) between documents. Good for identifying phrase-level similarity and paraphrasing.</td>
                </tr>
                <tr>
                    <td><strong>Average</strong></td>
                    <td>The mean of all similarity metrics, giving an overall indication of document similarity. Higher percentages suggest a greater likelihood of content overlap.</td>
                </tr>
            </tbody>
        </table>
        
        <div style="margin-top: 30px; color: #6b7280; font-size: 12px; text-align: center;">
            <p>Generated by DocSimilarity - PDF Document Similarity Analysis Tool</p>
        </div>
    </body>
    </html>
    """
    
    return html_content

# API endpoint for downloading results as HTML report
@app.route('/docsim/api/download/html', methods=['POST'])
def download_html():
    try:
        data = request.json
        results = data.get('results', [])
        file_names = data.get('fileNames', [])
        
        if not results:
            return jsonify({"error": "No results to download"}), 400
        
        # Generate HTML report
        html_content = generate_html_report(results, file_names)
        
        # Create a temporary file to save the HTML report
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
        with open(temp_file.name, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return send_file(temp_file.name, 
                         mimetype='text/html',
                         as_attachment=True, 
                         download_name=f'similarity_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html')
    except Exception as e:
        return jsonify({"error": str(e)}), 500



################ YOUTUBE CHATBOT ######################

client = Groq(api_key=groq_api_key)

# Define temp directory for transcript files
base_path = os.path.abspath(os.path.join(os.getcwd(), "temp_files"))
transcript_file = os.path.join(base_path, "transcript.txt")
timestamp_file = os.path.join(base_path, "transcripts_with_timestamps.txt")

print(f"📁 Base path: {base_path}")
print("Transcript File Exists:", os.path.exists(transcript_file))
print("Timestamp File Exists:", os.path.exists(timestamp_file))

def load_transcript():
    """Load existing transcript if available."""
    print(f"🔍 Checking for transcript file at: {transcript_file}")
    print(f"🔍 Checking for timestamp file at: {timestamp_file}")

    try:
        if os.path.exists(timestamp_file):
            with open(timestamp_file, "r", encoding="utf-8") as file:
                content = file.read()
                print(f"📂 Loaded timestamp transcript: {content[:200] if content else 'Empty file'}")
                if content:
                    return content
        
        if os.path.exists(transcript_file):
            with open(transcript_file, "r", encoding="utf-8") as file:
                content = file.read()
                print(f"📂 Loaded transcript: {content[:200] if content else 'Empty file'}")
                if content:
                    return content
        
        print("⚠️ No transcript file found or files are empty!")
        return None
    except Exception as e:
        print(f"⚠️ Error loading transcript: {e}")
        return None

def save_transcript_to_file(content, file_path):
    """Save transcript text to a file and confirm success."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)
        print(f"✅ Transcript saved successfully: {file_path}")
        return True
    except Exception as e:
        print(f"❌ Error saving transcript: {e}")
        return False

@app.route('/ytbot')
def ytbot():
    return render_template('ytbot.html')

@app.route('/ytbot/api/extract-transcript', methods=['POST'])
def extract_transcript():
    data = request.json
    youtube_url = data.get('url')

    if not youtube_url:
        return jsonify({'success': False, 'error': 'No YouTube URL provided'})

    # Extract new transcript
    try:
        result = get_transcript(youtube_url)
        if not result['success']:
            return jsonify({'success': False, 'error': result['error']})

        # Save base transcript
        save_success = save_transcript_to_file(result['transcript'], transcript_file)
        if not save_success:
            return jsonify({'success': False, 'error': 'Failed to save transcript'})

        # Try extracting transcript with timestamps
        timestamp_result = get_transcript_with_timestamps(youtube_url)
        if timestamp_result['success']:
            save_transcript_to_file(timestamp_result['transcript'], timestamp_file)
            transcript_text = timestamp_result['transcript']
        else:
            transcript_text = result['transcript']

        return jsonify({'success': True, 'transcript': transcript_text})

    except Exception as e:
        return jsonify({'success': False, 'error': f"An error occurred: {str(e)}"})

@app.route('/ytbot/api/ask-question', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question')

    if not question:
        return jsonify({'success': False, 'error': 'No question provided'})

    # Load transcript dynamically from file OR from request
    transcript_text = data.get('transcript')
    
    # If transcript wasn't sent in request, try to load it from file
    if not transcript_text:
        transcript_text = load_transcript()
    
    if not transcript_text:
        return jsonify({'success': False, 'error': 'No transcript found. Please extract the transcript first.'})

    try:
        answer = ask_groq_yt(question, client=client, transcript_text=transcript_text)
        return jsonify({'success': True, 'answer': answer})
    except Exception as e:
        return jsonify({'success': False, 'error': f"An error occurred: {str(e)}"})

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static/css', path)



########################### WEBSITE CHATBOT ############################

CONTENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'content')

# Use proper path separators for files
HISTORY_FILE = os.path.join(CONTENT_DIR, 'history.json')
CONTENT_FILE = os.path.join(CONTENT_DIR, 'website_content.txt')

# Initialize Groq client if key is available
client = None
if groq_api_key:
    client = groq.Groq(api_key=groq_api_key)

# Store chat history
history = {}

def extract_content(url):
    """Extract content from a website URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
            
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Remove blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Save the extracted content to a file
        os.makedirs(CONTENT_DIR, exist_ok=True)
        with open(CONTENT_FILE, 'w', encoding='utf-8') as f:
            f.write(text)
            
        return {"url": url, "content": text[:500] + "..."}  # Return truncated content
    except Exception as e:
        raise Exception(f"Failed to extract content: {str(e)}")

def ask_groq(question, content, client):
    """Ask a question based on the content"""
    if not client:
        return "API key not configured. Please set up your GROQ_API_KEY in the .env file."
    
    try:
        # Prepare context and question for the model
        prompt = f"""
        Based on the following website content:
        
        {content}
        
        Please answer the following question:
        {question}
        
        If the answer is not found in the content, please say "I couldn't find information about that in the website content."
        """
        
        # Make API call to Groq
        response = client.chat.completions.create(
            model="llama3-70b-8192",  # You can change this to a model of your choice
            messages=[
                {"role": "system", "content": ("You are a helpful assistant that answers questions based on website content." "your task is to answer questions based on its content accurately." "you can answer outside the content also but related to content topic")},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/webbot')
def webbot():
    print("Webbot route accessed!")
    return render_template('webbot.html')

@app.route('/webbot/extract', methods=['POST'])
def extract():
    data = request.json
    url = data.get('url')
    
    if not url:
        return jsonify({"error": "URL is required"}), 400
    
    try:
        # Extract content from the URL
        url_info = extract_content(url)
        
        # Initialize chat history for this URL
        if url not in history:
            history[url] = []
            
        return jsonify({"success": True, "message": "Content extracted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/webbot/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question')
    url = data.get('url')
    
    if not question or not url:
        return jsonify({"error": "Question and URL are required"}), 400
    
    try:
        # Read content from file
        with open(CONTENT_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Get response from Groq
        response = ask_groq(question, content, client)
        
        # Add to history
        if url not in history:
            history[url] = []
        
        history[url].append({"question": question, "answer": response})
        
        # Save history to JSON file
        os.makedirs(CONTENT_DIR, exist_ok=True)
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f)
        
        return jsonify({"success": True, "answer": response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/webbot/history')
def get_history():
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
                return jsonify({"success": True, "history": history_data}), 200
        else:
            return jsonify({"success": True, "history": {}}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/webbot/content')
def get_content():
    try:
        if os.path.exists(CONTENT_FILE):
            with open(CONTENT_FILE, 'r', encoding='utf-8') as f:
                content = f.read()
            return jsonify({"success": True, "content": content}), 200
        else:
            return jsonify({"error": "Content file not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/webbot/download')
def download_content():
    try:
        if os.path.exists(CONTENT_FILE):
            return send_file(CONTENT_FILE, 
                             as_attachment=True, 
                             download_name='website_content.txt')
        else:
            return jsonify({"error": "Content file not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500



############################################# RAG BASED CHATBOT ##########################################
@app.route('/rag')
def rag():
    return render_template('rag.html')


@app.route('/rag/upload', methods=['POST'])
def upload_file():
    global current_pdf_path

    if 'pdf_file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded.'})

    file = request.files['pdf_file']

    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected.'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        current_pdf_path = filepath

        if process_pdf(filepath):
            return jsonify({'status': 'success', 'message': 'PDF uploaded and processed successfully.'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to process PDF.'})

    return jsonify({'status': 'error', 'message': 'Invalid file format. Only PDFs are allowed.'})

@app.route('/rag/ask', methods=['POST'])
def ask_question_rag():
    global agent_executor
    global retriever_global

    if not agent_executor:
        return jsonify({'status': 'error', 'message': 'No PDF processed. Please upload a PDF first.'})

    if not retriever_global:
        return jsonify({'status': 'error', 'message': 'Retriever not ready.'})

    data = request.json
    query = data.get('query', '').strip()

    if not query:
        return jsonify({'status': 'error', 'message': 'Query is empty.'})

    try:
        start_time = time.time()

        # ✅ Use the global retriever
        docs = retriever_global.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        response = agent_executor.invoke({
            "input": query,
            "context": context,
            "agent_scratchpad": ""
        })

        answer = response.get('output', 'No output generated.')
        response_time = time.time() - start_time

        return jsonify({
            'status': 'success',
            'answer': answer,
            'response_time': f"{response_time:.2f}"
        })

    except Exception as e:
        logger.error(f"Error answering question: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'Backend Error: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)