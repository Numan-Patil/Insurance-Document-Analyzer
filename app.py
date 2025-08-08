import os
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
import json

from document_processor import DocumentProcessor
from vector_store import VectorStore
from llm_client import LLMClient
from translation_service import TranslationService

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('vector_db', exist_ok=True)

# Configure the database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///insurance_docs.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Initialize the app with the extension
db.init_app(app)

# Initialize processors
document_processor = DocumentProcessor()
vector_store = VectorStore()
llm_client = LLMClient()
translation_service = TranslationService()

# Clear any existing vector store on startup for fresh sessions
vector_store.clear_all_documents()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files selected'}), 400

        # Clear previous uploads for this session
        import glob
        for old_file in glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*')):
            if os.path.isfile(old_file) and not old_file.endswith('.gitkeep'):
                os.remove(old_file)

        # Clear vector store for new session/upload
        vector_store.clear_all_documents()
        logger.info("Cleared vector store for new document upload session")

        files = request.files.getlist('files')
        uploaded_files = []

        for file in files:
            if file.filename == '':
                continue

            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Process the document
                logger.info(f"Processing document: {filename}")
                chunks = document_processor.process_pdf(filepath, filename)

                # Add to vector store
                vector_store.add_documents(chunks)

                uploaded_files.append(filename)
                logger.info(f"Successfully processed: {filename}")

        if not uploaded_files:
            return jsonify({'error': 'No valid PDF files uploaded'}), 400

        return jsonify({
            'message': f'Successfully uploaded and processed {len(uploaded_files)} files',
            'files': uploaded_files
        })

    except Exception as e:
        logger.error(f"Error uploading files: {str(e)}")
        return jsonify({'error': f'Error processing files: {str(e)}'}), 500

@app.route('/query', methods=['POST'])
def process_query():
    try:
        data = request.get_json()
        logger.info(f"Received request data: {data}")

        if not data or 'query' not in data:
            logger.error("No query provided in request")
            return jsonify({'error': 'No query provided'}), 400

        user_query = data['query']
        selected_language = data.get('language', 'en-IN')
        logger.info(f"Processing query: {user_query} in language: {selected_language}")

        # Translate query to English if it's in a regional language
        translated_query = user_query
        is_regional_language = selected_language != 'en-IN'

        if is_regional_language:
            logger.info(f"Translating query from {selected_language} to English")
            translated_query = translation_service.translate_to_english(user_query, selected_language)
            logger.info(f"Translated query: {translated_query}")

        search_query = translated_query

        # Check if vector store is ready
        doc_count = vector_store.get_document_count()
        logger.info(f"Vector store has {doc_count} documents")

        if doc_count == 0:
            logger.warning("No documents in vector store")
            return jsonify({
                'response': 'NO_DOCUMENTS_MESSAGE',  # This will be translated on frontend
                'sources': [{'document': 'System', 'page': 'N/A', 'relevance': 'No documents uploaded'}],
                'query': user_query,
                'message_type': 'no_documents'
            })

        # Retrieve relevant documents using translated query
        logger.info("Searching for relevant documents...")
        relevant_docs = vector_store.search_documents(search_query, k=10)
        logger.info(f"Found {len(relevant_docs)} relevant documents")

        if not relevant_docs:
            logger.warning("No relevant documents found for the query")
            return jsonify({
                'response': 'NO_RELEVANT_DOCS_MESSAGE',  # This will be translated on frontend
                'sources': [{'document': 'Search Results', 'page': 'N/A', 'relevance': 'No relevant documents found'}],
                'query': user_query,
                'message_type': 'no_relevant_docs',
                'doc_count': vector_store.get_document_count()
            })

        # Generate LLM response
        logger.info(f"Generating AI decision with {len(relevant_docs)} documents...")
        logger.info(f"Top document excerpts: {[doc['text'][:100] + '...' for doc in relevant_docs[:3]]}")

        try:
            response = llm_client.generate_decision(search_query, relevant_docs)
            logger.info(f"LLM response type: {type(response)}")
            logger.info(f"LLM response keys: {response.keys() if isinstance(response, dict) else 'Not a dict'}")

            if isinstance(response, dict) and 'response' in response:
                logger.info(f"Generated response preview: {response['response'][:200]}...")

            # Validate response structure for new conversational format
            if not isinstance(response, dict):
                logger.error(f"Invalid response type: {type(response)}")
                response = {
                    'response': 'I apologize, but I encountered an error while processing your request. Please try again.',
                    'sources': [{'document': 'System Error', 'page': 'N/A', 'relevance': 'Error response'}]
                }

            # Handle both old and new format responses
            if 'response' not in response:
                # Convert old format to new if needed
                old_response = response.copy()
                response = {
                    'response': f"**Query:** {user_query}\n\n**Analysis:** {old_response.get('summary', 'No analysis available')}\n\n**Decision:** {old_response.get('decision', 'Unknown')}\n\n**Amount:** {old_response.get('amount', 'N/A')}",
                    'sources': [{
                        'document': old_response.get('justification', {}).get('source', 'Unknown'),
                        'page': str(old_response.get('justification', {}).get('page', 'N/A')),
                        'relevance': 'Policy document analysis'
                    }]
                }

            # Ensure sources exist
            if 'sources' not in response or not response['sources']:
                response['sources'] = [{'document': 'Policy Document', 'page': 'N/A', 'relevance': 'General policy information'}]

            # Translate response back to selected language if needed
            if is_regional_language and 'response' in response:
                logger.info(f"Translating response back to {selected_language}")
                translated_response = translation_service.translate_from_english(response['response'], selected_language)
                response['response'] = translated_response
                logger.info(f"Translated response: {translated_response[:200]}...")

            # Add the original query to the response
            response['query'] = user_query

            logger.info(f"Generated response: {response}")
            return jsonify(response)

        except Exception as llm_error:
            logger.error(f"LLM generation error: {str(llm_error)}", exc_info=True)
            return jsonify({
                'response': f"""I apologize, but I encountered a technical issue while analyzing your query: "{user_query}".

**Error Details:** There was a problem with the AI service that processes insurance documents.

**What you can do:**
• Try rephrasing your question
• Check if your documents were uploaded properly
• Try again in a few moments

**Technical Info:** {str(llm_error)}

I'm designed to help you understand your insurance coverage, so please feel free to try again!""",
                'sources': [{'document': 'System Error', 'page': 'N/A', 'relevance': 'Error response'}],
                'query': user_query
            })

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return jsonify({
            'error': f'An error occurred while processing your query: {str(e)}. Please try again.'
        }), 500

@app.route('/new-chat', methods=['POST'])
def new_chat():
    try:
        # Clear all uploaded files
        import glob
        for old_file in glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*')):
            if os.path.isfile(old_file) and not old_file.endswith('.gitkeep'):
                os.remove(old_file)

        # Clear vector store completely
        vector_store.clear_all_documents()
        logger.info("Started new chat session - cleared all documents and vector store")

        return jsonify({
            'message': 'New chat session started',
            'documents_loaded': 0,
            'vector_store_ready': False
        })

    except Exception as e:
        logger.error(f"Error starting new chat: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def get_status():
    try:
        doc_count = vector_store.get_document_count()
        return jsonify({
            'status': 'ready',
            'documents_loaded': doc_count,
            'vector_store_ready': doc_count > 0
        })
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/webhook/document-upload', methods=['POST'])
def webhook_document_upload():
    """Webhook endpoint for document upload notifications"""
    try:
        data = request.get_json()
        logger.info(f"Webhook received: {data}")
        
        # Process webhook data
        event_type = data.get('event_type', 'document_uploaded')
        document_info = data.get('document', {})
        
        # You can add custom logic here based on the webhook data
        response = {
            'status': 'success',
            'message': 'Webhook processed successfully',
            'event_type': event_type,
            'timestamp': data.get('timestamp'),
            'received_at': str(datetime.utcnow())
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Webhook processing error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Webhook processing failed: {str(e)}'
        }), 500

@app.route('/webhook/query', methods=['POST'])
def webhook_query():
    """Webhook endpoint for query notifications"""
    try:
        data = request.get_json()
        logger.info(f"Query webhook received: {data}")
        
        query = data.get('query', '')
        user_id = data.get('user_id', 'anonymous')
        
        # Process the query through webhook
        if query:
            # You can integrate this with your existing query processing
            response_data = {
                'status': 'received',
                'query': query,
                'user_id': user_id,
                'message': 'Query received via webhook'
            }
        else:
            response_data = {
                'status': 'error',
                'message': 'No query provided'
            }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Query webhook error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Query webhook failed: {str(e)}'
        }), 500

@app.route('/api/v1/hackrx/run', methods=['POST'])
def hackrx_run():
    """HackRX API endpoint for document processing and question answering"""
    try:
        # Get the payload
        payload = request.get_json()
        logger.info(f"HackRX API received request: {payload}")
        
        if not payload:
            return jsonify({'error': 'No JSON payload provided'}), 400
            
        # Validate required fields
        if 'documents' not in payload or 'questions' not in payload:
            return jsonify({'error': 'Missing required fields: documents and questions'}), 400
            
        documents_url = payload.get('documents')
        questions = payload.get('questions', [])
        
        if not documents_url:
            return jsonify({'error': 'Document URL is required'}), 400
            
        if not questions or not isinstance(questions, list):
            return jsonify({'error': 'Questions must be a non-empty list'}), 400
        
        logger.info(f"Processing document URL: {documents_url}")
        logger.info(f"Number of questions: {len(questions)}")
        
        # Download and process the document
        import requests
        import tempfile
        import os
        
        # Clear previous documents
        vector_store.clear_all_documents()
        
        # Download the document
        try:
            response = requests.get(documents_url, timeout=30)
            response.raise_for_status()
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(response.content)
                temp_filepath = temp_file.name
            
            # Process the PDF
            filename = 'downloaded_policy.pdf'
            chunks = document_processor.process_pdf(temp_filepath, filename)
            
            # Add to vector store
            vector_store.add_documents(chunks)
            
            # Clean up temp file
            os.unlink(temp_filepath)
            
            logger.info(f"Successfully processed document with {len(chunks)} chunks")
            
        except requests.RequestException as e:
            logger.error(f"Error downloading document: {str(e)}")
            return jsonify({'error': f'Failed to download document: {str(e)}'}), 400
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return jsonify({'error': f'Failed to process document: {str(e)}'}), 500
        
        # Process each question
        answers = []
        
        for question in questions:
            try:
                logger.info(f"Processing question: {question}")
                
                # Search for relevant documents
                relevant_docs = vector_store.search_documents(question, k=10)
                
                if not relevant_docs:
                    logger.warning(f"No relevant documents found for question: {question}")
                    answers.append("I couldn't find relevant information in the document to answer this question.")
                    continue
                
                # Generate answer using LLM
                response = llm_client.generate_decision(question, relevant_docs)
                
                if isinstance(response, dict) and 'response' in response:
                    answer = response['response']
                else:
                    answer = str(response) if response else "I couldn't generate a proper answer for this question."
                
                answers.append(answer)
                logger.info(f"Generated answer for question: {question[:50]}...")
                
            except Exception as e:
                logger.error(f"Error processing question '{question}': {str(e)}")
                answers.append(f"Error processing this question: {str(e)}")
        
        # Return the structured response
        response_data = {
            'answers': answers
        }
        
        logger.info(f"Returning {len(answers)} answers")
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"HackRX API error: {str(e)}", exc_info=True)
        return jsonify({
            'error': f'API processing failed: {str(e)}'
        }), 500

with app.app_context():
    # Import models to ensure tables are created
    import models
    db.create_all()

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
