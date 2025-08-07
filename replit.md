# Overview

This is an AI-powered insurance document processing system that combines document parsing, vector search, and LLM-based decision making. The application allows users to upload PDF insurance documents, process them into searchable chunks, and then query the system for insurance coverage decisions based on the document content.

The system provides a modern chat-like interface where users can ask questions about insurance coverage, and the AI will analyze relevant document sections to provide structured decisions with justifications citing specific clauses and page references.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Pure HTML/CSS/JavaScript** implementation without React or other frameworks
- **Bootstrap 5** for responsive design components and mobile compatibility
- **Chat-like interface** with message bubbles for user queries and AI responses
- **Drag-and-drop file upload** functionality with visual feedback
- **Dieter Rams-inspired design** principles emphasizing minimalism and functionality
- **Mobile-responsive** layout that adapts to different screen sizes

## Backend Architecture
- **Flask web framework** with SQLAlchemy ORM for the main application server
- **Modular component design** with separate classes for different responsibilities:
  - `DocumentProcessor`: Handles PDF parsing and text extraction using PyMuPDF
  - `VectorStore`: Manages document embeddings and similarity search using SentenceTransformers
  - `LLMClient`: Interfaces with OpenRouter API for AI decision generation
- **RESTful API endpoints** for file uploads, document processing, and query handling
- **SQLite database** for storing document metadata and query history

## Data Processing Pipeline
- **PDF text extraction** using PyMuPDF (fitz) with text cleaning and normalization
- **Document chunking** strategy with configurable chunk size (1000 chars) and overlap (200 chars)
- **Vector embeddings** generated using SentenceTransformers 'all-MiniLM-L6-v2' model
- **Semantic search** for retrieving relevant document sections based on user queries
- **Structured response generation** with decision, amount, and justification fields

## Database Schema
- **Document table**: Stores uploaded file metadata including filename, size, page count, and processing status
- **Query table**: Logs user queries with responses, timestamps, and processing metrics
- **File-based vector storage**: Pickled numpy arrays for embeddings and document chunks

## Authentication and Security
- **File upload restrictions** limited to PDF files with 16MB maximum size
- **Secure filename handling** using Werkzeug's secure_filename utility
- **Environment-based configuration** for sensitive data like API keys and database URLs

# External Dependencies

## AI and ML Services
- **OpenRouter API**: LLM service using DeepSeek model (deepseek/deepseek-r1-0528-qwen3-8b:free) for generating insurance decisions
- **SentenceTransformers**: Local embedding model for document vectorization and similarity search

## Document Processing
- **PyMuPDF (fitz)**: PDF text extraction and document parsing
- **NumPy**: Vector operations and similarity calculations

## Web Framework and Database
- **Flask**: Web application framework with extensions for SQLAlchemy integration
- **SQLAlchemy**: ORM for database operations and model definitions
- **SQLite**: Default database for development (configurable via DATABASE_URL)
- **Werkzeug**: WSGI utilities for file handling and security

## Frontend Libraries
- **Bootstrap 5**: CSS framework for responsive design and UI components
- **Font Awesome 6**: Icon library for user interface elements

## Development and Deployment
- **ProxyFix**: Werkzeug middleware for handling proxy headers in deployment environments
- **Python logging**: Built-in logging system for debugging and monitoring