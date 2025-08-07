import fitz  # PyMuPDF
import logging
import re
from typing import List, Dict

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.chunk_size = 1000
        self.chunk_overlap = 200
    
    def process_pdf(self, filepath: str, filename: str) -> List[Dict]:
        """
        Process a PDF file and extract text chunks with metadata
        """
        doc = None
        try:
            doc = fitz.open(filepath)
            chunks = []
            page_count = len(doc)
            
            for page_num in range(page_count):
                page = doc[page_num]
                text = page.get_text()
                
                # Clean and normalize text
                text = self._clean_text(text)
                
                if not text.strip():
                    continue
                
                # Split text into chunks
                page_chunks = self._create_chunks(text, page_num + 1, filename)
                chunks.extend(page_chunks)
            
            logger.info(f"Processed {filename}: {len(chunks)} chunks from {page_count} pages")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing PDF {filename}: {str(e)}")
            raise
        finally:
            if doc is not None:
                try:
                    doc.close()
                except:
                    pass  # Ignore close errors
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page headers/footers (common patterns)
        text = re.sub(r'Page \d+ of \d+', '', text)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Fix common OCR issues
        text = text.replace('—', '-')
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()
    
    def _create_chunks(self, text: str, page_num: int, filename: str) -> List[Dict]:
        """
        Split text into overlapping chunks with metadata
        """
        chunks = []
        
        # Try to split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size, save current chunk
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # Identify clause information
                clause_info = self._extract_clause_info(current_chunk)
                
                chunks.append({
                    'text': current_chunk.strip(),
                    'metadata': {
                        'source': filename,
                        'page': page_num,
                        'clause_title': clause_info.get('title', ''),
                        'clause_number': clause_info.get('number', ''),
                        'chunk_id': len(chunks)
                    }
                })
                
                # Start new chunk with overlap
                current_chunk = sentence + " "
            else:
                current_chunk += sentence + " "
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            clause_info = self._extract_clause_info(current_chunk)
            chunks.append({
                'text': current_chunk.strip(),
                'metadata': {
                    'source': filename,
                    'page': page_num,
                    'clause_title': clause_info.get('title', ''),
                    'clause_number': clause_info.get('number', ''),
                    'chunk_id': len(chunks)
                }
            })
        
        return chunks
    
    def _extract_clause_info(self, text: str) -> Dict[str, str]:
        """
        Extract clause number and title from text
        """
        clause_info = {'title': '', 'number': ''}
        
        # Look for clause patterns like "Clause 12.3.1:", "Section 5:", etc.
        clause_patterns = [
            r'(?:Clause|Section|Article)\s+(\d+(?:\.\d+)*)\s*:?\s*([^\n.]+)',
            r'(\d+(?:\.\d+)*)\.\s*([A-Z][^.]+)',
            r'([A-Z][^.]+)\s*-\s*Clause\s+(\d+(?:\.\d+)*)'
        ]
        
        for pattern in clause_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if 'Clause' in pattern or 'Section' in pattern:
                    clause_info['number'] = match.group(1)
                    clause_info['title'] = match.group(2).strip()
                else:
                    clause_info['number'] = match.group(1)
                    clause_info['title'] = match.group(2).strip()
                break
        
        return clause_info
