import numpy as np
import pickle
import os
import re
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, max_features: int = 5000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9
        )
        self.vectors = None
        self.documents = []
        self.is_fitted = False
        self.vector_db_path = 'vector_db'
        self.vectorizer_file = os.path.join(self.vector_db_path, 'vectorizer.pkl')
        self.vectors_file = os.path.join(self.vector_db_path, 'vectors.pkl')
        self.docs_file = os.path.join(self.vector_db_path, 'documents.pkl')

        # Load existing vectors if available
        self._load_vectors()

    def add_documents(self, documents: List[Dict]):
        """
        Replace all documents in the vector store with new ones
        """
        try:
            logger.info(f"Replacing vector store with {len(documents)} new documents")

            # Clear existing documents and start fresh
            self.documents = []
            
            # Filter documents to keep only relevant content
            filtered_documents = []
            for doc in documents:
                if self._is_relevant_content(doc['text']):
                    filtered_documents.append(doc)
            
            logger.info(f"Filtered to {len(filtered_documents)} relevant documents from {len(documents)} total")
            
            if not filtered_documents:
                logger.warning("No relevant documents found after filtering")
                return
            
            # Extract texts for vectorization
            texts = [doc['text'] for doc in filtered_documents]

            # Set filtered documents as the collection
            self.documents = filtered_documents

            # Fit the vectorizer on new texts only
            self.vectors = self.vectorizer.fit_transform(texts)
            self.is_fitted = True

            # Save to disk
            self._save_vectors()

            logger.info(f"Successfully replaced vector store with {len(filtered_documents)} relevant documents")

        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise

    def search_documents(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for relevant documents using semantic similarity with improved parsing
        """
        if not self.vectorizer or not self.vectors.size:
            logger.warning("Vector store is empty")
            return []

        try:
            # Parse structured queries like "46M, knee surgery, Pune, 3-month policy"
            parsed_terms = self._parse_structured_query(query)
            expanded_query = f"{query} {' '.join(parsed_terms)}"

            # Transform query to vector
            query_vector = self.vectorizer.transform([expanded_query])

            # Calculate cosine similarity
            similarities = cosine_similarity(query_vector, self.vectors)[0]

            # Get top k results with lower threshold for better recall
            top_indices = similarities.argsort()[-k:][::-1]

            results = []
            for idx in top_indices:
                if similarities[idx] > 0.01:  # Lower threshold for better recall
                    doc = self.documents[idx].copy()
                    doc['similarity_score'] = float(similarities[idx])

                    # Boost relevance for key medical terms and query terms
                    text_lower = doc.get('text', '').lower()
                    query_words = set(query.lower().split())
                    
                    # Boost for parsed medical terms
                    for term in parsed_terms:
                        if term.lower() in text_lower:
                            doc['similarity_score'] = min(1.0, doc['similarity_score'] * 1.3)
                    
                    # Boost for direct query word matches
                    text_words = set(text_lower.split())
                    common_words = query_words.intersection(text_words)
                    if common_words:
                        boost_factor = min(1.5, 1.0 + len(common_words) * 0.1)
                        doc['similarity_score'] = min(1.0, doc['similarity_score'] * boost_factor)

                    results.append(doc)

            logger.info(f"Search results before filtering: {len(results)} documents with scores: {[r['similarity_score'] for r in results[:5]]}")

            # Sort by similarity score
            results.sort(key=lambda x: x['similarity_score'], reverse=True)

            logger.info(f"Found {len(results)} relevant documents for query: {query}")
            return results

        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []

    def _parse_structured_query(self, query: str) -> List[str]:
        """
        Parse structured queries to extract key terms
        """
        import re

        terms = []

        # Extract age patterns (e.g., "46M", "46-year-old")
        age_match = re.search(r'(\d+)[M|F|m|f]?[-\s]?(?:year|yr)', query, re.IGNORECASE)
        if age_match:
            terms.append("age")

        # Extract medical procedures
        medical_terms = ['surgery', 'operation', 'treatment', 'procedure', 'therapy']
        for term in medical_terms:
            if term.lower() in query.lower():
                terms.append(term)

        # Extract body parts
        body_parts = ['knee', 'hip', 'heart', 'eye', 'spine', 'shoulder', 'ankle']
        for part in body_parts:
            if part.lower() in query.lower():
                terms.append(part)

        # Extract policy duration patterns
        duration_match = re.search(r'(\d+)[-\s]?(?:month|year|day)', query, re.IGNORECASE)
        if duration_match:
            terms.extend(["waiting period", "policy duration"])

        # Extract location terms
        location_terms = ['pune', 'mumbai', 'delhi', 'bangalore', 'chennai', 'hyderabad']
        for location in location_terms:
            if location.lower() in query.lower():
                terms.append("location")

        return list(set(terms))  # Remove duplicates

    def get_document_count(self) -> int:
        """
        Get the number of documents in the store
        """
        return len(self.documents)

    def clear_all_documents(self):
        """
        Clear all documents from the vector store for fresh sessions
        """
        try:
            logger.info("Clearing all documents from vector store")
            self.documents = []
            self.vectors = None
            self.is_fitted = False
            
            # Remove stored files
            import glob
            for file_path in [self.vectorizer_file, self.vectors_file, self.docs_file]:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    
            logger.info("Successfully cleared vector store")
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")

    def _save_vectors(self):
        """
        Save vectorizer, vectors and documents to disk
        """
        try:
            os.makedirs(self.vector_db_path, exist_ok=True)

            # Save vectorizer
            with open(self.vectorizer_file, 'wb') as f:
                pickle.dump(self.vectorizer, f)

            # Save vectors
            with open(self.vectors_file, 'wb') as f:
                pickle.dump(self.vectors, f)

            # Save documents
            with open(self.docs_file, 'wb') as f:
                pickle.dump(self.documents, f)

        except Exception as e:
            logger.error(f"Error saving vectors: {str(e)}")

    def _load_vectors(self):
        """
        Load vectorizer, vectors and documents from disk
        """
        try:
            if (os.path.exists(self.vectorizer_file) and 
                os.path.exists(self.vectors_file) and 
                os.path.exists(self.docs_file)):

                # Load vectorizer
                with open(self.vectorizer_file, 'rb') as f:
                    self.vectorizer = pickle.load(f)

                # Load vectors
                with open(self.vectors_file, 'rb') as f:
                    self.vectors = pickle.load(f)

                # Load documents
                with open(self.docs_file, 'rb') as f:
                    self.documents = pickle.load(f)

                self.is_fitted = True
                logger.info(f"Loaded {len(self.documents)} documents from disk")
            else:
                logger.info("No existing vector database found, starting fresh")
                self.vectors = None
                self.documents = []
                self.is_fitted = False

        except Exception as e:
            logger.error(f"Error loading vectors: {str(e)}")
            self.vectors = None
            self.documents = []
            self.is_fitted = False

    def _is_relevant_content(self, text: str) -> bool:
        """
        Filter out irrelevant content like headers, footers, and boilerplate text
        """
        text_lower = text.lower()

        # Skip very short chunks that are likely headers/footers
        if len(text.strip()) < 50:
            return False

        # Skip chunks that are ONLY contact information (be more lenient)
        pure_contact_patterns = [
            'bajaj allianz house, airport road, yerawada, pune',
            'www.bajajallianz.com',
            'bagichelp@bajajallianz.co.in',
            'for more details, log on to:'
        ]

        # Only reject if text is mostly just contact info
        is_pure_contact = any(pattern in text_lower for pattern in pure_contact_patterns) and len(text.strip()) < 200
        if is_pure_contact:
            return False

        # Skip chunks that are just UIN headers without content
        if re.match(r'^uin[\-\s]*[a-z0-9]+\s*$', text_lower.strip()):
            return False

        # Accept chunks with ANY policy-relevant content (be more inclusive)
        relevant_keywords = [
            'clause', 'section', 'coverage', 'covered', 'excluded', 'waiting period',
            'sum insured', 'premium', 'deductible', 'co-pay', 'treatment', 'surgery',
            'hospitalization', 'medical', 'claim', 'benefit', 'condition', 'terms',
            'policy', 'insured', 'eligible', 'reimbursement', 'expenses', 'limit',
            'pre-existing', 'exclusion', 'inclusion', 'cashless', 'network hospital',
            'procedure', 'operation', 'disease', 'illness', 'injury', 'emergency',
            'ambulance', 'consultation', 'diagnostic', 'therapy', 'medicine',
            'rupees', '₹', 'amount', 'cost', 'fee', 'charge', 'payable'
        ]

        has_relevant_content = any(keyword in text_lower for keyword in relevant_keywords)

        # Also accept chunks with medical procedure codes or lists (these might be relevant)
        has_medical_codes = bool(re.search(r'\d+\.\d+|\d+\s+[a-z].*?(surgery|procedure|treatment|removal|repair)', text_lower))

        # Accept chunks with substantial text content (more lenient threshold)
        has_substantial_text = len([word for word in text.split() if len(word) > 2]) > 5

        # Accept if it has relevant content OR medical codes OR substantial text
        return has_relevant_content or has_medical_codes or (has_substantial_text and len(text.strip()) > 100)