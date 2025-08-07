import requests
import json
import os
import logging
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self):
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        self.model = "deepseek/deepseek-r1-0528-qwen3-8b:free"
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def generate_decision(self, query: str, relevant_docs: List[Dict]) -> Dict:
        """
        Generate an insurance decision based on query and relevant documents
        """
        try:
            logger.info(f"Starting decision generation for query: {query}")
            logger.info(f"Number of relevant docs: {len(relevant_docs)}")

            # If no documents are relevant, provide a specific message
            if not relevant_docs:
                return {
                    'response': 'Please upload policy documents first to analyze your query.\n\nWhat this means:\n• Your query might be about something not covered in the uploaded documents\n• The documents might not contain the specific information you\'re looking for\n• You might need to upload additional policy documents\n\nSuggestions:\n• Try rephrasing your question with more specific terms\n• Upload any additional policy documents you might have\n• Check if your question relates to the type of insurance policy you\'ve uploaded',
                    'sources': [],
                    'query': query
                }

            # First, check if this is a legitimate insurance query
            if self._is_non_insurance_query(query):
                return self._create_non_insurance_response(query)

            # Prepare context from relevant documents
            context = self._prepare_context(relevant_docs)
            logger.info(f"Context prepared, length: {len(context)}")

            # Create system prompt
            system_prompt = self._create_system_prompt()

            # Create user prompt
            user_prompt = self._create_user_prompt(query, context)

            # Make API call
            logger.info("Making LLM API call...")
            response = self._call_llm(system_prompt, user_prompt)
            logger.info(f"LLM response received, length: {len(response)}")

            # Parse and validate response
            fallback_doc = relevant_docs[0] if relevant_docs else {}
            parsed_response = self._parse_response(response, fallback_doc, query)
            logger.info(f"Response parsed successfully: {type(parsed_response)}")

            return parsed_response

        except Exception as e:
            logger.error(f"Error generating decision: {str(e)}", exc_info=True)
            return {
                'decision': 'Error',
                'amount': 'N/A',
                'justification': {
                    'clause': f'Error processing request: {str(e)}',
                    'source': 'System Error',
                    'page': 'N/A'
                },
                'summary': 'An error occurred while processing your request.',
                'quoted_text': 'N/A'
            }

    def _is_non_insurance_query(self, query: str) -> bool:
        """
        Check if the query is not actually related to insurance matters
        """
        query_lower = query.lower().strip()

        # Very short or nonsensical queries
        if len(query_lower) < 3:
            return True

        # Insurance-related keywords that should NOT be filtered out
        insurance_keywords = [
            'coverage', 'covered', 'claim', 'policy', 'premium', 'deductible',
            'benefit', 'treatment', 'disease', 'illness', 'condition', 'medical',
            'hospital', 'doctor', 'surgery', 'medication', 'therapy', 'diagnosis',
            'accident', 'injury', 'disability', 'travel', 'flight', 'cancellation',
            'property', 'damage', 'loss', 'theft', 'fire', 'flood', 'earthquake',
            'dental', 'vision', 'maternity', 'pregnancy', 'mental health',
            'pre-existing', 'waiting period', 'exclusion', 'limit', 'sum insured',
            'reimbursement', 'cashless', 'network hospital', 'co-pay'
        ]

        # If query contains insurance keywords, it's likely insurance-related
        for keyword in insurance_keywords:
            if keyword in query_lower:
                return False

        # Common non-insurance items (only very obvious ones)
        obvious_non_insurance = [
            'candy', 'chocolate', 'snack', 'toy', 'game', 'pencil', 'pen',
            'homework', 'test score', 'weather', 'time', 'date', 'joke'
        ]

        # Only filter out if it's obviously not insurance-related
        for item in obvious_non_insurance:
            if query_lower.startswith(item) or query_lower == item:
                return True

        # Default to treating as insurance-related to be more helpful
        return False

    def _create_non_insurance_response(self, query: str) -> Dict:
        """
        Create a friendly response for non-insurance queries
        """
        return {
            'decision': 'Not applicable',
            'amount': 'N/A',
            'justification': {
                'clause': f'I understand you\'re asking about "{query}", but this doesn\'t appear to be related to insurance coverage. I\'m designed to help with insurance claims like medical treatments, travel issues, property damage, or other covered events.',
                'source': 'Assistant Guidelines',
                'page': 'N/A'
            },
            'summary': f'Your query about "{query}" isn\'t something that would typically be covered by insurance. If you have questions about actual insurance coverage, I\'d be happy to help!',
            'quoted_text': 'N/A'
        }

    def _prepare_context(self, relevant_docs: List[Dict]) -> str:
        """
        Prepare context from relevant documents
        """
        context_parts = []

        for i, doc in enumerate(relevant_docs[:3]):  # Use top 3 docs for better context
            metadata = doc.get('metadata', {})
            text = doc.get('text', '')[:600]  # Slightly longer text for better context
            similarity = doc.get('similarity_score', 0)

            # Structure each document section clearly
            context_part = f"""
Document {i+1} (Relevance: {similarity:.2f}):
File: {metadata.get('source', 'Unknown')}
Page: {metadata.get('page', '?')}
Clause: {metadata.get('clause_title', 'N/A')}
Content: {text}
---"""
            context_parts.append(context_part)

        return '\n'.join(context_parts)

    def _create_system_prompt(self) -> str:
        """
        Create system prompt for the LLM
        """
        return """You are an AI assistant specialized in analyzing insurance policy documents. You help users understand their coverage by examining specific policy clauses and providing clear, actionable answers.

        IMPORTANT: You MUST base your response on the provided document context. Do NOT provide generic insurance advice.

        When answering questions about insurance coverage:
        1. ALWAYS reference specific information from the provided documents
        2. Quote relevant clauses, sections, or policy terms when available
        3. Provide specific page numbers or document references
        4. Give clear coverage decisions when the documents support them
        5. Use simple, everyday language that's easy to understand
        6. If the documents don't contain enough information for a specific answer, explain what information is missing

        Context from uploaded insurance documents:
        {context_text}

        User question: {query}

        Based on the specific policy documents provided above, please analyze and respond to the user's question. Reference specific clauses, sections, or terms from these documents in your response.
"""

    def _create_user_prompt(self, query: str, context: str) -> str:
        """
        Create user prompt with query and context
        """
        truncated_context = context[:4000] + "..." if len(context) > 4000 else context
        return f"""The customer is asking: "{query}"

Here are the relevant sections from their insurance policy documents:
{truncated_context}

Please answer their question in a natural, conversational way. Look through the documents for specific information about their question. If you find coverage details, amounts, lists of covered items, exclusions, or other specific information, share it clearly.

Just respond like you're having a normal conversation - no templates, headers, or structured formats. Be helpful and direct."""

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Make API call to OpenRouter with improved error handling
        """
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://insurance-doc-processor.replit.app',
            'X-Title': 'Insurance Document Processor'
        }

        data = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            'temperature': 0.3,
            'max_tokens': 600,
            'stream': False
        }

        try:
            logger.info(f"Making API call to OpenRouter with model: {self.model}")
            response = requests.post(self.base_url, headers=headers, json=data, timeout=10)

            logger.info(f"API Response Status: {response.status_code}")

            if response.status_code != 200:
                logger.error(f"API call failed: {response.status_code} - {response.text}")
                return self._get_fallback_ai_response()

            result = response.json()

            if 'choices' not in result or not result['choices']:
                logger.error(f"Invalid API response structure: {result}")
                return self._get_fallback_ai_response()

            content = result['choices'][0]['message']['content']
            logger.info("API call successful, received response")
            return content

        except requests.exceptions.Timeout:
            logger.error("API call timed out after 10 seconds")
            return self._create_timeout_response()
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error: {str(e)}")
            return self._create_connection_error_response()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return self._get_fallback_ai_response()
        except Exception as e:
            logger.error(f"Unexpected error in API call: {str(e)}")
            return self._get_fallback_ai_response()

    def _get_fallback_ai_response(self) -> str:
        """
        Generate a fallback AI response when API fails
        """
        return "I'm having trouble connecting to the AI service right now. Could you try asking your question again in a moment? If the problem continues, there might be a temporary technical issue."

    def _create_timeout_response(self) -> str:
        """
        Generate response for timeout errors
        """
        return "Your question is taking longer than usual to process. This might be due to high server load. Could you try asking again, or maybe phrase your question a bit differently?"

    def _create_connection_error_response(self) -> str:
        """
        Generate response for connection errors
        """
        return "I'm having trouble connecting to the AI service. Please check your internet connection and try again. If the problem persists, it might be a temporary network issue."

    def _parse_response(self, response: str, fallback_doc: Dict = {}, query: str = "") -> Dict:
        """
        Parse LLM response and return in conversational format
        """
        try:
            # Clean the response
            response = response.strip()

            # Create source info from the fallback document
            sources = []
            if fallback_doc:
                metadata = fallback_doc.get('metadata', {})
                sources.append({
                    'document': metadata.get('source', 'Policy Document'),
                    'page': str(metadata.get('page', 'N/A')),
                    'relevance': 'Contains relevant policy information for this query'
                })

            # Return the natural response directly
            return {
                'response': response,
                'sources': sources,
                'query': query
            }

        except Exception as e:
            logger.error(f"Error in response parsing: {str(e)}")
            return {
                'response': f"I apologize, but I encountered an error while processing your question about '{query}'. Please try asking again or rephrase your question.",
                'sources': [{'document': 'System Error', 'page': 'N/A', 'relevance': 'Error response'}],
                'query': query
            }