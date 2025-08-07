
import requests
import logging
from typing import Dict

logger = logging.getLogger(__name__)

import requests
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class TranslationService:
    def __init__(self):
        # Language code mapping for different services
        self.language_mapping = {
            'en-IN': 'en',
            'hi-IN': 'hi',
            'bn-IN': 'bn',
            'te-IN': 'te',
            'mr-IN': 'mr',
            'ta-IN': 'ta',
            'gu-IN': 'gu',
            'kn-IN': 'kn',
            'ml-IN': 'ml',
            'pa-IN': 'pa',
            'or-IN': 'or',
            'as-IN': 'as'
        }
        
        # Language names for better logging
        self.language_names = {
            'en-IN': 'English',
            'hi-IN': 'Hindi',
            'bn-IN': 'Bengali',
            'te-IN': 'Telugu',
            'mr-IN': 'Marathi',
            'ta-IN': 'Tamil',
            'gu-IN': 'Gujarati',
            'kn-IN': 'Kannada',
            'ml-IN': 'Malayalam',
            'pa-IN': 'Punjabi',
            'or-IN': 'Odia',
            'as-IN': 'Assamese'
        }
    
    def translate_text(self, text: str, target_language: str, source_language: str = 'en') -> str:
        """
        Translate text from source language to target language
        """
        if not text or not text.strip():
            return text
            
        # If target is English or same as source, return original
        target_code = self.language_mapping.get(target_language, target_language)
        if target_code == 'en' or target_code == source_language:
            return text
            
        logger.info(f"Translating text to {self.language_names.get(target_language, target_language)}")
        
        # Try multiple translation methods
        translated = self._try_mymemory_translation(text, source_language, target_code)
        if translated and translated != text:
            return translated
            
        translated = self._try_libre_translation(text, source_language, target_code)
        if translated and translated != text:
            return translated
            
        # If all translation attempts fail, return original text
        logger.warning(f"All translation attempts failed for {target_language}, using original English text")
        return text
    
    def _try_mymemory_translation(self, text: str, source: str, target: str) -> Optional[str]:
        """Try translation using MyMemory API"""
        try:
            url = "https://api.mymemory.translated.net/get"
            params = {
                'q': text,
                'langpair': f"{source}|{target}",
                'de': 'insurance@docuanalyzer.com'
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('responseStatus') == 200:
                    translated = data.get('responseData', {}).get('translatedText', '')
                    if translated and translated.lower() != text.lower():
                        logger.info(f"MyMemory translation successful for {target}")
                        return translated
                        
        except Exception as e:
            logger.error(f"MyMemory translation error: {e}")
        
        return None
    
    def _try_libre_translation(self, text: str, source: str, target: str) -> Optional[str]:
        """Try translation using LibreTranslate (if available)"""
        try:
            # Public LibreTranslate instance
            url = "https://libretranslate.de/translate"
            data = {
                'q': text,
                'source': source,
                'target': target,
                'format': 'text'
            }
            
            response = requests.post(url, data=data, timeout=10)
            if response.status_code == 200:
                result = response.json()
                translated = result.get('translatedText', '')
                if translated and translated.lower() != text.lower():
                    logger.info(f"LibreTranslate translation successful for {target}")
                    return translated
                    
        except Exception as e:
            logger.error(f"LibreTranslate translation error: {e}")
        
        return None
    
    def get_supported_languages(self):
        """Get list of supported languages"""
        return list(self.language_mapping.keys())

    def translate_to_english(self, text: str, source_language: str) -> str:
        """
        Translate text from regional language to English with improved error handling
        """
        if source_language == 'en-IN':
            return text
            
        try:
            source_lang = self.language_mapping.get(source_language, 'auto')
            logger.info(f"Translating from {source_lang} to English: {text}")
            
            # Try multiple translation methods
            translation_methods = [
                self._translate_mymemory,
                self._translate_google_unofficial,
                self._translate_libre
            ]
            
            for method in translation_methods:
                try:
                    translated_text = method(text, source_lang, 'en')
                    if translated_text and translated_text.strip() and translated_text != text:
                        logger.info(f"Translation successful via {method.__name__}: {translated_text}")
                        return translated_text.strip()
                except Exception as method_error:
                    logger.warning(f"Translation method {method.__name__} failed: {str(method_error)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            
        # If all translation attempts fail, try to detect if text is already in English
        if self._is_english_text(text):
            logger.info("Text appears to be in English already")
            return text
            
        # Final fallback: return original text
        logger.warning("All translation attempts failed, using original text")
        return text

    def _translate_mymemory(self, text: str, source_lang: str, target_lang: str) -> str:
        """MyMemory translation API"""
        url = "https://api.mymemory.translated.net/get"
        params = {
            'q': text,
            'langpair': f"{source_lang}|{target_lang}"
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            result = response.json()
            if result.get('responseStatus') == 200:
                return result['responseData']['translatedText']
        return None

    def _translate_google_unofficial(self, text: str, source_lang: str, target_lang: str) -> str:
        """Google Translate unofficial API"""
        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            'client': 'gtx',
            'sl': source_lang,
            'tl': target_lang,
            'dt': 't',
            'q': text
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=15)
        if response.status_code == 200:
            result = response.json()
            if result and len(result) > 0 and isinstance(result[0], list):
                # Combine all translation segments
                translated_parts = []
                for segment in result[0]:
                    if isinstance(segment, list) and len(segment) > 0:
                        translated_parts.append(segment[0])
                
                if translated_parts:
                    return ''.join(translated_parts)
        return None

    def _translate_libre(self, text: str, source_lang: str, target_lang: str) -> str:
        """LibreTranslate API (free service)"""
        url = "https://libretranslate.com/translate"
        data = {
            'q': text,
            'source': source_lang,
            'target': target_lang,
            'format': 'text'
        }
        
        try:
            response = requests.post(url, data=data, timeout=10)
            if response.status_code == 200:
                result = response.json()
                return result.get('translatedText')
        except:
            pass
        return None

    def translate_from_english(self, text: str, target_language: str) -> str:
        """
        Translate text from English to regional language with improved reliability
        """
        if target_language == 'en-IN':
            return text
            
        try:
            target_lang = self.language_mapping.get(target_language, 'en')
            logger.info(f"Translating from English to {target_lang}: {text[:100]}...")
            
            # Try multiple translation methods
            translation_methods = [
                self._translate_mymemory,
                self._translate_google_unofficial,
                self._translate_libre
            ]
            
            for method in translation_methods:
                try:
                    translated_text = method(text, 'en', target_lang)
                    if translated_text and translated_text.strip() and translated_text != text:
                        logger.info(f"Translation successful via {method.__name__}: {translated_text[:100]}...")
                        return translated_text.strip()
                except Exception as method_error:
                    logger.warning(f"Translation method {method.__name__} failed: {str(method_error)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            
        # Fallback: return original text if translation fails
        logger.warning("All translation attempts failed, using original English text")
        return text

    def _is_english_text(self, text: str) -> bool:
        """
        Simple heuristic to detect if text is likely English
        """
        try:
            # Count English alphabet characters vs total characters
            english_chars = sum(1 for c in text.lower() if 'a' <= c <= 'z')
            total_chars = len([c for c in text if c.isalpha()])
            
            if total_chars == 0:
                return True  # No alphabetic characters, assume English
                
            # If more than 70% of alphabetic characters are English, consider it English
            english_ratio = english_chars / total_chars
            return english_ratio > 0.7
            
        except Exception:
            return False

    def detect_language(self, text: str) -> str:
        """
        Detect the language of the input text
        """
        try:
            url = "https://translate.googleapis.com/translate_a/single"
            params = {
                'client': 'gtx',
                'sl': 'auto',
                'tl': 'en',
                'dt': 't',
                'q': text
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result and len(result) > 2:
                    detected_lang = result[2]
                    logger.info(f"Detected language: {detected_lang}")
                    return detected_lang
                    
        except Exception as e:
            logger.error(f"Language detection error: {str(e)}")
            
        return 'en'  # Default to English
