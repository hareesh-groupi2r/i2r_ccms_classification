"""
Text Preprocessing Module
Handles NLP preprocessing including lemmatization, stemming, and sliding window creation
"""

import re
import string
from typing import List, Tuple, Optional
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Handles text preprocessing for contract correspondence documents.
    """
    
    def __init__(self, 
                 use_lemmatization: bool = True,
                 use_stemming: bool = False,
                 remove_stopwords: bool = True,
                 preserve_contract_terms: bool = True):
        """
        Initialize the text preprocessor.
        
        Args:
            use_lemmatization: Whether to apply lemmatization
            use_stemming: Whether to apply stemming (not recommended with lemmatization)
            remove_stopwords: Whether to remove stopwords
            preserve_contract_terms: Whether to preserve important contract terms
        """
        self.use_lemmatization = use_lemmatization
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        self.preserve_contract_terms = preserve_contract_terms
        
        # Initialize NLP components
        if use_lemmatization:
            self.lemmatizer = WordNetLemmatizer()
        if use_stemming:
            self.stemmer = PorterStemmer()
        
        # Load stopwords but preserve important contract terms
        self.stopwords = set(stopwords.words('english'))
        
        # Contract-specific terms to preserve (not remove as stopwords)
        self.contract_terms = {
            'shall', 'must', 'may', 'will', 'should', 'could',
            'agreement', 'contract', 'party', 'parties',
            'obligation', 'liability', 'indemnity', 'warranty',
            'breach', 'default', 'termination', 'force', 'majeure',
            'payment', 'invoice', 'milestone', 'deliverable',
            'change', 'variation', 'modification', 'amendment',
            'claim', 'dispute', 'arbitration', 'jurisdiction'
        }
        
        if preserve_contract_terms:
            self.stopwords = self.stopwords - self.contract_terms
        
        logger.info("TextPreprocessor initialized")
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters and normalizing whitespace.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        # Convert to string if not already
        text = str(text)
        
        # Remove multiple spaces and normalize whitespace
        text = ' '.join(text.split())
        
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\'\"]', ' ', text)
        
        # Remove multiple punctuation
        text = re.sub(r'([\.,:;!?])\1+', r'\1', text)
        
        # Normalize spaces around punctuation
        text = re.sub(r'\s+([\.,:;!?])', r'\1', text)
        
        return text.strip()
    
    def preprocess(self, text: str, 
                  lowercase: bool = True,
                  return_tokens: bool = False) -> str:
        """
        Apply full preprocessing pipeline to text.
        
        Args:
            text: Input text to preprocess
            lowercase: Whether to convert to lowercase
            return_tokens: Whether to return tokens instead of joined text
            
        Returns:
            Preprocessed text or list of tokens
        """
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Convert to lowercase if specified
        if lowercase:
            tokens = [token.lower() for token in tokens]
        
        # Remove punctuation (except important ones)
        tokens = [token for token in tokens 
                 if token not in string.punctuation or token in ['.', ',', ';', ':']]
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens 
                     if token.lower() not in self.stopwords or len(token) > 2]
        
        # Apply lemmatization
        if self.use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Apply stemming (not recommended with lemmatization)
        if self.use_stemming and not self.use_lemmatization:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        if return_tokens:
            return tokens
        
        return ' '.join(tokens)
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        Extract sentences from text.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        text = self.clean_text(text)
        sentences = sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def create_sliding_windows(self, 
                              text: str, 
                              window_size: int = 3, 
                              overlap: int = 1) -> List[Tuple[str, int, int]]:
        """
        Create sliding windows of sentences for better context understanding.
        
        Args:
            text: Input text
            window_size: Number of sentences per window
            overlap: Number of overlapping sentences between windows
            
        Returns:
            List of tuples (window_text, start_sentence_idx, end_sentence_idx)
        """
        sentences = self.extract_sentences(text)
        
        if not sentences:
            return []
        
        if window_size > len(sentences):
            # If text is shorter than window size, return entire text as single window
            return [(' '.join(sentences), 0, len(sentences)-1)]
        
        windows = []
        stride = window_size - overlap
        
        if stride <= 0:
            stride = 1
            logger.warning(f"Overlap ({overlap}) >= window_size ({window_size}), using stride of 1")
        
        for i in range(0, len(sentences), stride):
            end_idx = min(i + window_size, len(sentences))
            window_sentences = sentences[i:end_idx]
            window_text = ' '.join(window_sentences)
            windows.append((window_text, i, end_idx - 1))
            
            # Stop if we've reached the end
            if end_idx >= len(sentences):
                break
        
        logger.debug(f"Created {len(windows)} windows from {len(sentences)} sentences")
        return windows
    
    def extract_key_sentences(self, 
                             text: str, 
                             keywords: List[str] = None,
                             max_sentences: int = 10) -> List[str]:
        """
        Extract key sentences that likely contain important information.
        
        Args:
            text: Input text
            keywords: Optional list of keywords to prioritize
            max_sentences: Maximum number of sentences to return
            
        Returns:
            List of key sentences
        """
        sentences = self.extract_sentences(text)
        
        if not sentences:
            return []
        
        # Default contract-related keywords
        if keywords is None:
            keywords = [
                'change', 'scope', 'payment', 'delay', 'claim', 'dispute',
                'milestone', 'deliverable', 'variation', 'modification',
                'breach', 'default', 'liability', 'cost', 'schedule',
                'completion', 'extension', 'approval', 'notice', 'request'
            ]
        
        # Score sentences based on keyword presence
        scored_sentences = []
        for sent in sentences:
            sent_lower = sent.lower()
            score = sum(1 for kw in keywords if kw.lower() in sent_lower)
            
            # Boost score for sentences with multiple keywords
            if score > 1:
                score *= 1.5
            
            # Boost score for sentences with numbers (likely important)
            if any(char.isdigit() for char in sent):
                score += 0.5
            
            scored_sentences.append((sent, score))
        
        # Sort by score and return top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Return sentences maintaining original order
        top_sentences = [sent for sent, score in scored_sentences[:max_sentences] if score > 0]
        
        # If no sentences with keywords, return first few sentences
        if not top_sentences and sentences:
            top_sentences = sentences[:min(3, len(sentences))]
        
        return top_sentences
    
    def normalize_document(self, text: str) -> str:
        """
        Normalize entire document for consistent processing.
        
        Args:
            text: Input document text
            
        Returns:
            Normalized text
        """
        # Clean text
        text = self.clean_text(text)
        
        # Fix common OCR errors if present
        text = self.fix_ocr_errors(text)
        
        # Normalize section headers
        text = re.sub(r'^(\d+\.)+\s*', '', text, flags=re.MULTILINE)
        
        # Normalize dates
        text = re.sub(r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b', r'\1/\2/\3', text)
        
        # Normalize amounts (remove currency symbols for processing)
        text = re.sub(r'[$£€¥]\s*', '', text)
        
        return text
    
    def fix_ocr_errors(self, text: str) -> str:
        """
        Fix common OCR errors in scanned documents.
        
        Args:
            text: Input text potentially containing OCR errors
            
        Returns:
            Text with common OCR errors fixed
        """
        # Common OCR substitutions
        replacements = {
            r'\bl\b': 'I',  # Lowercase L often misread as I
            r'\bO\b': '0',  # Letter O misread as zero in numbers
            r'\.{2,}': '...',  # Multiple dots to ellipsis
            r'\s+': ' ',  # Multiple spaces to single space
            r'([a-z])([A-Z])': r'\1 \2',  # Add space between lower and uppercase
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def chunk_text(self, text: str, max_chunk_size: int = 3000) -> List[str]:
        """
        Chunk text into smaller pieces for processing with token limits.
        
        Args:
            text: Input text
            max_chunk_size: Maximum characters per chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= max_chunk_size:
            return [text]
        
        # Try to chunk by sentences first
        sentences = self.extract_sentences(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sent in sentences:
            sent_size = len(sent)
            
            if current_size + sent_size > max_chunk_size and current_chunk:
                # Save current chunk and start new one
                chunks.append(' '.join(current_chunk))
                current_chunk = [sent]
                current_size = sent_size
            else:
                current_chunk.append(sent)
                current_size += sent_size + 1  # +1 for space
        
        # Add last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks
    
    def __repr__(self):
        return (f"TextPreprocessor(lemmatization={self.use_lemmatization}, "
                f"stemming={self.use_stemming}, "
                f"remove_stopwords={self.remove_stopwords})")