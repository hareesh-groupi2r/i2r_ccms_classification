"""
Document Quality Checker Module
Detects and filters out low-quality documents before classification
"""

import re
import logging
from typing import Dict, List, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


class DocumentQualityChecker:
    """
    Validates document quality and detects OCR artifacts, scan issues, and low-content documents.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the document quality checker.
        
        Args:
            config: Configuration dictionary with quality thresholds
        """
        self.config = config or {}
        
        # Quality thresholds
        self.min_text_length = self.config.get('min_text_length', 50)
        self.min_meaningful_words = self.config.get('min_meaningful_words', 10)
        self.max_repetition_ratio = self.config.get('max_repetition_ratio', 0.8)
        self.min_unique_words = self.config.get('min_unique_words', 5)
        
        # OCR artifact patterns
        self.ocr_artifacts = [
            r'scanned\s+by\s+cam\s*scanner',
            r'scanned\s+with\s+cam\s*scanner',
            r'created\s+with\s+cam\s*scanner',
            r'digitized\s+by\s+.*scanner',
            r'adobe\s+scan',
            r'microsoft\s+lens',
            r'office\s+lens',
            r'document\s+scanner',
            r'pdf\s+scanner',
            r'scan\s+to\s+pdf',
            r'scanning\s+app',
            r'mobile\s+scanner',
            r'iphone\s+scanner',
            r'android\s+scanner',
            r'document\s+capture',
            r'photo\s+to\s+pdf'
        ]
        
        # Compile patterns for performance
        self.artifact_patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in self.ocr_artifacts
        ]
        
        # Common meaningless patterns
        self.meaningless_patterns = [
            r'^[\s\n\r]*$',  # Empty or whitespace only
            r'^[^\w]*$',     # Only punctuation/symbols
            r'^\d+$',        # Only numbers
            r'^[a-z\s]*$',   # Only lowercase letters (often OCR artifacts)
            r'^[A-Z\s]*$',   # Only uppercase letters
        ]
        
        self.meaningless_compiled = [
            re.compile(pattern) for pattern in self.meaningless_patterns
        ]
        
        # Common filler words that don't indicate meaningful content
        self.filler_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'a', 'an', 'page', 'document', 'file', 'pdf', 'scan', 'scanned'
        }

    def check_document_quality(self, text: str) -> Dict:
        """
        Comprehensive document quality check.
        
        Args:
            text: Document text to analyze
            
        Returns:
            Dictionary with quality assessment results
        """
        logger.info(f"🔍 QUALITY CHECK: Analyzing document ({len(text)} chars)")
        
        # Initialize result
        result = {
            'is_quality': True,
            'quality_score': 1.0,
            'issues': [],
            'warnings': [],
            'metrics': {}
        }
        
        # Basic metrics
        text_length = len(text.strip())
        result['metrics']['text_length'] = text_length
        
        # Check 1: Minimum text length
        if text_length < self.min_text_length:
            result['is_quality'] = False
            result['issues'].append(f"Document too short: {text_length} chars < {self.min_text_length} minimum")
            result['quality_score'] *= 0.1
            logger.warning(f"   ❌ Text too short: {text_length} chars")
            
        # Check 2: OCR artifacts detection
        artifact_check = self._detect_ocr_artifacts(text)
        if artifact_check['has_artifacts']:
            result['is_quality'] = False
            result['issues'].extend(artifact_check['issues'])
            result['quality_score'] *= 0.2
            logger.warning(f"   ❌ OCR artifacts detected: {artifact_check['artifacts_found']}")
            
        # Check 3: Content meaningfulness
        meaningfulness_check = self._check_content_meaningfulness(text)
        if not meaningfulness_check['is_meaningful']:
            result['is_quality'] = False
            result['issues'].extend(meaningfulness_check['issues'])
            result['quality_score'] *= 0.3
            logger.warning(f"   ❌ Content not meaningful: {meaningfulness_check['reason']}")
            
        # Check 4: Repetition analysis
        repetition_check = self._analyze_repetition(text)
        if repetition_check['excessive_repetition']:
            result['is_quality'] = False
            result['issues'].extend(repetition_check['issues'])
            result['quality_score'] *= 0.4
            logger.warning(f"   ❌ Excessive repetition: {repetition_check['repetition_ratio']:.2f}")
            
        # Merge all metrics
        result['metrics'].update(artifact_check.get('metrics', {}))
        result['metrics'].update(meaningfulness_check.get('metrics', {}))
        result['metrics'].update(repetition_check.get('metrics', {}))
        
        # Final quality assessment
        if result['is_quality']:
            logger.info(f"   ✅ Document quality: GOOD (score: {result['quality_score']:.2f})")
        else:
            logger.warning(f"   ❌ Document quality: POOR (score: {result['quality_score']:.2f})")
            logger.warning(f"   📋 Issues: {', '.join(result['issues'])}")
            
        return result

    def _detect_ocr_artifacts(self, text: str) -> Dict:
        """Detect common OCR scanning artifacts."""
        result = {
            'has_artifacts': False,
            'artifacts_found': [],
            'issues': [],
            'metrics': {}
        }
        
        text_lower = text.lower()
        
        # Check for known OCR artifact patterns
        for i, pattern in enumerate(self.artifact_patterns):
            matches = pattern.findall(text_lower)
            if matches:
                result['has_artifacts'] = True
                artifact_name = self.ocr_artifacts[i]
                result['artifacts_found'].append(artifact_name)
                result['issues'].append(f"OCR artifact detected: '{artifact_name}' ({len(matches)} occurrences)")
        
        # Additional heuristics for OCR artifacts
        lines = text.split('\n')
        repeated_lines = [line.strip() for line in lines if line.strip()]
        
        if repeated_lines:
            line_counts = Counter(repeated_lines)
            most_common = line_counts.most_common(1)[0]
            
            # If the same line appears more than 3 times and it's the majority of content
            if most_common[1] > 3 and most_common[1] / len(repeated_lines) > 0.6:
                result['has_artifacts'] = True
                result['issues'].append(f"Repeated line detected: '{most_common[0][:50]}...' ({most_common[1]} times)")
        
        result['metrics']['artifact_patterns_found'] = len(result['artifacts_found'])
        return result

    def _check_content_meaningfulness(self, text: str) -> Dict:
        """Check if the document contains meaningful content."""
        result = {
            'is_meaningful': True,
            'issues': [],
            'reason': '',
            'metrics': {}
        }
        
        # Basic word analysis
        words = re.findall(r'\b\w+\b', text.lower())
        result['metrics']['total_words'] = len(words)
        
        if not words:
            result['is_meaningful'] = False
            result['reason'] = 'No recognizable words found'
            result['issues'].append('Document contains no recognizable words')
            return result
        
        # Check for minimum meaningful words
        if len(words) < self.min_meaningful_words:
            result['is_meaningful'] = False
            result['reason'] = f'Too few words: {len(words)} < {self.min_meaningful_words}'
            result['issues'].append(f'Document has only {len(words)} words (minimum: {self.min_meaningful_words})')
        
        # Check word diversity
        unique_words = set(words)
        result['metrics']['unique_words'] = len(unique_words)
        result['metrics']['word_diversity'] = len(unique_words) / len(words) if words else 0
        
        if len(unique_words) < self.min_unique_words:
            result['is_meaningful'] = False
            result['reason'] = f'Too few unique words: {len(unique_words)} < {self.min_unique_words}'
            result['issues'].append(f'Document has only {len(unique_words)} unique words (minimum: {self.min_unique_words})')
        
        # Check for meaningful vs filler words ratio
        meaningful_words = [w for w in words if w not in self.filler_words and len(w) > 2]
        filler_ratio = (len(words) - len(meaningful_words)) / len(words) if words else 1
        result['metrics']['meaningful_words'] = len(meaningful_words)
        result['metrics']['filler_ratio'] = filler_ratio
        
        if len(meaningful_words) < 3:
            result['is_meaningful'] = False
            result['reason'] = f'Too few meaningful words: {len(meaningful_words)}'
            result['issues'].append(f'Document has only {len(meaningful_words)} meaningful words')
        
        # Check for meaningless patterns
        text_clean = text.strip()
        for pattern in self.meaningless_compiled:
            if pattern.match(text_clean):
                result['is_meaningful'] = False
                result['reason'] = 'Document matches meaningless pattern'
                result['issues'].append('Document content matches known meaningless patterns')
                break
        
        return result

    def _analyze_repetition(self, text: str) -> Dict:
        """Analyze text for excessive repetition."""
        result = {
            'excessive_repetition': False,
            'repetition_ratio': 0.0,
            'issues': [],
            'metrics': {}
        }
        
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return result
        
        # Calculate repetition ratio
        word_counts = Counter(words)
        most_common_word, max_count = word_counts.most_common(1)[0]
        repetition_ratio = max_count / len(words)
        
        result['repetition_ratio'] = repetition_ratio
        result['metrics']['most_repeated_word'] = most_common_word
        result['metrics']['max_word_count'] = max_count
        result['metrics']['repetition_ratio'] = repetition_ratio
        
        if repetition_ratio > self.max_repetition_ratio:
            result['excessive_repetition'] = True
            result['issues'].append(f"Excessive repetition: '{most_common_word}' appears {max_count}/{len(words)} times ({repetition_ratio:.1%})")
        
        # Check for repeated phrases
        phrases = []
        words_list = words
        for i in range(len(words_list) - 2):
            phrase = ' '.join(words_list[i:i+3])
            phrases.append(phrase)
        
        if phrases:
            phrase_counts = Counter(phrases)
            most_common_phrase, phrase_max_count = phrase_counts.most_common(1)[0]
            phrase_repetition = phrase_max_count / len(phrases)
            
            result['metrics']['most_repeated_phrase'] = most_common_phrase
            result['metrics']['phrase_repetition_ratio'] = phrase_repetition
            
            if phrase_repetition > 0.5:  # More than half the phrases are the same
                result['excessive_repetition'] = True
                result['issues'].append(f"Repeated phrase: '{most_common_phrase}' appears {phrase_max_count} times")
        
        return result

    def should_skip_classification(self, text: str) -> Tuple[bool, str]:
        """
        Quick check to determine if document should be skipped entirely.
        
        Args:
            text: Document text
            
        Returns:
            Tuple of (should_skip: bool, reason: str)
        """
        quality_result = self.check_document_quality(text)
        
        if not quality_result['is_quality']:
            reason = f"Document quality too low: {', '.join(quality_result['issues'][:2])}"
            return True, reason
        
        return False, ""