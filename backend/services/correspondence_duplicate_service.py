#!/usr/bin/env python3
"""
Correspondence Duplicate Detection Service
Detects and manages duplicate correspondence letters based on multiple criteria
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, date
from dataclasses import dataclass
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

@dataclass
class DuplicateMatch:
    """Represents a potential duplicate match"""
    original_id: str
    original_document_id: str
    similarity_score: float
    matching_criteria: Dict[str, Any]
    detection_method: str
    created_at: datetime

@dataclass
class DuplicateCheckResult:
    """Result of duplicate detection check"""
    has_duplicates: bool
    duplicate_count: int
    duplicates: List[DuplicateMatch]
    highest_similarity: float
    error: Optional[str] = None

class CorrespondenceDuplicateService:
    """
    Service for detecting duplicate correspondence letters
    """
    
    def __init__(self):
        """Initialize the duplicate detection service"""
        self.similarity_threshold = 0.75  # 75% similarity threshold
        self.exact_match_threshold = 0.98  # 98% for near-exact matches
        
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings using SequenceMatcher
        Returns a value between 0.0 and 1.0
        """
        if not text1 or not text2:
            return 0.0
            
        # Normalize texts (lowercase, strip whitespace)
        text1_norm = text1.lower().strip()
        text2_norm = text2.lower().strip()
        
        if text1_norm == text2_norm:
            return 1.0
            
        # Use SequenceMatcher for fuzzy matching
        matcher = SequenceMatcher(None, text1_norm, text2_norm)
        return matcher.ratio()
    
    def extract_key_words(self, text: str, min_length: int = 4) -> set:
        """Extract key words from text for comparison"""
        if not text:
            return set()
            
        # Simple word extraction (can be enhanced with NLP libraries)
        words = text.lower().split()
        # Filter out short words and common stop words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should'}
        
        return {word.strip('.,!?;:"()[]{}') for word in words 
                if len(word) >= min_length and word not in stop_words}
    
    def calculate_word_overlap_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity based on word overlap"""
        words1 = self.extract_key_words(text1)
        words2 = self.extract_key_words(text2)
        
        if not words1 or not words2:
            return 0.0
            
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_comprehensive_similarity(self, 
                                        letter_id1: str, letter_id2: str,
                                        date1: Optional[date], date2: Optional[date],
                                        subject1: str, subject2: str,
                                        body1: str, body2: str) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate comprehensive similarity score based on multiple criteria
        Returns (similarity_score, matching_criteria_dict)
        """
        
        similarity_score = 0.0
        criteria = {}
        
        # 1. Letter ID exact match (40 points)
        if letter_id1 and letter_id2 and letter_id1.strip() == letter_id2.strip():
            similarity_score += 40.0
            criteria['letter_id_match'] = True
            criteria['letter_id_similarity'] = 1.0
        elif letter_id1 and letter_id2:
            letter_id_sim = self.calculate_text_similarity(letter_id1, letter_id2)
            if letter_id_sim > 0.8:  # High similarity in letter IDs
                similarity_score += letter_id_sim * 40.0
                criteria['letter_id_similarity'] = letter_id_sim
        
        # 2. Date match (20 points)
        if date1 and date2:
            if date1 == date2:
                similarity_score += 20.0
                criteria['date_match'] = True
            else:
                # Check if dates are close (within a few days)
                date_diff = abs((date1 - date2).days)
                if date_diff <= 3:  # Within 3 days
                    date_similarity = max(0, 1.0 - (date_diff / 7.0))  # Decay over a week
                    similarity_score += date_similarity * 20.0
                    criteria['date_similarity'] = date_similarity
        
        # 3. Subject similarity (25 points max)
        if subject1 and subject2:
            # Use both sequence matching and word overlap
            seq_similarity = self.calculate_text_similarity(subject1, subject2)
            word_similarity = self.calculate_word_overlap_similarity(subject1, subject2)
            
            # Take the higher of the two approaches
            subject_similarity = max(seq_similarity, word_similarity)
            similarity_score += subject_similarity * 25.0
            criteria['subject_similarity'] = subject_similarity
            criteria['subject_sequence_match'] = seq_similarity
            criteria['subject_word_overlap'] = word_similarity
        
        # 4. Body content similarity (15 points max)
        if body1 and body2:
            # For long texts, word overlap might be more accurate than sequence matching
            seq_similarity = self.calculate_text_similarity(body1, body2)
            word_similarity = self.calculate_word_overlap_similarity(body1, body2)
            
            body_similarity = max(seq_similarity, word_similarity)
            similarity_score += body_similarity * 15.0
            criteria['body_similarity'] = body_similarity
            criteria['body_sequence_match'] = seq_similarity
            criteria['body_word_overlap'] = word_similarity
        
        return similarity_score, criteria
    
    def check_for_duplicates(self, 
                           new_letter_data: Dict[str, Any], 
                           existing_letters: List[Dict[str, Any]]) -> DuplicateCheckResult:
        """
        Check if the new letter is a duplicate of any existing letters
        
        Args:
            new_letter_data: Dictionary containing the new letter's data
            existing_letters: List of existing letters in the same project
            
        Returns:
            DuplicateCheckResult object
        """
        
        try:
            duplicates = []
            highest_similarity = 0.0
            
            new_letter_id = new_letter_data.get('letter_id', '')
            new_date = new_letter_data.get('date_sent')
            new_subject = new_letter_data.get('subject', '')
            new_body = new_letter_data.get('body', '')
            
            # Convert string date to date object if needed
            if isinstance(new_date, str):
                try:
                    new_date = datetime.strptime(new_date, '%Y-%m-%d').date()
                except:
                    new_date = None
            
            for existing_letter in existing_letters:
                existing_id = existing_letter.get('id')
                existing_doc_id = existing_letter.get('document_id')
                existing_letter_id = existing_letter.get('letter_id', '')
                existing_date = existing_letter.get('date_sent')
                existing_subject = existing_letter.get('subject', '')
                existing_body = existing_letter.get('body', '')
                existing_created_at = existing_letter.get('created_at')
                
                # Convert string date to date object if needed
                if isinstance(existing_date, str):
                    try:
                        existing_date = datetime.strptime(existing_date, '%Y-%m-%d').date()
                    except:
                        existing_date = None
                
                # Calculate similarity
                similarity_score, criteria = self.calculate_comprehensive_similarity(
                    new_letter_id, existing_letter_id,
                    new_date, existing_date,
                    new_subject, existing_subject,
                    new_body, existing_body
                )
                
                # Determine detection method
                detection_method = 'similarity_algorithm'
                if criteria.get('letter_id_match'):
                    detection_method = 'letter_id_match'
                elif similarity_score >= 98.0:
                    detection_method = 'high_similarity_match'
                
                # Consider as duplicate if above threshold
                if similarity_score >= (self.similarity_threshold * 100) or criteria.get('letter_id_match'):
                    
                    # Parse created_at if it's a string
                    created_at = existing_created_at
                    if isinstance(created_at, str):
                        try:
                            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        except:
                            created_at = datetime.now()
                    
                    duplicate_match = DuplicateMatch(
                        original_id=existing_id,
                        original_document_id=existing_doc_id,
                        similarity_score=similarity_score,
                        matching_criteria=criteria,
                        detection_method=detection_method,
                        created_at=created_at
                    )
                    
                    duplicates.append(duplicate_match)
                    highest_similarity = max(highest_similarity, similarity_score)
            
            return DuplicateCheckResult(
                has_duplicates=len(duplicates) > 0,
                duplicate_count=len(duplicates),
                duplicates=duplicates,
                highest_similarity=highest_similarity
            )
            
        except Exception as e:
            logger.error(f"Error in duplicate detection: {str(e)}")
            return DuplicateCheckResult(
                has_duplicates=False,
                duplicate_count=0,
                duplicates=[],
                highest_similarity=0.0,
                error=str(e)
            )
    
    def format_duplicate_result_for_database(self, result: DuplicateCheckResult) -> Dict[str, Any]:
        """Format duplicate check result for database storage"""
        
        duplicates_json = []
        for duplicate in result.duplicates:
            duplicates_json.append({
                'original_id': duplicate.original_id,
                'original_document_id': duplicate.original_document_id,
                'similarity_score': duplicate.similarity_score,
                'matching_criteria': duplicate.matching_criteria,
                'detection_method': duplicate.detection_method,
                'created_at': duplicate.created_at.isoformat() if duplicate.created_at else None
            })
        
        return {
            'has_duplicates': result.has_duplicates,
            'duplicate_count': result.duplicate_count,
            'duplicates': duplicates_json,
            'highest_similarity': result.highest_similarity,
            'error': result.error
        }

# Example usage and testing
if __name__ == "__main__":
    # Test the duplicate detection service
    service = CorrespondenceDuplicateService()
    
    # Sample data for testing
    new_letter = {
        'letter_id': 'LTR/2024/001',
        'date_sent': '2024-01-15',
        'subject': 'Request for Extension of Project Timeline',
        'body': 'We hereby request an extension of the project timeline due to unforeseen circumstances...'
    }
    
    existing_letters = [
        {
            'id': 'uuid-1',
            'document_id': 'doc-uuid-1',
            'letter_id': 'LTR/2024/001',  # Same letter ID
            'date_sent': '2024-01-15',
            'subject': 'Request for Extension of Project Timeline',
            'body': 'We hereby request an extension of the project timeline due to unforeseen circumstances...',
            'created_at': '2024-01-15T10:00:00Z'
        }
    ]
    
    result = service.check_for_duplicates(new_letter, existing_letters)
    print(f"Duplicate check result: {result}")
    print(f"Formatted for DB: {service.format_duplicate_result_for_database(result)}")