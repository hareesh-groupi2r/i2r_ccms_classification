#!/usr/bin/env python3
"""
Enhanced Correspondence Content Extraction
Focuses on extracting Subject and Body content from correspondence documents
"""

import re
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class CorrespondenceExtractor:
    """
    Extracts relevant content from correspondence documents (letters, emails, memos)
    """
    
    def __init__(self):
        # Common subject line patterns - enhanced to handle various formats including OCR text
        self.subject_patterns = [
            # OCR-friendly patterns that handle continuous text (no line breaks)
            r'(?i)subject\s*[:]\s*(.{10,200}?)(?=\s+(?:ref|reference|dear\s+sir|sir\s*,|to\s+whom|\-\s*reg|dated\s*:|letter\s+no))',
            r'(?i)sub\s*[:]\s*(.{10,200}?)(?=\s+(?:ref|reference|dear\s+sir|sir\s*,|to\s+whom|\-\s*reg|dated\s*:|letter\s+no))',
            
            # Multi-line subjects that continue until reference or greeting  
            r'(?i)sub\s*[:]\s*(.+?)(?=\n\s*ref\s*[:\.]|\n\s*dear\s+sir)',
            r'(?i)subject\s*[:]\s*(.+?)(?=\n\s*ref\s*[:\.]|\n\s*dear\s+sir)',
            
            # Standard formats with colon - single line
            r'(?i)subject\s*[:]\s*(.+?)(?=\n|$)',
            r'(?i)sub\s*[:]\s*(.+?)(?=\n|$)', 
            r'(?i)re\s*[:]\s*(.+?)(?=\n|$)',
            r'(?i)regarding\s*[:]\s*(.+?)(?=\n|$)',
            
            # Formats with period
            r'(?i)subject\s*[.]\s*(.+?)(?=\n|$)',
            r'(?i)sub\s*[.]\s*(.+?)(?=\n|$)',
            r'(?i)re\s*[.]\s*(.+?)(?=\n|$)',
            
            # Formats without punctuation (with whitespace) 
            r'(?i)subject\s+(.+?)(?=\n|$)',
            r'(?i)sub\s+(.+?)(?=\n|$)',
            r'(?i)re\s+(.+?)(?=\n|$)',
        ]
        
        # Patterns to identify letter body start (after salutation)
        self.body_start_patterns = [
            r'(?i)dear\s+(?:sir|madam|mr|ms|dr|prof).*?,?\s*',
            r'(?i)to\s+whom\s+it\s+may\s+concern.*?,?\s*',
            r'(?i)respected\s+(?:sir|madam).*?,?\s*',
            r'(?i)sir\s*,?\s*',
            r'(?i)madam\s*,?\s*',
            # OCR-friendly patterns (may not have perfect line breaks)
            r'(?i)dear\s+sir\s*,?',
            r'(?i)dear\s+madam\s*,?',
            r'(?i)sir\s*,',
        ]
        
        # Patterns to identify letter body end (signatures, regards, etc.)
        self.body_end_patterns = [
            r'(?i)(?:yours?\s+(?:sincerely|faithfully|truly)|sincerely|regards?|best\s+regards?)\s*,?\s*\n',
            r'(?i)(?:thanking\s+you|thank\s+you)\s*,?\s*\n',
            r'(?i)signature\s*:',
            r'(?i)(?:name|designation)\s*:',
            r'(?i)(?:cc|copy\s+to)\s*:',
            r'(?i)enclosure\s*:',
            r'(?i)attachment\s*:',
        ]
        
        # Header/Footer noise patterns to remove
        self.noise_patterns = [
            r'(?i)page\s+\d+\s+of\s+\d+',
            r'(?i)confidential',
            r'(?i)printed\s+on\s+\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
            r'(?i)copyright\s+\d{4}',
            r'(?i)all\s+rights\s+reserved',
            r'(?i)this\s+document\s+is\s+confidential',
            r'\*{3,}|\-{3,}|_{3,}|={3,}',  # Lines of repeated characters
        ]
    
    def extract_correspondence_content(self, text: str) -> Dict[str, str]:
        """
        Extract subject and body from correspondence text
        
        Args:
            text: Raw extracted text from PDF
            
        Returns:
            Dict with 'subject', 'body', 'full_text', and 'extraction_method'
        """
        
        if not text or len(text.strip()) < 50:
            return {
                'subject': '',
                'body': text.strip(),
                'full_text': text,
                'extraction_method': 'insufficient_content'
            }
        
        # Clean the text first
        cleaned_text = self._clean_text(text)
        
        # Extract subject
        subject = self._extract_subject(cleaned_text)
        
        # Extract body
        body = self._extract_body(cleaned_text)
        
        # If body extraction failed, use cleaned full text
        if not body or len(body.strip()) < 50:
            body = cleaned_text
            extraction_method = 'full_text_fallback'
        else:
            extraction_method = 'structured_extraction'
        
        return {
            'subject': subject.strip(),
            'body': body.strip(),
            'full_text': cleaned_text,
            'extraction_method': extraction_method
        }
    
    def _clean_text(self, text: str) -> str:
        """Remove headers, footers, and other noise - OCR-friendly version"""
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # For OCR text, be less aggressive - only skip obvious noise
            is_noise = False
            for pattern in self.noise_patterns:
                if re.search(pattern, line):
                    is_noise = True
                    break
            
            # Skip very short lines that are likely OCR artifacts (but keep "Page X:" markers)
            if len(line) < 5 and not line.startswith('Page'):
                is_noise = True
            
            if not is_noise:
                cleaned_lines.append(line)
        
        # For OCR text, if we have very few lines, return original text
        # This handles cases where OCR produces one long line
        if len(cleaned_lines) <= 3 and len(text.strip()) > 200:
            return text.strip()
        
        return '\n'.join(cleaned_lines)
    
    def _extract_subject(self, text: str) -> str:
        """Extract subject line from text"""
        
        for pattern in self.subject_patterns:
            match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
            if match:
                subject = match.group(1).strip()
                # Clean up the subject (remove extra whitespace, line breaks)
                subject = ' '.join(subject.split())
                
                # Additional cleaning for multi-line subjects
                # Remove common footer patterns that might have been included
                clean_patterns = [
                    r'\s*-\s*reg\.?$',  # Remove "- Reg" at end
                    r'\s*received\s+on\s+\d{2}\.\d{2}\.\d{4}.*$',  # Remove received date
                    r'\s*dated\s*[:]\s*\d{2}\.\d{2}\.\d{4}.*$',  # Remove date references
                ]
                
                for clean_pattern in clean_patterns:
                    subject = re.sub(clean_pattern, '', subject, flags=re.IGNORECASE)
                
                # Limit subject length to reasonable size
                if len(subject) > 200:
                    subject = subject[:200] + "..."
                
                logger.info(f"Found subject using pattern: {pattern[:20]}...")
                return subject.strip()
        
        # Fallback: Look for lines that might be subjects (all caps, specific keywords)
        lines = text.split('\n')
        for i, line in enumerate(lines[:10]):  # Check first 10 lines
            line = line.strip()
            if len(line) > 10 and len(line) < 200:  # Reasonable subject length
                # Check if line contains correspondence keywords
                if any(word in line.lower() for word in [
                    'proposal', 'request', 'submission', 'approval', 'change', 'scope',
                    'payment', 'extension', 'time', 'completion', 'construction',
                    'clarification', 'compliance', 'observation'
                ]):
                    logger.info("Found subject using keyword matching")
                    return line
        
        return ''
    
    def _extract_body(self, text: str) -> str:
        """Extract body content from correspondence - starts after salutation"""
        
        # First, try to find content from subject onwards (skip headers)
        content_from_subject = self._extract_from_subject_onwards(text)
        
        # Find body start (after salutation) in the subject-onwards content
        body_start = 0
        for pattern in self.body_start_patterns:
            match = re.search(pattern, content_from_subject, re.MULTILINE | re.DOTALL)
            if match:
                body_start = match.end()
                logger.info(f"Found body start using pattern: {pattern[:20]}...")
                break
        
        # If no salutation found, try to find where actual content starts after subject
        if body_start == 0:
            # Look for patterns that indicate start of actual letter content
            content_start_patterns = [
                r'(?i)(?:sir|madam)\s*,?\s*',  # Just sir/madam
                r'(?i)it\s+is\s+(?:to\s+)?(?:inform|state|mention)',  # "It is to inform..."
                r'(?i)(?:we|i)\s+(?:are\s+)?(?:writing|submitting|requesting)',  # "We are writing..."
                r'(?i)(?:this|the)\s+is\s+(?:to|with)\s+(?:regard|reference)',  # "This is to regard..."
                r'(?i)(?:please|kindly)\s+(?:find|note)',  # "Please find..."
            ]
            
            for pattern in content_start_patterns:
                match = re.search(pattern, content_from_subject, re.MULTILINE | re.DOTALL)
                if match:
                    body_start = match.start()
                    logger.info(f"Found content start using pattern: {pattern[:30]}...")
                    break
        
        # Extract text from body start position
        if body_start > 0:
            body_text = content_from_subject[body_start:].strip()
        else:
            # If no specific body start found, use content from subject onwards
            body_text = content_from_subject.strip()
            logger.info("Using content from subject onwards (no salutation found)")
        
        # Find body end (before signature/closing)
        body_end = len(body_text)
        for pattern in self.body_end_patterns:
            match = re.search(pattern, body_text, re.MULTILINE)
            if match:
                body_end = match.start()
                logger.info(f"Found body end using pattern: {pattern[:20]}...")
                break
        
        # Extract final body content
        body_content = body_text[:body_end].strip()
        
        return body_content
    
    def get_focused_content(self, text: str) -> str:
        """
        Get the most relevant content for classification (subject + body)
        Ignores everything before the subject line to remove headers
        
        Args:
            text: Raw extracted text
            
        Returns:
            Focused content string for classification
        """
        
        extraction_result = self.extract_correspondence_content(text)
        
        subject = extraction_result['subject']
        body = extraction_result['body']
        
        # Combine subject and body with clear separation
        if subject and body:
            focused_content = f"Subject: {subject}\n\nContent: {body}"
        elif subject:
            focused_content = f"Subject: {subject}"
        elif body:
            focused_content = body
        else:
            # Fallback: try to find subject in text and use everything from there
            focused_content = self._extract_from_subject_onwards(text)
            if not focused_content:
                focused_content = extraction_result['full_text']
        
        logger.info(f"Focused content extraction: {len(focused_content)} chars from {len(text)} chars ({extraction_result['extraction_method']})")
        
        return focused_content
    
    def _extract_from_subject_onwards(self, text: str) -> str:
        """
        Extract content starting from the subject line, ignoring headers before it
        
        Args:
            text: Raw text
            
        Returns:
            Text from subject line onwards
        """
        # Find where subject starts
        for pattern in self.subject_patterns[:4]:  # Use first 4 patterns (most reliable)
            match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
            if match:
                # Get position where subject starts
                subject_start = match.start()
                # Return everything from subject onwards
                content_from_subject = text[subject_start:]
                logger.info(f"Found subject start at position {subject_start}, extracted {len(content_from_subject)} chars from subject onwards")
                return content_from_subject
        
        # If no subject found, return original text
        return text

def test_correspondence_extraction():
    """Test the correspondence extraction with sample text"""
    
    sample_text = """
    Page 1 of 2
    CONFIDENTIAL
    
    Letter No: AE/PD/392/2020
    Date: 21/08/2020
    
    Subject: Provision of Toll Plaza at Ch. Km. 104+917 (Chennasamudram Village)
    
    Dear Sir,
    
    We are writing to submit our revised proposal for the construction of toll plaza at the specified location. The approximate cost with the new 8+8 lane provisions is worked out to 18,14,66,425 and there is a savings of 5.68 Crores.
    
    This change in scope is necessitated due to the updated traffic projections and the need for enhanced capacity. We request your approval for proceeding with this revised proposal.
    
    Please find attached the detailed cost breakdown and technical specifications for your review.
    
    Thanking you,
    
    Yours faithfully,
    
    Project Manager
    Construction Division
    
    CC: Regional Manager
    """
    
    extractor = CorrespondenceExtractor()
    result = extractor.extract_correspondence_content(sample_text)
    
    print("🧪 Testing Correspondence Extraction")
    print("=" * 50)
    print(f"📧 Subject: {result['subject']}")
    print(f"📝 Body Length: {len(result['body'])} chars")
    print(f"🔍 Method: {result['extraction_method']}")
    print("\n📄 Extracted Body:")
    print(result['body'][:300] + "..." if len(result['body']) > 300 else result['body'])
    
    print("\n🎯 Focused Content:")
    focused = extractor.get_focused_content(sample_text)
    print(focused[:400] + "..." if len(focused) > 400 else focused)

if __name__ == "__main__":
    test_correspondence_extraction()