#!/usr/bin/env python3
"""
Debug correspondence pattern matching
"""
import sys
sys.path.append('.')
import re

from classifier.pdf_extractor import PDFExtractor

def debug_patterns():
    """Debug why patterns aren't matching"""
    
    pdf_path = "data/Lot-11/20200821_AE_PD_392 - Provision of Toll Plaza at Ch. Km. 104+917 (Chennasamudram Village).pdf"
    
    # Extract text
    extractor = PDFExtractor(max_pages=2, ocr_threshold=50)
    raw_text, method = extractor.extract_text(pdf_path)
    
    print(f"🧪 Debugging Pattern Matching")
    print("=" * 50)
    print(f"📊 Raw text: {len(raw_text)} chars")
    
    # Test subject patterns directly
    subject_patterns = [
        r'(?i)subject\\s*[:]\\s*(.+?)(?=\\n|$)',
        r'(?i)sub\\s*[:]\\s*(.+?)(?=\\n|$)', 
    ]
    
    print(f"\n🔍 Testing subject patterns on raw text:")
    
    for i, pattern in enumerate(subject_patterns, 1):
        match = re.search(pattern, raw_text, re.MULTILINE | re.DOTALL)
        if match:
            print(f"✅ Pattern {i} MATCHED: '{match.group(1)[:100]}...'")
        else:
            print(f"❌ Pattern {i} failed")
    
    # Look for "Subject:" manually
    lines = raw_text.split('\n')
    print(f"\n📝 Looking for 'Subject' in {len(lines)} lines:")
    
    for i, line in enumerate(lines):
        if 'subject' in line.lower():
            print(f"  Line {i}: '{line.strip()}'")

if __name__ == "__main__":
    debug_patterns()