#!/usr/bin/env python3
"""
Test correspondence extraction on OCR text
"""
import sys
sys.path.append('.')

from classifier.pdf_extractor import PDFExtractor
from extract_correspondence_content import CorrespondenceExtractor
import logging

logging.basicConfig(level=logging.INFO)

def test_correspondence_extraction():
    """Test correspondence extraction on OCR text"""
    
    pdf_path = "data/Lot-11/20200821_AE_PD_392 - Provision of Toll Plaza at Ch. Km. 104+917 (Chennasamudram Village).pdf"
    
    print(f"🧪 Testing Correspondence Extraction")
    print("=" * 50)
    
    # Extract text
    extractor = PDFExtractor(max_pages=2, ocr_threshold=50)
    raw_text, method = extractor.extract_text(pdf_path)
    
    print(f"📊 Raw extraction: {len(raw_text)} chars using {method}")
    
    # Show sample text
    print(f"\n📝 Raw text sample (first 500 chars):")
    print(raw_text[:500])
    print("..." if len(raw_text) > 500 else "")
    
    # Test correspondence extraction
    correspondence_extractor = CorrespondenceExtractor()
    result = correspondence_extractor.extract_correspondence_content(raw_text)
    
    print(f"\n🔍 Correspondence Extraction Results:")
    print(f"📧 Subject: '{result['subject']}'")
    print(f"📄 Body length: {len(result['body'])} chars")
    print(f"🎯 Method: {result['extraction_method']}")
    
    # Show body sample
    if result['body']:
        print(f"\n📝 Body sample (first 300 chars):")
        print(result['body'][:300])
        print("..." if len(result['body']) > 300 else "")
    
    # Test focused content
    focused_content = correspondence_extractor.get_focused_content(raw_text)
    print(f"\n🎯 Focused content: {len(focused_content)} chars")
    
    if focused_content:
        print(f"📝 Focused content sample:")
        print(focused_content[:400])
        print("..." if len(focused_content) > 400 else "")

if __name__ == "__main__":
    test_correspondence_extraction()