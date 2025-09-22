#!/usr/bin/env python3
"""
Test PDF extraction specifically for the problematic first PDF
"""
import sys
sys.path.append('.')

from classifier.pdf_extractor import PDFExtractor
from extract_correspondence_content import CorrespondenceExtractor
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

def test_specific_pdf():
    """Test extraction on the first PDF that had issues"""
    
    pdf_path = "data/Lot-11/20200821_AE_PD_392 - Provision of Toll Plaza at Ch. Km. 104+917 (Chennasamudram Village).pdf"
    
    print("🧪 Testing PDF Extraction on Problematic File")
    print("=" * 60)
    print(f"📄 File: {pdf_path}")
    
    # Test with different max_pages settings
    for max_pages in [2, 5, None]:
        print(f"\n🔍 Testing with max_pages={max_pages}")
        
        extractor = PDFExtractor(max_pages=max_pages)
        raw_text, method = extractor.extract_text(pdf_path)
        
        print(f"📊 Raw extraction: {len(raw_text)} chars using {method}")
        
        if raw_text:
            print(f"📝 First 200 chars: {raw_text[:200]}")
            print(f"📝 Last 200 chars: {raw_text[-200:]}")
        
        # Test correspondence extraction
        correspondence_extractor = CorrespondenceExtractor()
        result = correspondence_extractor.extract_correspondence_content(raw_text)
        focused_content = correspondence_extractor.get_focused_content(raw_text)
        
        print(f"📧 Subject found: '{result['subject']}'")
        print(f"📄 Body length: {len(result['body'])} chars")
        print(f"🎯 Focused content: {len(focused_content)} chars")
        print(f"🔍 Method: {result['extraction_method']}")
        
        if focused_content and len(focused_content) < 500:
            print(f"🔎 Full focused content: {focused_content}")

if __name__ == "__main__":
    test_specific_pdf()