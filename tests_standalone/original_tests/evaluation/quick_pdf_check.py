#!/usr/bin/env python3
"""
Quick check of PDF extraction without debug logging
"""
import sys
sys.path.append('.')

from classifier.pdf_extractor import PDFExtractor
import logging

# Minimal logging
logging.basicConfig(level=logging.WARNING)

def quick_check():
    """Quick check of problematic PDF"""
    
    pdf_path = "data/Lot-11/20200821_AE_PD_392 - Provision of Toll Plaza at Ch. Km. 104+917 (Chennasamudram Village).pdf"
    
    print(f"🧪 Quick PDF Check: {pdf_path.split('/')[-1]}")
    
    # Test with 2 pages only
    extractor = PDFExtractor(max_pages=2, ocr_threshold=50)
    
    # Check if it's a scanned PDF
    is_scanned = extractor.is_scanned_pdf(pdf_path)
    print(f"📄 Is scanned PDF: {is_scanned}")
    
    # Try extraction
    try:
        raw_text, method = extractor.extract_text(pdf_path)
        print(f"📊 Extracted: {len(raw_text)} chars using {method}")
        
        if raw_text and len(raw_text) > 0:
            # Show first few lines
            lines = raw_text.split('\n')[:10]
            print("📝 First 10 lines:")
            for i, line in enumerate(lines, 1):
                if line.strip():
                    print(f"  {i}: {line.strip()[:60]}...")
        else:
            print("❌ No text extracted")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    quick_check()