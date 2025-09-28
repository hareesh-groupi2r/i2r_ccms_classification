#!/usr/bin/env python3
"""
Unit tests for PDF extraction functionality
"""

import unittest
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from classifier.pdf_extractor import PDFExtractor

# Suppress logging during tests
logging.getLogger().setLevel(logging.WARNING)


class TestPDFExtractor(unittest.TestCase):
    """Test cases for PDF extraction functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = PDFExtractor(max_pages=2)
        self.sample_pdf_path = Path("data/Lot-11/20200821_AE_PD_392 - Provision of Toll Plaza at Ch. Km. 104+917 (Chennasamudram Village).pdf")
    
    def test_extractor_initialization(self):
        """Test PDF extractor initialization."""
        # Test default initialization
        default_extractor = PDFExtractor()
        self.assertIsNone(default_extractor.max_pages)
        self.assertEqual(default_extractor.dpi, 300)
        self.assertEqual(default_extractor.ocr_threshold, 50)
        
        # Test custom initialization  
        custom_extractor = PDFExtractor(max_pages=5, dpi=200, ocr_threshold=100)
        self.assertEqual(custom_extractor.max_pages, 5)
        self.assertEqual(custom_extractor.dpi, 200)
        self.assertEqual(custom_extractor.ocr_threshold, 100)
    
    def test_file_not_found_handling(self):
        """Test handling of non-existent files."""
        non_existent_path = Path("non_existent_file.pdf")
        
        with self.assertRaises(FileNotFoundError):
            self.extractor.extract_text(non_existent_path)
    
    @patch('classifier.pdf_extractor.pdfplumber')
    def test_hybrid_extraction_text_only(self, mock_pdfplumber):
        """Test hybrid extraction when text extraction is sufficient."""
        # Mock PDF with good text extraction
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "This is sufficient text content for extraction" + "x" * 100
        mock_pdf.pages = [mock_page, mock_page]  # 2 pages
        mock_pdf.__enter__.return_value = mock_pdf
        mock_pdf.__exit__.return_value = None
        mock_pdfplumber.open.return_value = mock_pdf
        
        text, method = self.extractor._extract_hybrid(Path("test.pdf"))
        
        self.assertIn("This is sufficient text", text)
        self.assertIn("text_only", method)
        self.assertIn("2pages", method)
    
    @patch('classifier.pdf_extractor.pdfplumber')
    @patch('classifier.pdf_extractor.PDFExtractor._extract_page_ocr')
    def test_hybrid_extraction_ocr_fallback(self, mock_ocr, mock_pdfplumber):
        """Test hybrid extraction with OCR fallback."""
        # Mock PDF with insufficient text (requires OCR)
        mock_pdf = MagicMock()
        mock_page = MagicMock() 
        mock_page.extract_text.return_value = "Short"  # Insufficient text
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__.return_value = mock_pdf
        mock_pdf.__exit__.return_value = None
        mock_pdfplumber.open.return_value = mock_pdf
        
        # Mock OCR to return good text
        mock_ocr.return_value = "This is OCR extracted text content" + "x" * 100
        
        text, method = self.extractor._extract_hybrid(Path("test.pdf"))
        
        self.assertIn("OCR extracted text", text)
        self.assertIn("ocr_only", method)
        mock_ocr.assert_called()
    
    @patch('classifier.pdf_extractor.pdfplumber')
    @patch('classifier.pdf_extractor.PDFExtractor._extract_page_ocr')
    def test_hybrid_extraction_mixed(self, mock_ocr, mock_pdfplumber):
        """Test hybrid extraction with mixed text and OCR pages."""
        # Mock PDF with one good text page and one OCR page
        mock_pdf = MagicMock()
        good_page = MagicMock()
        good_page.extract_text.return_value = "Good text content" + "x" * 100
        bad_page = MagicMock()
        bad_page.extract_text.return_value = "Bad"  # Needs OCR
        
        mock_pdf.pages = [good_page, bad_page]
        mock_pdf.__enter__.return_value = mock_pdf
        mock_pdf.__exit__.return_value = None
        mock_pdfplumber.open.return_value = mock_pdf
        
        # Mock OCR for the bad page
        mock_ocr.return_value = "OCR text content" + "x" * 50
        
        text, method = self.extractor._extract_hybrid(Path("test.pdf"))
        
        self.assertIn("Good text content", text)
        self.assertIn("OCR text content", text)
        self.assertIn("hybrid_text+ocr", method)
        self.assertIn("2pages", method)
    
    def test_max_pages_enforcement(self):
        """Test that max_pages limit is enforced."""
        extractor_with_limit = PDFExtractor(max_pages=1)
        
        with patch('classifier.pdf_extractor.pdfplumber') as mock_pdfplumber:
            # Mock PDF with 3 pages
            mock_pdf = MagicMock()
            mock_pages = []
            for i in range(3):
                page = MagicMock()
                page.extract_text.return_value = f"Page {i+1} content" + "x" * 100
                mock_pages.append(page)
            
            mock_pdf.pages = mock_pages
            mock_pdf.__enter__.return_value = mock_pdf
            mock_pdf.__exit__.return_value = None
            mock_pdfplumber.open.return_value = mock_pdf
            
            text, method = extractor_with_limit._extract_hybrid(Path("test.pdf"))
            
            # Should only process first page
            self.assertIn("Page 1 content", text)
            self.assertNotIn("Page 2 content", text)
            self.assertNotIn("Page 3 content", text)
            self.assertIn("1pages", method)
    
    @patch('classifier.pdf_extractor.convert_from_path')
    @patch('classifier.pdf_extractor.pytesseract')
    def test_page_ocr_extraction(self, mock_tesseract, mock_convert):
        """Test OCR extraction for individual pages."""
        # Mock PDF to image conversion
        mock_image = MagicMock()
        mock_convert.return_value = [mock_image]
        
        # Mock OCR text extraction
        mock_tesseract.image_to_string.return_value = "OCR extracted text content"
        
        ocr_text = self.extractor._extract_page_ocr(Path("test.pdf"), 1)
        
        self.assertEqual(ocr_text, "OCR extracted text content")
        mock_convert.assert_called_once()
        mock_tesseract.image_to_string.assert_called_once()
    
    @patch('classifier.pdf_extractor.convert_from_path')
    def test_page_ocr_failure(self, mock_convert):
        """Test OCR extraction failure handling."""
        # Mock conversion failure
        mock_convert.side_effect = Exception("OCR failed")
        
        ocr_text = self.extractor._extract_page_ocr(Path("test.pdf"), 1)
        
        self.assertEqual(ocr_text, "")  # Should return empty string on failure
    
    def test_clean_ocr_text(self):
        """Test OCR text cleaning functionality."""
        dirty_ocr_text = """
        This is some text with
        
        extra whitespace
        and   multiple    spaces
        
        and empty lines.
        """
        
        cleaned_text = self.extractor._clean_ocr_text(dirty_ocr_text)
        
        self.assertNotIn("\n\n\n", cleaned_text)  # No multiple newlines
        self.assertIn("This is some text", cleaned_text)
        self.assertIn("extra whitespace", cleaned_text)
        # Should normalize multiple spaces to single spaces
        self.assertNotIn("   ", cleaned_text)
    
    @unittest.skipUnless(Path("data/Lot-11").exists(), "Test data not available")
    def test_real_pdf_extraction(self):
        """Integration test with real PDF file if available."""
        if self.sample_pdf_path.exists():
            text, method = self.extractor.extract_text(self.sample_pdf_path)
            
            self.assertIsInstance(text, str)
            self.assertGreater(len(text), 100)  # Should extract meaningful text
            self.assertIsInstance(method, str)
            self.assertTrue(method in ['text_only', 'ocr_only', 'hybrid_text+ocr'] or 
                          'pages' in method)
        else:
            self.skipTest("Sample PDF not available")
    
    def test_extraction_method_reporting(self):
        """Test that extraction methods are reported correctly."""
        with patch('classifier.pdf_extractor.pdfplumber') as mock_pdfplumber:
            mock_pdf = MagicMock()
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "Good text" + "x" * 100
            mock_pdf.pages = [mock_page, mock_page, mock_page]  # 3 pages
            mock_pdf.__enter__.return_value = mock_pdf
            mock_pdf.__exit__.return_value = None
            mock_pdfplumber.open.return_value = mock_pdf
            
            extractor = PDFExtractor(max_pages=2)  # Limit to 2 pages
            text, method = extractor._extract_hybrid(Path("test.pdf"))
            
            # Should indicate text extraction and page count
            self.assertIn("text_only", method)
            self.assertIn("2pages", method)  # Should respect max_pages
    
    def test_insufficient_content_handling(self):
        """Test handling when no meaningful content is extracted."""
        with patch('classifier.pdf_extractor.pdfplumber') as mock_pdfplumber:
            with patch.object(self.extractor, '_extract_page_ocr') as mock_ocr:
                # Mock PDF with no text and failed OCR
                mock_pdf = MagicMock()
                mock_page = MagicMock()
                mock_page.extract_text.return_value = ""  # No text
                mock_pdf.pages = [mock_page]
                mock_pdf.__enter__.return_value = mock_pdf
                mock_pdf.__exit__.return_value = None
                mock_pdfplumber.open.return_value = mock_pdf
                
                mock_ocr.return_value = ""  # OCR also fails
                
                text, method = self.extractor._extract_hybrid(Path("test.pdf"))
                
                self.assertEqual(text, "")
                self.assertIn("failed", method.lower())


class TestPDFExtractionIntegration(unittest.TestCase):
    """Integration tests for PDF extraction with other components."""
    
    def test_integration_with_correspondence_extractor(self):
        """Test integration between PDF extractor and correspondence extractor."""
        if not Path("data/Lot-11").exists():
            self.skipTest("Test data not available")
            
        from extract_correspondence_content import CorrespondenceExtractor
        
        pdf_extractor = PDFExtractor(max_pages=2)
        correspondence_extractor = CorrespondenceExtractor()
        
        sample_pdf = Path("data/Lot-11/20200821_AE_PD_392 - Provision of Toll Plaza at Ch. Km. 104+917 (Chennasamudram Village).pdf")
        
        if sample_pdf.exists():
            # Extract raw text
            raw_text, extraction_method = pdf_extractor.extract_text(sample_pdf)
            self.assertGreater(len(raw_text), 100)
            
            # Extract correspondence content
            correspondence_result = correspondence_extractor.extract_correspondence_content(raw_text)
            
            self.assertIsInstance(correspondence_result['subject'], str)
            self.assertIsInstance(correspondence_result['body'], str)
            self.assertIn(correspondence_result['extraction_method'], 
                         ['structured_extraction', 'full_text_fallback', 'insufficient_content'])


if __name__ == '__main__':
    # Support running specific test classes or methods
    if len(sys.argv) > 1:
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        for arg in sys.argv[1:]:
            if hasattr(sys.modules[__name__], arg):
                suite.addTests(loader.loadTestsFromTestCase(getattr(sys.modules[__name__], arg)))
            else:
                # Try to find test method
                for cls_name in ['TestPDFExtractor', 'TestPDFExtractionIntegration']:
                    if hasattr(sys.modules[__name__], cls_name):
                        cls = getattr(sys.modules[__name__], cls_name)
                        if hasattr(cls, arg):
                            suite.addTest(cls(arg))
        
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)
    else:
        unittest.main(verbosity=2)