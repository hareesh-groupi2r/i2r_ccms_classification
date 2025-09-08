#!/usr/bin/env python3
"""
Unit tests for correspondence content extraction
"""

import unittest
import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from extract_correspondence_content import CorrespondenceExtractor

# Suppress logging during tests
logging.getLogger().setLevel(logging.WARNING)


class TestCorrespondenceExtraction(unittest.TestCase):
    """Test cases for correspondence extraction functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = CorrespondenceExtractor()
        
        # Sample correspondence text
        self.sample_text = """
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
        
        # OCR-like text (continuous, no line breaks)
        self.ocr_text = "Subject: Change of Scope Proposal for toll plaza construction at Chennasamudram Village Sir, It is to inform that as per the present Scope of Works the number of Toll Lanes to be made at the Chennasamudram Toll Plaza are 24. However, based on the NHAI letter it is understood that the number of Toll Lanes to be re-designed duly considering Hybrid ETC system. We request approval for proceeding with this revised proposal."
    
    def test_extract_subject_standard_format(self):
        """Test subject extraction from standard format."""
        result = self.extractor.extract_correspondence_content(self.sample_text)
        
        self.assertIn("Provision of Toll Plaza", result['subject'])
        self.assertIn("Chennasamudram Village", result['subject'])
        self.assertEqual(result['extraction_method'], 'structured_extraction')
    
    def test_extract_subject_ocr_format(self):
        """Test subject extraction from OCR-like text."""
        result = self.extractor.extract_correspondence_content(self.ocr_text)
        
        self.assertIn("Change of Scope", result['subject'])
        self.assertIn("toll plaza", result['subject'])
    
    def test_extract_body_after_salutation(self):
        """Test body extraction starts after salutation."""
        result = self.extractor.extract_correspondence_content(self.sample_text)
        
        # Body should start after "Dear Sir," and not include headers
        self.assertIn("We are writing to submit", result['body'])
        self.assertNotIn("Letter No:", result['body'])
        self.assertNotIn("CONFIDENTIAL", result['body'])
        self.assertNotIn("Dear Sir", result['body'])
    
    def test_extract_body_before_signature(self):
        """Test body extraction stops before signature."""
        result = self.extractor.extract_correspondence_content(self.sample_text)
        
        # Body should not include signature section
        self.assertNotIn("Yours faithfully", result['body'])
        self.assertNotIn("Project Manager", result['body'])
        self.assertNotIn("CC:", result['body'])
    
    def test_focused_content_generation(self):
        """Test focused content combines subject and body properly."""
        focused_content = self.extractor.get_focused_content(self.sample_text)
        
        self.assertIn("Subject:", focused_content)
        self.assertIn("Content:", focused_content)
        self.assertIn("Provision of Toll Plaza", focused_content)
        self.assertIn("We are writing to submit", focused_content)
    
    def test_insufficient_content_handling(self):
        """Test handling of insufficient content."""
        short_text = "Too short"
        result = self.extractor.extract_correspondence_content(short_text)
        
        self.assertEqual(result['extraction_method'], 'insufficient_content')
        self.assertEqual(result['subject'], '')
        self.assertEqual(result['body'], short_text)
    
    def test_no_subject_fallback(self):
        """Test fallback when no subject is found."""
        no_subject_text = """
        Dear Sir,
        
        This is a letter without a clear subject line.
        We need to process this somehow.
        
        Thank you.
        """
        
        result = self.extractor.extract_correspondence_content(no_subject_text)
        
        # Should still extract body content
        self.assertIn("This is a letter", result['body'])
        # Subject might be empty or found via keyword matching
        self.assertTrue(isinstance(result['subject'], str))
    
    def test_multiple_subject_patterns(self):
        """Test different subject line formats."""
        test_cases = [
            "Sub: Test Subject Line\nDear Sir,\nContent here.",
            "RE: Test Subject Line\nDear Sir,\nContent here.",
            "SUBJECT. Test Subject Line\nDear Sir,\nContent here.",
            "Regarding: Test Subject Line\nDear Sir,\nContent here."
        ]
        
        for text in test_cases:
            result = self.extractor.extract_correspondence_content(text)
            self.assertIn("Test Subject Line", result['subject'])
    
    def test_noise_removal(self):
        """Test removal of headers, footers, and noise."""
        noisy_text = """
        Page 1 of 3
        CONFIDENTIAL
        Copyright 2020
        
        Subject: Clean Subject Line
        
        Dear Sir,
        Important content that should remain.
        
        Yours faithfully.
        """
        
        result = self.extractor.extract_correspondence_content(noisy_text)
        
        self.assertNotIn("Page 1 of 3", result['body'])
        self.assertNotIn("CONFIDENTIAL", result['body'])
        self.assertNotIn("Copyright", result['body'])
        self.assertIn("Important content", result['body'])
    
    def test_body_start_patterns(self):
        """Test different salutation patterns for body start."""
        salutation_patterns = [
            "Dear Sir,",
            "Dear Madam,",
            "To whom it may concern,",
            "Respected Sir,",
            "Sir,",
            "Madam,"
        ]
        
        for salutation in salutation_patterns:
            text = f"Subject: Test\n\n{salutation}\n\nBody content here.\n\nThank you."
            result = self.extractor.extract_correspondence_content(text)
            
            self.assertIn("Body content here", result['body'])
            self.assertNotIn(salutation, result['body'])


class TestCorrespondenceIntegration(unittest.TestCase):
    """Integration tests with actual PDF extraction."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = CorrespondenceExtractor()
    
    def test_integration_with_pdf_extractor(self):
        """Test integration with PDF extractor if sample file exists."""
        from pathlib import Path
        from classifier.pdf_extractor import PDFExtractor
        
        sample_pdf = Path("data/Lot-11/20200821_AE_PD_392 - Provision of Toll Plaza at Ch. Km. 104+917 (Chennasamudram Village).pdf")
        
        if sample_pdf.exists():
            pdf_extractor = PDFExtractor(max_pages=2)
            raw_text, method = pdf_extractor.extract_text(sample_pdf)
            
            self.assertGreater(len(raw_text), 100)
            
            # Test correspondence extraction on real PDF text
            result = self.extractor.extract_correspondence_content(raw_text)
            
            self.assertTrue(isinstance(result['subject'], str))
            self.assertTrue(isinstance(result['body'], str))
            self.assertIn(result['extraction_method'], ['structured_extraction', 'full_text_fallback'])
        else:
            self.skipTest("Sample PDF not available for integration test")


if __name__ == '__main__':
    # Run specific test methods if needed
    if len(sys.argv) > 1:
        # Run specific test
        suite = unittest.TestSuite()
        for test_name in sys.argv[1:]:
            if hasattr(TestCorrespondenceExtraction, test_name):
                suite.addTest(TestCorrespondenceExtraction(test_name))
            elif hasattr(TestCorrespondenceIntegration, test_name):
                suite.addTest(TestCorrespondenceIntegration(test_name))
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)
    else:
        # Run all tests
        unittest.main(verbosity=2)