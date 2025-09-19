"""
PDF Text Extraction Module
Handles text extraction from PDF files, including OCR for scanned documents
"""

import os
from pathlib import Path
from typing import Tuple, Optional, List
import logging
import pdfplumber
import PyPDF2
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import tempfile

logger = logging.getLogger(__name__)


class PDFExtractor:
    """
    Extracts text from PDF files with fallback to OCR for scanned documents.
    """
    
    def __init__(self, 
                 ocr_threshold: int = 100,
                 tesseract_cmd: str = None,
                 dpi: int = 300,
                 max_pages: int = None):
        """
        Initialize PDF extractor.
        
        Args:
            ocr_threshold: Minimum characters to consider text extraction successful
            tesseract_cmd: Path to tesseract executable
            dpi: DPI for PDF to image conversion
            max_pages: Maximum number of pages to extract (None for all pages)
        """
        self.ocr_threshold = ocr_threshold
        self.dpi = dpi
        self.max_pages = max_pages
        
        # Set tesseract command if provided or from environment
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        elif os.getenv('TESSERACT_CMD'):
            pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_CMD')
        
        logger.info(f"PDFExtractor initialized (max_pages: {max_pages if max_pages else 'all'})")
    
    def extract_text(self, pdf_path: str) -> Tuple[str, str]:
        """
        Extract text from PDF, using OCR if necessary.
        Now supports mixed documents with both text and scanned pages.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (extracted_text, extraction_method)
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Try hybrid extraction (text + OCR for scanned pages)
        text, method = self._extract_hybrid(pdf_path)
        
        if not text:
            logger.warning(f"Failed to extract text from {pdf_path}")
            return "", "failed"
        
        logger.info(f"Extracted {len(text)} characters using {method}")
        return text, method
    
    def _extract_hybrid(self, pdf_path: Path) -> Tuple[str, str]:
        """
        Extract text using hybrid approach: text extraction + OCR for scanned pages.
        Respects max_pages limit and processes pages in order.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (extracted_text, extraction_method)
        """
        try:
            text_pages = []
            methods_used = []
            
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                pages_to_process = min(total_pages, self.max_pages) if self.max_pages else total_pages
                
                logger.info(f"Processing {pages_to_process} of {total_pages} pages (max_pages: {self.max_pages})")
                
                for i, page in enumerate(pdf.pages[:pages_to_process]):
                    page_num = i + 1
                    page_text = ""
                    method = "failed"
                    
                    try:
                        # First try text extraction
                        page_text = page.extract_text()
                        
                        # Check if we got meaningful text
                        if page_text and len(page_text.strip()) > 50:  # At least 50 chars
                            method = "text"
                            logger.debug(f"Page {page_num}: Extracted {len(page_text)} chars via text")
                        else:
                            # Try OCR for this page
                            logger.info(f"Page {page_num}: Text extraction insufficient, trying OCR...")
                            page_text = self._extract_page_ocr(pdf_path, page_num)
                            
                            if page_text and len(page_text.strip()) > 20:  # Lower threshold for OCR
                                method = "ocr"
                                logger.debug(f"Page {page_num}: Extracted {len(page_text)} chars via OCR")
                            else:
                                logger.warning(f"Page {page_num}: Both text and OCR extraction failed")
                                page_text = ""
                                method = "failed"
                    
                    except Exception as e:
                        logger.warning(f"Error processing page {page_num}: {e}")
                        page_text = ""
                        method = "error"
                    
                    # Add page content if we got something
                    if page_text and page_text.strip():
                        text_pages.append(f"Page {page_num}:\n{page_text.strip()}")
                        methods_used.append(method)
                
                # Compile results
                if text_pages:
                    combined_text = "\n\n".join(text_pages)
                    
                    # Determine overall method
                    if "ocr" in methods_used and "text" in methods_used:
                        overall_method = f"hybrid_text+ocr({len(text_pages)}pages)"
                    elif "ocr" in methods_used:
                        overall_method = f"ocr_only({len(text_pages)}pages)" 
                    else:
                        overall_method = f"text_only({len(text_pages)}pages)"
                    
                    logger.info(f"Hybrid extraction successful: {len(combined_text)} chars using {overall_method}")
                    return combined_text, overall_method
                else:
                    logger.warning("No pages could be processed successfully")
                    return "", "hybrid_failed"
                    
        except Exception as e:
            logger.error(f"Hybrid extraction failed: {e}")
            return "", f"hybrid_error: {str(e)}"
    
    def _extract_page_ocr(self, pdf_path: Path, page_num: int) -> str:
        """
        Extract text from a specific page using OCR.
        
        Args:
            pdf_path: Path to the PDF file
            page_num: Page number (1-indexed)
            
        Returns:
            Extracted text from the page
        """
        try:
            # Convert specific page to image
            images = convert_from_path(
                pdf_path, 
                dpi=self.dpi,
                first_page=page_num,
                last_page=page_num
            )
            
            if not images:
                return ""
            
            # Apply OCR to the page
            page_text = pytesseract.image_to_string(images[0])
            
            # Clean up images to free resources
            for image in images:
                try:
                    if hasattr(image, 'close'):
                        image.close()
                except Exception:
                    pass
            images.clear()
            
            return self._clean_ocr_text(page_text)
            
        except Exception as e:
            logger.warning(f"OCR failed for page {page_num}: {e}")
            return ""
    
    def _extract_with_pdfplumber(self, pdf_path: Path) -> Tuple[str, str]:
        """
        Extract text using pdfplumber.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (extracted_text, method_name)
        """
        try:
            text_pages = []
            
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                pages_to_process = min(total_pages, self.max_pages) if self.max_pages else total_pages
                
                logger.debug(f"Processing {pages_to_process} of {total_pages} pages (max_pages: {self.max_pages})")
                
                for i, page in enumerate(pdf.pages[:pages_to_process]):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_pages.append(f"Page {i+1}:\n{page_text}")
                    except Exception as e:
                        logger.warning(f"Error extracting page {i+1} with pdfplumber: {e}")
            
            if text_pages:
                method = f"pdfplumber({pages_to_process}pages)" if self.max_pages else "pdfplumber"
                return "\n\n".join(text_pages), method
            
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
        
        return "", "pdfplumber_failed"
    
    def _extract_with_pypdf2(self, pdf_path: Path) -> Tuple[str, str]:
        """
        Extract text using PyPDF2 as fallback.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (extracted_text, method_name)
        """
        try:
            text_pages = []
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                pages_to_process = min(total_pages, self.max_pages) if self.max_pages else total_pages
                
                logger.debug(f"Processing {pages_to_process} of {total_pages} pages (max_pages: {self.max_pages})")
                
                for i, page in enumerate(pdf_reader.pages[:pages_to_process]):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_pages.append(f"Page {i+1}:\n{page_text}")
                    except Exception as e:
                        logger.warning(f"Error extracting page {i+1} with PyPDF2: {e}")
            
            if text_pages:
                method = f"pypdf2({pages_to_process}pages)" if self.max_pages else "pypdf2"
                return "\n\n".join(text_pages), method
            
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {e}")
        
        return "", "pypdf2_failed"
    
    def _extract_with_ocr(self, pdf_path: Path) -> Tuple[str, str]:
        """
        Extract text using OCR (Tesseract) for scanned PDFs.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (extracted_text, method_name)
        """
        try:
            # Convert PDF to images
            with tempfile.TemporaryDirectory() as temp_dir:
                # Convert only the first max_pages if specified
                images = convert_from_path(
                    pdf_path, 
                    dpi=self.dpi,
                    output_folder=temp_dir,
                    first_page=1,
                    last_page=self.max_pages if self.max_pages else None
                )
                
                text_pages = []
                total_pages = len(images)
                
                logger.debug(f"Processing {total_pages} pages for OCR (max_pages: {self.max_pages})")
                
                for i, image in enumerate(images):
                    try:
                        # Apply OCR to each page
                        page_text = pytesseract.image_to_string(image)
                        
                        # Clean up OCR text
                        page_text = self._clean_ocr_text(page_text)
                        
                        if page_text:
                            text_pages.append(f"Page {i+1}:\n{page_text}")
                            
                    except Exception as e:
                        logger.warning(f"Error in OCR for page {i+1}: {e}")
                    finally:
                        # Explicitly close image to free resources
                        try:
                            if hasattr(image, 'close'):
                                image.close()
                        except Exception:
                            pass
                
                # Clear images list to help with garbage collection
                images.clear()
                
                if text_pages:
                    method = f"tesseract_ocr({total_pages}pages)" if self.max_pages else "tesseract_ocr"
                    return "\n\n".join(text_pages), method
                
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
        
        return "", "ocr_failed"
    
    def _clean_ocr_text(self, text: str) -> str:
        """
        Clean up common OCR artifacts and errors.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Fix common OCR errors
        replacements = {
            '|': 'I',  # Pipe often confused with I
            '0': 'O',  # Zero confused with O in certain contexts
            '1': 'l',  # One confused with lowercase L in certain contexts
        }
        
        # Context-aware replacement would be better, but for now, skip replacements
        # that might break numbers and proper text
        
        # Remove lines that are mostly special characters (likely OCR noise)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Count alphanumeric vs special characters
            if line:
                alnum_count = sum(c.isalnum() or c.isspace() for c in line)
                total_count = len(line)
                
                # Keep line if it's at least 50% alphanumeric
                if total_count > 0 and alnum_count / total_count > 0.5:
                    cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def extract_from_multiple(self, pdf_paths: List[str]) -> List[Tuple[str, str, str]]:
        """
        Extract text from multiple PDF files.
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            List of tuples (file_path, extracted_text, extraction_method)
        """
        results = []
        
        for pdf_path in pdf_paths:
            try:
                text, method = self.extract_text(pdf_path)
                results.append((pdf_path, text, method))
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
                results.append((pdf_path, "", "error"))
        
        return results
    
    def extract_metadata(self, pdf_path: str) -> dict:
        """
        Extract metadata from PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing metadata
        """
        metadata = {
            'page_count': 0,
            'title': '',
            'author': '',
            'subject': '',
            'creator': '',
            'producer': '',
            'creation_date': None,
            'modification_date': None,
            'file_size': 0
        }
        
        try:
            pdf_path = Path(pdf_path)
            metadata['file_size'] = pdf_path.stat().st_size
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata['page_count'] = len(pdf_reader.pages)
                
                if pdf_reader.metadata:
                    info = pdf_reader.metadata
                    metadata['title'] = info.get('/Title', '')
                    metadata['author'] = info.get('/Author', '')
                    metadata['subject'] = info.get('/Subject', '')
                    metadata['creator'] = info.get('/Creator', '')
                    metadata['producer'] = info.get('/Producer', '')
                    metadata['creation_date'] = info.get('/CreationDate')
                    metadata['modification_date'] = info.get('/ModDate')
                    
        except Exception as e:
            logger.warning(f"Error extracting metadata from {pdf_path}: {e}")
        
        return metadata
    
    def is_scanned_pdf(self, pdf_path: str) -> bool:
        """
        Check if a PDF is likely a scanned document.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            True if PDF appears to be scanned
        """
        try:
            # Try to extract text with regular method
            text, _ = self._extract_with_pdfplumber(Path(pdf_path))
            
            # If very little text extracted, likely scanned
            if len(text.strip()) < self.ocr_threshold:
                return True
            
            # Check if text is mostly whitespace
            non_whitespace = len(text.replace(' ', '').replace('\n', '').replace('\t', ''))
            if non_whitespace < 50:
                return True
                
        except Exception as e:
            logger.warning(f"Error checking if PDF is scanned: {e}")
        
        return False
    
    def __repr__(self):
        return f"PDFExtractor(ocr_threshold={self.ocr_threshold}, dpi={self.dpi})"