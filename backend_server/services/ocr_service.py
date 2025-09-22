"""
OCR Service
Modular service for text extraction from documents using multiple OCR backends
"""

import os
import base64
import tempfile
import concurrent.futures
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract
from google.cloud import documentai_v1beta3 as documentai

from .interfaces import IOCRService, ProcessingResult, ProcessingStatus
from .configuration_service import get_config_service


class OCRService(IOCRService):
    """Service for text extraction using multiple OCR backends"""
    
    def __init__(self, config_service=None):
        self.config_service = config_service or get_config_service()
        self.config = self.config_service.get_service_config("ocr")
        self.docai_config = self.config_service.get_service_config("docai")
        
        # OCR configuration
        self.fallback_to_tesseract = self.config.get("fallback_to_tesseract", True)
        self.tesseract_language = self.config.get("tesseract_language", "eng")
        self.min_text_length = self.config.get("min_text_length", 50)
        self.concurrent_pages = self.config.get("concurrent_pages", 4)
        self.image_dpi = self.config.get("image_dpi", 300)
        
        # Document AI configuration
        self.docai_project_id = self.docai_config.get("project_id")
        self.docai_location = self.docai_config.get("location", "us")
        self.docai_processor_id = self.docai_config.get("processor_id")
        self.docai_enabled = all([self.docai_project_id, self.docai_processor_id])
        
        if self.docai_enabled:
            try:
                self.docai_client = documentai.DocumentProcessorServiceClient()
            except Exception as e:
                print(f"Warning: Could not initialize Document AI client: {e}")
                self.docai_enabled = False
    
    def extract_text(self, file_path: str, **kwargs) -> ProcessingResult:
        """
        Extract text from document using best available method
        
        Args:
            file_path: Path to the document file
            **kwargs: Extraction options
                - method: 'auto', 'docai', 'pypdf', 'tesseract'
                - page_range: tuple of (start, end) pages
                - fallback_on_error: bool, try alternative methods on failure
        
        Returns:
            ProcessingResult with extracted text
        """
        try:
            if not Path(file_path).exists():
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    error_message=f"File not found: {file_path}"
                )
            
            extraction_method = kwargs.get("method", "auto")
            page_range = kwargs.get("page_range")
            fallback_on_error = kwargs.get("fallback_on_error", True)
            
            # Try extraction methods in order of preference
            methods_to_try = self._get_extraction_methods(extraction_method)
            
            last_error = None
            for method_name, method_func in methods_to_try:
                try:
                    result = method_func(file_path, page_range=page_range, **kwargs)
                    
                    if result.status == ProcessingStatus.SUCCESS:
                        result.metadata = result.metadata or {}
                        result.metadata["extraction_method"] = method_name
                        return result
                    else:
                        last_error = result.error_message
                        if not fallback_on_error:
                            return result
                        
                except Exception as e:
                    last_error = str(e)
                    if not fallback_on_error:
                        break
                    continue
            
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message=f"All extraction methods failed. Last error: {last_error}"
            )
            
        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message=f"Error in text extraction: {str(e)}"
            )
    
    def extract_text_from_pages(self, file_path: str, page_numbers: List[int], **kwargs) -> ProcessingResult:
        """
        Extract text from specific pages
        
        Args:
            file_path: Path to the document file
            page_numbers: List of page numbers (1-indexed)
            **kwargs: Extraction options
        
        Returns:
            ProcessingResult with dict of {page_num: text}
        """
        try:
            if not page_numbers:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    error_message="No page numbers specified"
                )
            
            # Convert to 0-indexed
            zero_indexed_pages = [p - 1 for p in page_numbers if p > 0]
            
            if not zero_indexed_pages:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    error_message="Invalid page numbers provided"
                )
            
            extraction_method = kwargs.get("method", "pypdf")  # Use PyPDF for specific pages
            
            if extraction_method == "pypdf":
                return self._extract_pages_with_pypdf(file_path, zero_indexed_pages, **kwargs)
            elif extraction_method == "tesseract":
                return self._extract_pages_with_tesseract(file_path, zero_indexed_pages, **kwargs)
            else:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    error_message=f"Page-specific extraction not supported for method: {extraction_method}"
                )
                
        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message=f"Error extracting from specific pages: {str(e)}"
            )
    
    def _get_extraction_methods(self, preferred_method: str) -> List[Tuple[str, callable]]:
        """Get list of extraction methods to try in order"""
        methods = []
        
        if preferred_method == "auto":
            # Try best methods first
            if self.docai_enabled:
                methods.append(("docai", self._extract_with_docai))
            methods.append(("pypdf", self._extract_with_pypdf))
            if self.fallback_to_tesseract:
                methods.append(("tesseract", self._extract_with_tesseract))
        
        elif preferred_method == "docai" and self.docai_enabled:
            methods.append(("docai", self._extract_with_docai))
        
        elif preferred_method == "pypdf":
            methods.append(("pypdf", self._extract_with_pypdf))
        
        elif preferred_method == "tesseract":
            methods.append(("tesseract", self._extract_with_tesseract))
        
        else:
            # Fallback to available methods
            if self.docai_enabled:
                methods.append(("docai", self._extract_with_docai))
            methods.append(("pypdf", self._extract_with_pypdf))
        
        return methods
    
    def _extract_with_docai(self, file_path: str, **kwargs) -> ProcessingResult:
        """Extract text using Google Document AI"""
        try:
            with open(file_path, "rb") as pdf_file:
                pdf_content = pdf_file.read()
            
            # Create processor name
            processor_name = self.docai_client.processor_path(
                self.docai_project_id,
                self.docai_location,
                self.docai_processor_id
            )
            
            # Create request
            request = documentai.ProcessRequest(
                name=processor_name,
                raw_document=documentai.RawDocument(
                    content=pdf_content,
                    mime_type="application/pdf"
                )
            )
            
            # Process document
            result = self.docai_client.process_document(request=request)
            
            if not result.document or not result.document.text:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    error_message="Document AI returned no text content"
                )
            
            extracted_text = result.document.text
            
            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=extracted_text,
                confidence=0.9,  # High confidence for Document AI
                metadata={
                    "text_length": len(extracted_text),
                    "processor_id": self.docai_processor_id,
                    "pages_processed": len(result.document.pages) if result.document.pages else 0
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message=f"Document AI extraction failed: {str(e)}"
            )
    
    def _extract_with_pypdf(self, file_path: str, **kwargs) -> ProcessingResult:
        """Extract text using PyPDF"""
        try:
            reader = PdfReader(file_path)
            text_content = ""
            pages_processed = 0
            
            page_range = kwargs.get("page_range")
            if page_range:
                start_page, end_page = page_range
                pages_to_process = range(start_page, min(end_page + 1, len(reader.pages)))
            else:
                pages_to_process = range(len(reader.pages))
            
            for i in pages_to_process:
                try:
                    page = reader.pages[i]
                    extracted = page.extract_text()
                    if extracted and extracted.strip():
                        text_content += extracted + "\n"
                        pages_processed += 1
                except Exception as page_error:
                    print(f"Warning: Could not extract text from page {i+1}: {page_error}")
                    continue
            
            if len(text_content.strip()) < self.min_text_length:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    error_message=f"Extracted text too short ({len(text_content)} chars)"
                )
            
            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=text_content,
                confidence=0.7,  # Medium confidence for PyPDF
                metadata={
                    "text_length": len(text_content),
                    "pages_processed": pages_processed,
                    "total_pages": len(reader.pages)
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message=f"PyPDF extraction failed: {str(e)}"
            )
    
    def _extract_with_tesseract(self, file_path: str, **kwargs) -> ProcessingResult:
        """Extract text using Tesseract OCR"""
        try:
            # Convert PDF to images
            page_range = kwargs.get("page_range")
            if page_range:
                start_page, end_page = page_range
                images = convert_from_path(
                    file_path, 
                    dpi=self.image_dpi,
                    first_page=start_page + 1,  # convert_from_path uses 1-indexing
                    last_page=end_page + 1
                )
            else:
                images = convert_from_path(file_path, dpi=self.image_dpi)
            
            if not images:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    error_message="Could not convert PDF pages to images"
                )
            
            # Extract text from images using concurrent processing
            text_parts = []
            
            def process_image(image):
                try:
                    return pytesseract.image_to_string(image, lang=self.tesseract_language)
                except Exception as e:
                    return f"[OCR Error: {str(e)}]"
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrent_pages) as executor:
                text_parts = list(executor.map(process_image, images))
            
            # Combine all text
            full_text = "\n".join(text_parts)
            
            if len(full_text.strip()) < self.min_text_length:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    error_message=f"OCR extracted text too short ({len(full_text)} chars)"
                )
            
            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=full_text,
                confidence=0.6,  # Lower confidence for OCR
                metadata={
                    "text_length": len(full_text),
                    "pages_processed": len(images),
                    "ocr_language": self.tesseract_language,
                    "image_dpi": self.image_dpi
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message=f"Tesseract OCR failed: {str(e)}"
            )
    
    def _extract_pages_with_pypdf(self, file_path: str, page_indices: List[int], **kwargs) -> ProcessingResult:
        """Extract text from specific pages using PyPDF"""
        try:
            reader = PdfReader(file_path)
            page_texts = {}
            
            for page_idx in page_indices:
                if page_idx >= len(reader.pages):
                    page_texts[page_idx + 1] = "[Page not found]"
                    continue
                
                try:
                    page = reader.pages[page_idx]
                    extracted = page.extract_text()
                    page_texts[page_idx + 1] = extracted or "[No text extracted]"
                except Exception as page_error:
                    page_texts[page_idx + 1] = f"[Error: {str(page_error)}]"
            
            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=page_texts,
                confidence=0.7,
                metadata={
                    "pages_requested": len(page_indices),
                    "pages_processed": len(page_texts)
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message=f"PyPDF page extraction failed: {str(e)}"
            )
    
    def _extract_pages_with_tesseract(self, file_path: str, page_indices: List[int], **kwargs) -> ProcessingResult:
        """Extract text from specific pages using Tesseract"""
        try:
            page_texts = {}
            
            for page_idx in page_indices:
                try:
                    # Convert single page to image
                    images = convert_from_path(
                        file_path,
                        dpi=self.image_dpi,
                        first_page=page_idx + 1,
                        last_page=page_idx + 1
                    )
                    
                    if images:
                        text = pytesseract.image_to_string(images[0], lang=self.tesseract_language)
                        page_texts[page_idx + 1] = text or "[No text extracted]"
                    else:
                        page_texts[page_idx + 1] = "[Could not convert page to image]"
                        
                except Exception as page_error:
                    page_texts[page_idx + 1] = f"[OCR Error: {str(page_error)}]"
            
            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=page_texts,
                confidence=0.6,
                metadata={
                    "pages_requested": len(page_indices),
                    "pages_processed": len(page_texts),
                    "ocr_method": "tesseract"
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message=f"Tesseract page extraction failed: {str(e)}"
            )
    
    def get_available_methods(self) -> ProcessingResult:
        """Get list of available OCR methods"""
        methods = []
        
        # Check PyPDF availability
        methods.append({
            "name": "pypdf",
            "available": True,
            "description": "Direct PDF text extraction"
        })
        
        # Check Document AI availability
        methods.append({
            "name": "docai",
            "available": self.docai_enabled,
            "description": "Google Document AI OCR",
            "config": {
                "project_id": self.docai_project_id,
                "processor_id": self.docai_processor_id
            } if self.docai_enabled else None
        })
        
        # Check Tesseract availability
        tesseract_available = False
        try:
            pytesseract.get_tesseract_version()
            tesseract_available = True
        except Exception:
            pass
        
        methods.append({
            "name": "tesseract",
            "available": tesseract_available,
            "description": "Tesseract OCR for image-based text extraction"
        })
        
        return ProcessingResult(
            status=ProcessingStatus.SUCCESS,
            data=methods,
            metadata={"total_methods": len(methods)}
        )