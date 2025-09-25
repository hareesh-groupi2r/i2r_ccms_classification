"""
Document Processing Orchestrator
Coordinates multiple document processing services for end-to-end document analysis
"""

import tempfile
import shutil
import os
import logging
import zipfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
try:
    from pypdf import PdfReader, PdfWriter
except ImportError:
    # Fallback to PyPDF2 if pypdf not available
    try:
        from PyPDF2 import PdfReader, PdfWriter
    except ImportError:
        print("Warning: Neither pypdf nor PyPDF2 available. PDF optimization disabled.")
        PdfReader = None
        PdfWriter = None

from .interfaces import IDocumentProcessingOrchestrator, ProcessingResult, ProcessingStatus, DocumentType
from .configuration_service import get_config_service
from .document_type_service import DocumentTypeService
from .ocr_service import OCRService
from .llm_service import LLMService
from .category_mapping_service import CategoryMappingService


@dataclass
class ProcessingContext:
    """Context information for document processing"""
    file_path: str
    document_id: Optional[str] = None
    processing_options: Dict[str, Any] = field(default_factory=dict)
    temp_files: List[str] = field(default_factory=list)
    processing_start_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingResults:
    """Complete processing results for a document"""
    document_type: DocumentType
    extracted_text: str
    structured_data: Dict[str, Any] = field(default_factory=dict)
    classifications: Dict[str, str] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class DocumentProcessingOrchestrator(IDocumentProcessingOrchestrator):
    """Orchestrates document processing using modular services"""
    
    def __init__(self, config_service=None):
        self.config_service = config_service or get_config_service()
        self.config = self.config_service.get_service_config("pipeline")
        self.logger = logging.getLogger(__name__)
        
        # Initialize services
        self.document_type_service = DocumentTypeService(config_service)
        self.ocr_service = OCRService(config_service)
        self.llm_service = LLMService(config_service)
        self.category_mapping_service = CategoryMappingService(config_service)
        
        # Configuration
        self.temp_dir_prefix = self.config.get("temp_dir_prefix", "ccms_processing_")
        self.cleanup_temp_files = self.config.get("cleanup_temp_files", True)
        self.max_file_size_mb = self.config.get("max_file_size_mb", 200)
        self.processing_timeout = self.config.get("processing_timeout", 600)
        self.supported_formats = self.config.get("supported_formats", ["pdf", "png", "jpg", "jpeg", "tiff", "docx"])
        
        # PDF optimization settings
        self.optimize_pdfs = self.config.get("optimize_pdfs", True)
        self.short_pdf_cache_dir = self.config.get("short_pdf_cache_dir", "/tmp/ccms_short_pdfs")
        self.important_pages_keywords = self.config.get("pdf_optimization_keywords", [
            "agreement", "contract", "scope", "work", "payment", "schedule", 
            "milestone", "completion", "terms", "conditions", "parties",
            "contractor", "authority", "value", "amount", "duration"
        ])
        self.max_pages_to_analyze = self.config.get("max_pages_to_analyze", 20)
        self.always_include_first = self.config.get("always_include_first_pages", 3)
        self.always_include_last = self.config.get("always_include_last_pages", 2)
    
    def _get_file_content_hash(self, file_path: str) -> str:
        """Generate MD5 hash of file content for cache key"""
        import hashlib
        try:
            with open(file_path, 'rb') as f:
                file_content = f.read()
                md5_hash = hashlib.md5(file_content).hexdigest()
                return md5_hash[:16]  # Use first 16 chars for shorter filename
        except Exception as e:
            self.logger.warning(f"Could not generate file hash, falling back to size+name: {e}")
            # Fallback to file size + name if hash fails
            file_stat = os.stat(file_path)
            return f"{Path(file_path).stem}_{file_stat.st_size}"

    def _is_zip_file(self, file_path: str) -> bool:
        """Check if file is actually a ZIP file regardless of extension"""
        try:
            # Method 1: Check magic number/file signature
            with open(file_path, 'rb') as f:
                magic_bytes = f.read(4)
                # ZIP file signatures
                zip_signatures = [
                    b'\x50\x4B\x03\x04',  # Standard ZIP
                    b'\x50\x4B\x05\x06',  # Empty ZIP
                    b'\x50\x4B\x07\x08'   # Spanned ZIP
                ]
                if magic_bytes in zip_signatures:
                    return True
            
            # Method 2: Try to open as ZIP
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                # If we can read the file list, it's a valid ZIP
                zip_ref.namelist()
                return True
                
        except Exception as e:
            self.logger.debug(f"Not a ZIP file: {file_path} - {e}")
            return False

    def _extract_zip_contents(self, zip_path: str, extract_to: str) -> List[Dict[str, Any]]:
        """Extract ZIP contents and return list of extracted files with metadata"""
        extracted_files = []
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                self.logger.info(f"📦 ZIP contains {len(file_list)} files: {file_list}")
                
                for file_info in zip_ref.infolist():
                    if file_info.is_dir():
                        continue
                        
                    # Skip hidden files and system files
                    filename = file_info.filename
                    if filename.startswith('.') or filename.startswith('__MACOSX'):
                        continue
                        
                    # Extract file
                    try:
                        extracted_path = zip_ref.extract(file_info, extract_to)
                        file_size = os.path.getsize(extracted_path)
                        
                        # Determine file type
                        file_ext = Path(filename).suffix.lower()
                        is_pdf = file_ext == '.pdf'
                        is_image = file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp']
                        is_docx = file_ext in ['.docx', '.doc']

                        if is_pdf or is_image or is_docx:
                            file_type = 'pdf' if is_pdf else ('image' if is_image else 'docx')
                            extracted_files.append({
                                'path': extracted_path,
                                'original_name': filename,
                                'size': file_size,
                                'type': file_type,
                                'extension': file_ext
                            })
                            self.logger.info(f"✅ Extracted: {filename} ({file_size} bytes, type: {file_type.upper()})")
                        else:
                            self.logger.info(f"⚠️ Skipped unsupported file: {filename} (type: {file_ext})")
                            
                    except Exception as extract_error:
                        self.logger.error(f"❌ Failed to extract {filename}: {extract_error}")
                        
        except Exception as e:
            self.logger.error(f"❌ ZIP extraction failed: {e}")
            raise
            
        return extracted_files
    
    def _get_short_pdf_path(self, original_path: str) -> str:
        """Generate path for optimized short PDF version using content hash"""
        original_file = Path(original_path)
        cache_dir = Path(self.short_pdf_cache_dir)
        cache_dir.mkdir(exist_ok=True)
        
        # Create unique filename based on file content hash (not modification time)
        file_hash = self._get_file_content_hash(original_path)
        short_path = cache_dir / f"{original_file.stem}_{file_hash}_short.pdf"
        
        self.logger.info(f"🔍 PDF Cache: Generated cache path for '{original_file.name}' -> '{short_path.name}'")
        return str(short_path)
    
    def _should_use_short_version(self, original_path: str, short_path: str) -> bool:
        """Check if short version exists and is valid to use"""
        self.logger.info(f"🔍 PDF Cache: Checking if cached file exists: {Path(short_path).name}")
        
        if not os.path.exists(short_path):
            self.logger.info(f"❌ PDF Cache: Cached file does not exist, will create new one")
            return False
            
        try:
            # For content-hash based cache, we just need to verify file exists and is readable
            short_stat = os.stat(short_path)
            if short_stat.st_size > 0:
                self.logger.info(f"✅ PDF Cache: Found existing cached file ({short_stat.st_size} bytes), will reuse!")
                return True
            else:
                self.logger.info(f"❌ PDF Cache: Cached file exists but is empty, will recreate")
                return False
        except Exception as e:
            self.logger.warning(f"❌ PDF Cache: Error accessing cached file, will recreate: {e}")
            return False
    
    def _process_page_chunk(self, pdf_path, start_page_idx, end_page_idx, search_terms):
        """Helper function to process a chunk of pages in a separate thread - EXACT CLONE of utils.py"""
        local_important_pages = set()
        try:
            reader = PdfReader(pdf_path)
            for page_num in range(start_page_idx, end_page_idx):
                if page_num >= len(reader.pages):
                    break

                page = reader.pages[page_num]
                text = page.extract_text()

                if not text or len(text.strip()) < 50:
                    try:
                        from pdf2image import convert_from_path
                        import pytesseract
                        # Specify tesseract path explicitly
                        pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'
                        images = convert_from_path(pdf_path, first_page=page_num + 1, last_page=page_num + 1)
                        if images:
                            text = pytesseract.image_to_string(images[0])
                    except Exception as ocr_error:
                        print(f"Could not process page {page_num + 1} with OCR: {ocr_error}")
                        text = ""

                for term, flags in search_terms.items():
                    import re
                    if re.search(term, text, re.IGNORECASE | flags):
                        local_important_pages.add(page_num)
                        if page_num + 1 < len(reader.pages):
                            local_important_pages.add(page_num + 1)
                        print(f"  - Found '{term}' on page {page_num + 1}.")
        except Exception as e:
            print(f"Error processing chunk in {pdf_path}: {e}")
        return local_important_pages

    def _find_important_pages_with_regex(self, pdf_path: str) -> List[int]:
        """Find important pages using CONTRACT_SEARCH_TERMS - EXACT CLONE of utils.py find_important_pages"""
        import re
        import concurrent.futures
        
        # Focused CONTRACT_SEARCH_TERMS - Only find section headers, not every mention
        CONTRACT_SEARCH_TERMS = {
            "(?:This\\s+)?Agreement\\s+is\\s+entered\\s+into": 0,
            # Only find SCHEDULE J as a header (not every reference to Schedule J)
            "(?:\\n|^)\\s*SCHEDULE\\s*[- ]*\\s*J\\s*(?:\\n|$)": re.MULTILINE | re.IGNORECASE,
            # Only find SCHEDULE H as a header (not every reference to Schedule H) 
            "(?:\\n|^)\\s*SCHEDULE\\s*[- ]*\\s*H\\s*(?:\\n|$)": re.MULTILINE | re.IGNORECASE,
            # Only find ARTICLE 19 as a section header (not every mention)
            "(?:\\n|^)\\s*ARTICLE\\s+19\\s*(?:\\n|$)": re.MULTILINE,
        }
        
        important_pages = {0, 1}  # Always include the first two pages
        try:
            reader = PdfReader(pdf_path)
            num_pages = len(reader.pages)
            print(f"  Scanning {num_pages} pages using {os.cpu_count()} threads...")

            page_chunks = [(i, min(i + 50, num_pages)) for i in range(0, num_pages, 50)]

            with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                future_to_chunk = {
                    executor.submit(self._process_page_chunk, pdf_path, start, end, CONTRACT_SEARCH_TERMS):
                    (start, end) for start, end in page_chunks
                }
                for future in concurrent.futures.as_completed(future_to_chunk):
                    important_pages.update(future.result())
        except Exception as e:
            print(f"Error finding important pages in {pdf_path}: {e}")

        print(f"\\n  Finished scanning. Found {len(important_pages)} important pages.")
        return sorted(list(important_pages))

    def _create_short_pdf(self, original_path: str, short_path: str, document_type: DocumentType = None) -> ProcessingResult:
        """Create optimized short version of PDF using document type-specific approach"""
        try:
            # Check if PDF libraries are available
            if PdfReader is None or PdfWriter is None:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    error_message="PDF processing libraries not available",
                    confidence=0.0
                )
            
            print(f"Creating optimized PDF for {document_type}: {original_path} -> {short_path}")
            
            # Determine pages to extract based on document type
            important_pages = []
            optimization_method = "first_pages"
            
            if document_type in [DocumentType.MEETING_MINUTES, DocumentType.PROGRESS_REPORTS]:
                # For MOMs and Progress Reports, use only first 5 pages
                reader = PdfReader(original_path)
                total_pages = len(reader.pages)
                important_pages = list(range(min(5, total_pages)))
                optimization_method = f"first_5_pages_{document_type.value}"
                print(f"Using first 5 pages for {document_type.value} (total: {total_pages})")
                
            elif document_type == DocumentType.CONTRACT_AGREEMENTS:
                # For contracts, use the original CONTRACT_SEARCH_TERMS approach
                important_pages = self._find_important_pages_with_regex(original_path)
                optimization_method = "CONTRACT_SEARCH_TERMS_regex"
                
                if not important_pages:
                    print("Warning: No important pages found, using first 10 pages as fallback")
                    reader = PdfReader(original_path)
                    total_pages = len(reader.pages)
                    important_pages = list(range(min(10, total_pages)))
                    optimization_method = "first_10_pages_fallback"
            else:
                # For other document types, use first 5 pages
                reader = PdfReader(original_path)
                total_pages = len(reader.pages)
                important_pages = list(range(min(5, total_pages)))
                optimization_method = f"first_5_pages_default"
                print(f"Using first 5 pages for {document_type} (total: {total_pages})")
            
            # Create short PDF with selected pages
            with open(original_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                pdf_writer = PdfWriter()
                
                total_pages = len(pdf_reader.pages)
                
                # Add selected pages to new PDF
                for page_num in important_pages:
                    if page_num < total_pages:
                        pdf_writer.add_page(pdf_reader.pages[page_num])
                
                # Write optimized PDF
                os.makedirs(os.path.dirname(short_path), exist_ok=True)
                with open(short_path, 'wb') as output_file:
                    pdf_writer.write(output_file)
                
                optimization_info = {
                    "original_pages": total_pages,
                    "optimized_pages": len(important_pages),
                    "selected_pages": important_pages,
                    "compression_ratio": f"{len(important_pages)}/{total_pages} ({len(important_pages)/total_pages*100:.1f}%)",
                    "optimization_method": optimization_method,
                    "document_type": document_type.value if document_type else "unknown"
                }
                
                print(f"PDF optimization complete: {optimization_info}")
                
                return ProcessingResult(
                    status=ProcessingStatus.SUCCESS,
                    data=short_path,
                    metadata=optimization_info,
                    confidence=1.0
                )
                
        except Exception as e:
            print(f"Error creating short PDF: {e}")
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message=f"Failed to create optimized PDF: {str(e)}",
                confidence=0.0
            )
    
    def _get_optimized_file_path(self, original_path: str, document_type: DocumentType = None) -> Tuple[str, Dict[str, Any]]:
        """Get the best file path to use for processing (original or optimized)"""
        metadata = {"optimization_used": False, "file_path_used": original_path}
        
        self.logger.info(f"🚀 PDF Optimization: Starting optimization check for: {Path(original_path).name}")
        
        # Only optimize PDF files and if PDF libraries are available
        if (not original_path.lower().endswith('.pdf') or 
            not self.optimize_pdfs or 
            PdfReader is None or 
            PdfWriter is None):
            if PdfReader is None:
                metadata["optimization_disabled_reason"] = "PDF libraries not available"
                self.logger.warning(f"⚠️  PDF Optimization: Disabled - PDF libraries not available")
            else:
                self.logger.info(f"⚠️  PDF Optimization: Disabled - not a PDF or optimization turned off")
            return original_path, metadata
            
        short_path = self._get_short_pdf_path(original_path)
        
        # Check if we can use existing short version
        if self._should_use_short_version(original_path, short_path):
            self.logger.info(f"🎯 PDF Cache: SUCCESS - Using cached optimized PDF: {Path(short_path).name}")
            metadata.update({
                "optimization_used": True,
                "file_path_used": short_path,
                "optimization_source": "cached"
            })
            return short_path, metadata
        
        # Create new short version
        self.logger.info(f"🔨 PDF Optimization: Cache miss - Creating new optimized PDF...")
        optimization_result = self._create_short_pdf(original_path, short_path, document_type)
        
        if optimization_result.status == ProcessingStatus.SUCCESS:
            self.logger.info(f"✅ PDF Optimization: SUCCESS - Created new optimized PDF: {Path(short_path).name}")
            metadata.update({
                "optimization_used": True,
                "file_path_used": short_path,
                "optimization_source": "created",
                "optimization_stats": optimization_result.metadata
            })
            return short_path, metadata
        else:
            self.logger.error(f"❌ PDF Optimization: FAILED - Using original file: {optimization_result.error_message}")
            return original_path, metadata
    
    def process_document(self, file_path: str, processing_options: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """
        Process document end-to-end with all services
        
        Args:
            file_path: Path to document file
            processing_options: Dict of processing options
                - document_type: Force specific document type
                - extraction_method: OCR method preference
                - extract_fields: Specific fields to extract
                - classify_issues: Enable issue classification
                - skip_validation: Skip result validation
        
        Returns:
            ProcessingResult with complete ProcessingResults
        """
        context = ProcessingContext(
            file_path=file_path,
            processing_options=processing_options or {},
            document_id=processing_options.get("document_id") if processing_options else None
        )
        
        try:
            # Validate input
            validation_result = self._validate_input(context)
            if validation_result.status == ProcessingStatus.ERROR:
                return validation_result
            
            # Check if file is actually a ZIP file (but exclude DOCX files which are technically ZIP but should be processed as documents)
            file_extension = Path(context.file_path).suffix.lower()
            if self._is_zip_file(context.file_path) and file_extension != '.docx':
                self.logger.info(f"📦 Detected ZIP file masquerading as {Path(context.file_path).suffix}: {Path(context.file_path).name}")
                
                # Create temp directory for extraction
                temp_extract_dir = tempfile.mkdtemp(prefix="zip_extract_")
                context.temp_files.append(temp_extract_dir)
                
                try:
                    # Extract ZIP contents
                    extracted_files = self._extract_zip_contents(context.file_path, temp_extract_dir)
                    
                    if not extracted_files:
                        return self._create_error_result("No processable files found in ZIP archive", context)
                    
                    # Process each extracted file individually
                    all_results = []
                    for file_info in extracted_files:
                        self.logger.info(f"🔄 Processing extracted file: {file_info['original_name']}")
                        
                        # Create new processing options for extracted file
                        file_options = processing_options.copy() if processing_options else {}
                        file_options['original_filename'] = file_info['original_name']
                        file_options['source_zip'] = Path(context.file_path).name
                        file_options['extracted_file_info'] = file_info
                        
                        # Process the individual file
                        file_result = self.process_document(file_info['path'], file_options)
                        
                        # Add ZIP context to result
                        if hasattr(file_result, 'data') and file_result.data:
                            file_result.data['is_from_zip'] = True
                            file_result.data['source_zip'] = Path(context.file_path).name
                            file_result.data['extracted_file_info'] = file_info
                        
                        all_results.append({
                            'file_info': file_info,
                            'result': file_result
                        })
                    
                    # Return combined result for ZIP processing
                    return ProcessingResult(
                        status=ProcessingStatus.SUCCESS,
                        data={
                            'is_zip_archive': True,
                            'source_zip': Path(context.file_path).name,
                            'extracted_files_count': len(extracted_files),
                            'processing_results': all_results,
                            'message': f"Successfully processed ZIP archive with {len(extracted_files)} files"
                        },
                        error_message=None,
                        metadata=context.metadata
                    )
                    
                except Exception as zip_error:
                    return self._create_error_result(f"ZIP processing failed: {zip_error}", context)
            
            # Step 1: Document Type Classification (before optimization to guide page selection)
            type_result = self._classify_document_type(context)
            if type_result.status == ProcessingStatus.ERROR:
                return self._create_error_result(f"Document classification failed: {type_result.error_message}", context)
            
            document_type = type_result.data
            
            # OPTIMIZATION: Get optimized file path with document type guidance
            optimized_path, optimization_metadata = self._get_optimized_file_path(context.file_path, document_type)
            context.file_path = optimized_path  # Update context to use optimized file
            context.metadata.update(optimization_metadata)
            
            # Step 2: Text Extraction
            extraction_result = self._extract_text(context)
            if extraction_result.status == ProcessingStatus.ERROR:
                return self._create_error_result(f"Text extraction failed: {extraction_result.error_message}", context)
            
            extracted_text = extraction_result.data
            
            # Step 3: Structured Data Extraction (if requested)
            structured_data = {}
            if context.processing_options.get("extract_structured_data", True):
                extraction_schema = self._get_extraction_schema(document_type, context)
                if extraction_schema:
                    llm_result = self._extract_structured_data(extracted_text, extraction_schema, context)
                    if llm_result.status == ProcessingStatus.SUCCESS:
                        structured_data = llm_result.data
                    else:
                        context.metadata.setdefault("warnings", []).append(f"Structured data extraction failed: {llm_result.error_message}")
            
            # Step 4: Issue Classification (if requested and applicable)
            classifications = {}
            if context.processing_options.get("classify_issues", True) and structured_data:
                classification_result = self._classify_issues(structured_data, context)
                if classification_result.status == ProcessingStatus.SUCCESS:
                    classifications = classification_result.data
                else:
                    context.metadata.setdefault("warnings", []).append(f"Issue classification failed: {classification_result.error_message}")
            
            # Compile results
            results = ProcessingResults(
                document_type=document_type,
                extracted_text=extracted_text,
                structured_data=structured_data,
                classifications=classifications,
                confidence_scores={
                    "document_type": type_result.confidence,
                    "text_extraction": extraction_result.confidence,
                    "structured_extraction": llm_result.confidence if 'llm_result' in locals() else 0.0,
                    "issue_classification": classification_result.confidence if 'classification_result' in locals() else 0.0
                },
                processing_metadata={
                    "processing_time": (datetime.now() - context.processing_start_time).total_seconds(),
                    "services_used": self._get_services_used(context),
                    "file_metadata": validation_result.metadata,
                    **context.metadata
                }
            )
            
            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=results,
                confidence=self._calculate_overall_confidence(results),
                metadata={
                    "document_id": context.document_id,
                    "processing_context": context.processing_options
                }
            )
            
        except Exception as e:
            return self._create_error_result(f"Processing orchestration failed: {str(e)}", context)
        finally:
            self._cleanup_processing_context(context)
    
    def process_document_partial(self, file_path: str, services: List[str], **kwargs) -> ProcessingResult:
        """
        Process document with specific services only
        
        Args:
            file_path: Path to document file
            services: List of service names to use
            **kwargs: Service-specific options
        
        Returns:
            ProcessingResult with partial results
        """
        try:
            context = ProcessingContext(
                file_path=file_path,
                processing_options=kwargs
            )
            
            # Validate input
            validation_result = self._validate_input(context)
            if validation_result.status == ProcessingStatus.ERROR:
                return validation_result
            
            partial_results = {}
            
            for service_name in services:
                if service_name == "document_type":
                    result = self._classify_document_type(context)
                    partial_results["document_type"] = result
                
                elif service_name == "ocr":
                    result = self._extract_text(context)
                    partial_results["text_extraction"] = result
                
                elif service_name == "llm" and "extraction_schema" in kwargs:
                    # Need text first
                    if "text_extraction" not in partial_results:
                        text_result = self._extract_text(context)
                        if text_result.status == ProcessingStatus.SUCCESS:
                            partial_results["text_extraction"] = text_result
                        else:
                            partial_results["llm"] = ProcessingResult(
                                status=ProcessingStatus.ERROR,
                                error_message="Cannot run LLM without text extraction"
                            )
                            continue
                    
                    text = partial_results["text_extraction"].data
                    result = self._extract_structured_data(text, kwargs["extraction_schema"], context)
                    partial_results["llm"] = result
                
                elif service_name == "category_mapping" and "issues" in kwargs:
                    issues = kwargs["issues"]
                    result = self._classify_issues_list(issues, context)
                    partial_results["category_mapping"] = result
            
            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=partial_results,
                metadata={"services_requested": services}
            )
            
        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message=f"Partial processing failed: {str(e)}"
            )
        finally:
            self._cleanup_processing_context(context)
    
    def _validate_input(self, context: ProcessingContext) -> ProcessingResult:
        """Validate input file and processing options"""
        try:
            file_path = Path(context.file_path)
            
            # Check file exists
            if not file_path.exists():
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    error_message=f"File not found: {context.file_path}"
                )
            
            # Check file size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    error_message=f"File too large: {file_size_mb:.1f}MB (max: {self.max_file_size_mb}MB)"
                )
            
            # Check file format
            file_extension = file_path.suffix.lower().lstrip('.')
            if file_extension not in self.supported_formats:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    error_message=f"Unsupported format: {file_extension} (supported: {self.supported_formats})"
                )
            
            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data="Input validation passed",
                metadata={
                    "file_size_mb": file_size_mb,
                    "file_extension": file_extension,
                    "file_name": file_path.name
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message=f"Input validation failed: {str(e)}"
            )
    
    def _classify_document_type(self, context: ProcessingContext) -> ProcessingResult:
        """Classify document type"""
        # Check if type is forced in options
        forced_type = context.processing_options.get("document_type")
        if forced_type:
            try:
                doc_type = DocumentType(forced_type)
                return ProcessingResult(
                    status=ProcessingStatus.SUCCESS,
                    data=doc_type,
                    confidence=1.0,
                    metadata={"forced_type": True}
                )
            except ValueError:
                pass  # Fall back to automatic classification
        
        return self.document_type_service.classify_document(
            context.file_path,
            **context.processing_options
        )
    
    def _extract_text(self, context: ProcessingContext) -> ProcessingResult:
        """Extract text using OCR service"""
        extraction_options = {
            "method": context.processing_options.get("extraction_method", "auto"),
            "fallback_on_error": context.processing_options.get("fallback_on_error", True)
        }
        
        return self.ocr_service.extract_text(context.file_path, **extraction_options)
    
    def _get_extraction_schema(self, document_type: DocumentType, context: ProcessingContext) -> Optional[Dict[str, str]]:
        """Get extraction schema based on document type"""
        ref_data = self.config_service.get_service_config("reference_data")
        
        # Use custom schema if provided
        custom_schema = context.processing_options.get("extraction_schema")
        if custom_schema:
            return custom_schema
        
        # Use predefined schemas based on document type
        if document_type == DocumentType.CONTRACT_AGREEMENTS:
            return ref_data.get("contract_fields", {})
        elif document_type == DocumentType.CORRESPONDENCE:
            return {
                "letter_id": "Complete letter reference number (look for patterns like SPK/MISC./NH-4/EPC/2019/031, extract full reference with slashes and numbers)",
                "letter_date": "Date of the letter (convert any format like DD.MM.YYYY or DD/MM/YYYY to YYYY-MM-DD)",
                "from_party": "Complete name of sender organization/person",
                "to_party": "Complete name of recipient organization/person", 
                "subject": "Complete subject line of the letter (don't truncate, include all details)",
                "body": "Complete letter content/body text (extract all paragraphs between header and signature, maintain structure)",
                "main_issue": "Detailed summary of the primary issue or concern discussed",
                "action_required": "Any action requested or required",
                "letter_type": "Type of letter: complaint, request, notice, response, claim, or other",
                "urgency_level": "Urgency level: urgent, normal, or routine",
                "copy_to": "Who else is copied on this letter",
                "key_points": "Key points from the letter as comma-separated list",
                "response_deadline": "Response deadline date in YYYY-MM-DD format if mentioned, null otherwise",
                "extraction_confidence": "Confidence level of extraction as decimal between 0.0 and 1.0"
            }
        elif document_type == DocumentType.MEETING_MINUTES:
            return {
                "meeting_date": "Date of the meeting (extract exact date)",
                "meeting_time": "Time when meeting started",
                "venue": "Meeting venue or location",
                "meeting_type": "Type of meeting (weekly review, milestone review, etc.)",
                "chair_person": "Person who chaired the meeting",
                "secretary": "Meeting secretary or note taker",
                "attendees": "List of ALL meeting attendees with their designations as comma-separated list. Format: 'Name - Designation, Name - Designation'",
                "agenda_items": "Complete list of agenda items discussed, separated by commas",
                "key_discussions": "Summary of key discussions and deliberations in the meeting",
                "decisions_made": "All key decisions and resolutions made during meeting, separated by periods",
                "action_items": "ALL action items assigned to people with deadlines. List each action item clearly",
                "next_meeting_date": "Next meeting date if mentioned",
                "next_meeting_venue": "Next meeting venue if mentioned",
                "meeting_summary": "Overall summary of the meeting outcome",
                "plain_text": "Complete extracted text from the document for reference"
            }
        elif document_type == DocumentType.PROGRESS_REPORTS:
            return {
                "report_period": "Reporting period (month/quarter/year)",
                "report_date": "Date of the report",
                "reporting_period_start": "Start date of reporting period",
                "reporting_period_end": "End date of reporting period",
                "physical_progress": "Physical progress percentage (extract number)",
                "financial_progress": "Financial progress percentage (extract number)", 
                "overall_project_status": "Overall project status (On Track, Delayed, Ahead, etc.)",
                "milestones_achieved": "All milestones achieved during this period, separated by commas",
                "activities_completed": "Major activities completed during this period, separated by commas",
                "key_issues_discussed": "ALL key issues discussed in the report. List each issue clearly separated by periods",
                "challenges_faced": "Challenges and problems encountered during the period",
                "actions_for_stakeholders": "Actions required from various stakeholders (client, authority, contractor), list clearly",
                "upcoming_activities": "Planned activities for next reporting period, separated by commas",
                "planned_milestones": "Upcoming milestones and their target dates",
                "amount_claimed": "Amount claimed in this billing period (extract number only)",
                "amount_certified": "Amount certified by authority (extract number only)",
                "cumulative_expenditure": "Total expenditure till date (extract number only)",
                "executive_summary": "Executive summary or overview of the report",
                "plain_text": "Complete extracted text from the document for reference"
            }
        elif document_type == DocumentType.CHANGE_ORDERS:
            return {
                "change_order_number": "Change order reference number",
                "change_description": "Description of scope change",
                "reason_for_change": "Reason for the modification",
                "cost_impact": "Financial impact of the change",
                "time_impact": "Schedule impact and time extension",
                "approval_status": "Approval status of the change"
            }
        elif document_type == DocumentType.PAYMENT_STATEMENTS:
            return {
                "bill_number": "Bill or invoice number",
                "billing_period": "Period covered by the bill",
                "work_description": "Description of work done",
                "total_amount": "Total bill amount",
                "previous_payment": "Previous payments made",
                "current_due": "Current amount due",
                "deductions": "Any deductions made"
            }
        elif document_type == DocumentType.COURT_ORDERS:
            return {
                "case_number": "Court case number",
                "court_name": "Name of the court",
                "judge_name": "Presiding judge name",
                "order_date": "Date of the order",
                "parties": "Plaintiff and defendant details",
                "order_summary": "Summary of the court order",
                "next_hearing": "Next hearing date if mentioned"
            }
        elif document_type == DocumentType.POLICY_CIRCULARS:
            return {
                "circular_number": "Circular reference number", 
                "issuing_authority": "Authority issuing the circular",
                "circular_date": "Date of issue",
                "subject": "Subject of the circular",
                "policy_details": "Main policy content",
                "effective_date": "Effective date of the policy",
                "compliance_requirements": "Compliance requirements"
            }
        elif document_type == DocumentType.TECHNICAL_DRAWINGS:
            return {
                "drawing_number": "Drawing reference number",
                "drawing_title": "Title of the drawing",
                "scale": "Drawing scale",
                "revision": "Revision number and date",
                "dimensions": "Key dimensions",
                "specifications": "Technical specifications",
                "approval_details": "Drawn by, checked by, approved by"
            }
        
        return None
    
    def _extract_structured_data(self, text: str, extraction_schema: Dict[str, str], context: ProcessingContext) -> ProcessingResult:
        """Extract structured data using LLM service"""
        llm_options = {
            "output_format": context.processing_options.get("output_format", "json"),
            "additional_instructions": context.processing_options.get("llm_instructions"),
            "max_retries": context.processing_options.get("llm_retries", 2)
        }
        
        return self.llm_service.extract_structured_data(text, extraction_schema, **llm_options)
    
    def _classify_issues(self, structured_data: Dict[str, Any], context: ProcessingContext) -> ProcessingResult:
        """Classify issues from structured data"""
        try:
            # Extract potential issues from structured data
            issues = []
            
            # Look for issue-related fields
            issue_fields = ["main_issue", "issue_type", "problem", "concern", "dispute"]
            for field in issue_fields:
                if field in structured_data and structured_data[field]:
                    issues.append(str(structured_data[field]))
            
            # Look for action items that might indicate issues
            if "action_required" in structured_data and structured_data["action_required"]:
                issues.append(f"Action Required: {structured_data['action_required']}")
            
            if not issues:
                return ProcessingResult(
                    status=ProcessingStatus.SUCCESS,
                    data={},
                    metadata={"reason": "No issues found in structured data"}
                )
            
            return self._classify_issues_list(issues, context)
            
        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message=f"Issue classification preparation failed: {str(e)}"
            )
    
    def _classify_issues_list(self, issues: List[str], context: ProcessingContext) -> ProcessingResult:
        """Classify a list of issues"""
        try:
            mapping_options = {
                "use_fuzzy_matching": context.processing_options.get("fuzzy_matching", True),
                "min_confidence": context.processing_options.get("classification_confidence", 0.7)
            }
            
            # Classify each issue
            classifications = {}
            for i, issue in enumerate(issues):
                result = self.category_mapping_service.map_issue_to_category(issue, **mapping_options)
                if result.status == ProcessingStatus.SUCCESS:
                    classifications[f"issue_{i+1}"] = {
                        "issue": issue,
                        "category": result.data,
                        "confidence": result.confidence,
                        "match_type": result.metadata.get("match_type") if result.metadata else "unknown"
                    }
                else:
                    classifications[f"issue_{i+1}"] = {
                        "issue": issue,
                        "category": "Unclassified",
                        "confidence": 0.0,
                        "error": result.error_message
                    }
            
            # Calculate overall confidence
            confidences = [c["confidence"] for c in classifications.values() if "confidence" in c]
            overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=classifications,
                confidence=overall_confidence,
                metadata={
                    "total_issues": len(issues),
                    "classified_issues": len([c for c in classifications.values() if c.get("confidence", 0) > 0])
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message=f"Issue list classification failed: {str(e)}"
            )
    
    def _calculate_overall_confidence(self, results: ProcessingResults) -> float:
        """Calculate overall confidence score"""
        scores = [score for score in results.confidence_scores.values() if score > 0]
        return sum(scores) / len(scores) if scores else 0.0
    
    def _get_services_used(self, context: ProcessingContext) -> List[str]:
        """Get list of services used during processing"""
        services = ["document_type", "ocr"]
        
        if context.processing_options.get("extract_structured_data", True):
            services.append("llm")
        
        if context.processing_options.get("classify_issues", True):
            services.append("category_mapping")
        
        return services
    
    def _create_error_result(self, error_message: str, context: ProcessingContext) -> ProcessingResult:
        """Create standardized error result"""
        return ProcessingResult(
            status=ProcessingStatus.ERROR,
            error_message=error_message,
            metadata={
                "processing_time": (datetime.now() - context.processing_start_time).total_seconds(),
                "context": context.processing_options
            }
        )
    
    def _cleanup_processing_context(self, context: ProcessingContext):
        """Clean up temporary files and resources"""
        if self.cleanup_temp_files and context.temp_files:
            for temp_file in context.temp_files:
                try:
                    Path(temp_file).unlink(missing_ok=True)
                except Exception as e:
                    print(f"Warning: Could not clean up temp file {temp_file}: {e}")
    
    def get_orchestrator_status(self) -> ProcessingResult:
        """Get status of orchestrator and all services"""
        try:
            status_data = {
                "orchestrator_config": {
                    "max_file_size_mb": self.max_file_size_mb,
                    "supported_formats": self.supported_formats,
                    "processing_timeout": self.processing_timeout,
                    "cleanup_temp_files": self.cleanup_temp_files
                },
                "services": {}
            }
            
            # Check each service status
            service_checks = [
                ("document_type", self.document_type_service),
                ("ocr", self.ocr_service), 
                ("llm", self.llm_service),
                ("category_mapping", self.category_mapping_service)
            ]
            
            all_healthy = True
            
            for service_name, service in service_checks:
                try:
                    if hasattr(service, 'get_service_status'):
                        service_status = service.get_service_status()
                        status_data["services"][service_name] = service_status.data
                        if service_status.status != ProcessingStatus.SUCCESS:
                            all_healthy = False
                    elif hasattr(service, 'get_available_methods'):
                        methods_result = service.get_available_methods()
                        status_data["services"][service_name] = {
                            "available": methods_result.status == ProcessingStatus.SUCCESS,
                            "methods": methods_result.data if methods_result.status == ProcessingStatus.SUCCESS else []
                        }
                    else:
                        status_data["services"][service_name] = {"available": True, "status": "unknown"}
                except Exception as e:
                    status_data["services"][service_name] = {"available": False, "error": str(e)}
                    all_healthy = False
            
            return ProcessingResult(
                status=ProcessingStatus.SUCCESS if all_healthy else ProcessingStatus.PARTIAL,
                data=status_data,
                metadata={"total_services": len(service_checks)}
            )
            
        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message=f"Status check failed: {str(e)}"
            )
    
    def update_processing_config(self, new_config: Dict[str, Any]) -> ProcessingResult:
        """Update orchestrator processing configuration"""
        try:
            if "max_file_size_mb" in new_config:
                self.max_file_size_mb = new_config["max_file_size_mb"]
            
            if "processing_timeout" in new_config:
                self.processing_timeout = new_config["processing_timeout"]
            
            if "supported_formats" in new_config:
                self.supported_formats = new_config["supported_formats"]
            
            if "cleanup_temp_files" in new_config:
                self.cleanup_temp_files = new_config["cleanup_temp_files"]
            
            # Update configuration service
            updated_config = {
                "max_file_size_mb": self.max_file_size_mb,
                "processing_timeout": self.processing_timeout,
                "supported_formats": self.supported_formats,
                "cleanup_temp_files": self.cleanup_temp_files
            }
            
            self.config_service.update_service_config("pipeline", updated_config)
            
            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data="Processing configuration updated successfully"
            )
            
        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message=f"Error updating processing configuration: {str(e)}"
            )
    
    def process_document_end_to_end(self, file_path: str, processing_options: Dict[str, Any] = None) -> ProcessingResult:
        """Process document through complete pipeline - implements abstract method"""
        return self.process_document(file_path, processing_options)
    
    def process_with_custom_pipeline(self, file_path: str, pipeline_steps: List[str], **kwargs) -> ProcessingResult:
        """Process document with custom pipeline steps - implements abstract method"""
        # Create processing options from pipeline steps
        processing_options = kwargs.copy()
        processing_options['pipeline_steps'] = pipeline_steps
        
        return self.process_document(file_path, processing_options)