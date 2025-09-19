"""
Service Interface Abstractions
Defines abstract base classes for all document processing services
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass


class DocumentType(Enum):
    """Enumeration of supported document types"""
    CORRESPONDENCE = "correspondence"
    MEETING_MINUTES = "meeting_minutes"
    PROGRESS_REPORTS = "progress_reports"
    CHANGE_ORDERS = "change_orders"
    CONTRACT_AGREEMENTS = "contract_agreements"
    PAYMENT_STATEMENTS = "payment_statements"
    COURT_ORDERS = "court_orders"
    POLICY_CIRCULARS = "policy_circulars"
    TECHNICAL_DRAWINGS = "technical_drawings"
    OTHERS = "others"
    UNKNOWN = "unknown"


class ProcessingStatus(Enum):
    """Status of processing operations"""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"


@dataclass
class ProcessingResult:
    """Standard result format for all processing operations"""
    status: ProcessingStatus
    data: Any = None
    error_message: str = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DocumentMetadata:
    """Document metadata container"""
    file_path: str
    file_name: str = None
    file_size: int = 0
    page_count: int = 0
    mime_type: str = "application/pdf"
    created_at: str = None


class IDocumentTypeService(ABC):
    """Interface for document type classification services"""
    
    @abstractmethod
    def classify_document(self, file_path: str, **kwargs) -> ProcessingResult:
        """
        Classify document type from file
        
        Args:
            file_path: Path to the document file
            **kwargs: Additional classification options
            
        Returns:
            ProcessingResult with DocumentType in data field
        """
        pass
    
    @abstractmethod
    def classify_from_text(self, text_content: str, **kwargs) -> ProcessingResult:
        """
        Classify document type from text content
        
        Args:
            text_content: Raw text content of document
            **kwargs: Additional classification options
            
        Returns:
            ProcessingResult with DocumentType in data field
        """
        pass


class IOCRService(ABC):
    """Interface for OCR and text extraction services"""
    
    @abstractmethod
    def extract_text(self, file_path: str, **kwargs) -> ProcessingResult:
        """
        Extract text from document
        
        Args:
            file_path: Path to the document file
            **kwargs: OCR options (page_range, language, etc.)
            
        Returns:
            ProcessingResult with extracted text in data field
        """
        pass
    
    @abstractmethod
    def extract_text_from_pages(self, file_path: str, page_numbers: List[int], **kwargs) -> ProcessingResult:
        """
        Extract text from specific pages
        
        Args:
            file_path: Path to the document file
            page_numbers: List of page numbers to process
            **kwargs: OCR options
            
        Returns:
            ProcessingResult with dict of {page_num: text} in data field
        """
        pass


class ILLMService(ABC):
    """Interface for LLM-based data extraction services"""
    
    @abstractmethod
    def extract_structured_data(self, text_content: str, extraction_schema: Dict[str, str], **kwargs) -> ProcessingResult:
        """
        Extract structured data using LLM
        
        Args:
            text_content: Text to analyze
            extraction_schema: Dict of field_name -> description
            **kwargs: LLM options (model, temperature, etc.)
            
        Returns:
            ProcessingResult with structured data dict in data field
        """
        pass
    
    @abstractmethod
    def classify_content(self, text_content: str, classification_options: List[str], **kwargs) -> ProcessingResult:
        """
        Classify content into predefined categories
        
        Args:
            text_content: Text to classify
            classification_options: List of possible categories
            **kwargs: Classification options
            
        Returns:
            ProcessingResult with classification result in data field
        """
        pass


class ICategoryMappingService(ABC):
    """Interface for category mapping and issue classification"""
    
    @abstractmethod
    def map_issue_to_category(self, issue_type: str, **kwargs) -> ProcessingResult:
        """
        Map issue type to category
        
        Args:
            issue_type: Issue type to map
            **kwargs: Mapping options
            
        Returns:
            ProcessingResult with mapped category in data field
        """
        pass
    
    @abstractmethod
    def get_available_categories(self, **kwargs) -> ProcessingResult:
        """
        Get list of available categories
        
        Returns:
            ProcessingResult with list of categories in data field
        """
        pass
    
    @abstractmethod
    def get_issue_types_for_category(self, category: str, **kwargs) -> ProcessingResult:
        """
        Get issue types that map to a specific category
        
        Args:
            category: Category name
            
        Returns:
            ProcessingResult with list of issue types in data field
        """
        pass


class IDocumentProcessingOrchestrator(ABC):
    """Interface for coordinating multiple services"""
    
    @abstractmethod
    def process_document_end_to_end(self, file_path: str, processing_options: Dict[str, Any] = None) -> ProcessingResult:
        """
        Process document through complete pipeline
        
        Args:
            file_path: Path to document
            processing_options: Options for processing pipeline
            
        Returns:
            ProcessingResult with complete processing results
        """
        pass
    
    @abstractmethod
    def process_with_custom_pipeline(self, file_path: str, pipeline_steps: List[str], **kwargs) -> ProcessingResult:
        """
        Process document with custom pipeline steps
        
        Args:
            file_path: Path to document
            pipeline_steps: List of processing steps to execute
            
        Returns:
            ProcessingResult with pipeline results
        """
        pass


class IConfigurationService(ABC):
    """Interface for configuration management"""
    
    @abstractmethod
    def get_service_config(self, service_name: str) -> Dict[str, Any]:
        """Get configuration for specific service"""
        pass
    
    @abstractmethod
    def update_service_config(self, service_name: str, config: Dict[str, Any]) -> bool:
        """Update configuration for specific service"""
        pass
    
    @abstractmethod
    def validate_configuration(self) -> ProcessingResult:
        """Validate all service configurations"""
        pass