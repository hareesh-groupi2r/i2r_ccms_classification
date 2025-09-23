"""
Hybrid RAG Classification Service
Integrates the existing hybrid RAG classification system into the CCMS backend
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Add the classification system to path
classification_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(classification_path))

from .interfaces import ProcessingResult, ProcessingStatus
from .configuration_service import get_config_service

# Import hybrid RAG system components
try:
    from classifier.config_manager import ConfigManager
    from classifier.unified_issue_mapper import UnifiedIssueCategoryMapper  
    from classifier.validation import ValidationEngine
    from classifier.data_sufficiency import DataSufficiencyAnalyzer
    from classifier.hybrid_rag import HybridRAGClassifier
    from classifier.pure_llm import PureLLMClassifier
except ImportError as e:
    print(f"Warning: Could not import classification components: {e}")
    ConfigManager = None
    UnifiedIssueCategoryMapper = None
    ValidationEngine = None
    DataSufficiencyAnalyzer = None
    HybridRAGClassifier = None
    PureLLMClassifier = None

logger = logging.getLogger(__name__)


class HybridRAGClassificationService:
    """Service for hybrid RAG document classification integrated into CCMS backend"""
    
    def __init__(self, config_service=None):
        """Initialize the classification service"""
        self.config_service = config_service or get_config_service()
        self.service_config = self.config_service.get_service_config("hybrid_rag_classification")
        
        # Classification system components
        self.config_manager = None
        self.issue_mapper = None
        self.validator = None
        self.data_analyzer = None
        self.hybrid_rag_classifier = None
        self.pure_llm_classifier = None
        
        # Service state
        self.is_initialized = False
        self.initialization_error = None
        
        # Initialize if components are available
        if all([ConfigManager, UnifiedIssueCategoryMapper, ValidationEngine, DataSufficiencyAnalyzer]):
            try:
                self._initialize_classification_system()
            except Exception as e:
                logger.error(f"Failed to initialize classification system: {e}")
                self.initialization_error = str(e)
        else:
            self.initialization_error = "Classification system components not available"
    
    def _initialize_classification_system(self):
        """Initialize the hybrid RAG classification system"""
        logger.info("Initializing Hybrid RAG Classification System...")
        
        # Initialize configuration manager
        self.config_manager = ConfigManager()
        if not self.config_manager.validate_config():
            raise RuntimeError("Classification system configuration validation failed")
        
        # Determine training data path with priority order
        training_paths = [
            # Priority 1: Enhanced training data with synthetic samples
            str(classification_path / 'data' / 'synthetic' / 'enhanced_training_claude_*.xlsx'),
            str(classification_path / 'data' / 'synthetic' / 'enhanced_training_priority_*.xlsx'),
            # Priority 2: Combined training data (original + existing synthetic)
            str(classification_path / 'data' / 'synthetic' / 'combined_training_data.xlsx'),
            # Priority 3: Raw consolidated data (fallback)
            str(classification_path / 'data' / 'raw' / 'Consolidated_labeled_data.xlsx')
        ]
        
        training_data_path = None
        for path_pattern in training_paths:
            if '*' in path_pattern:
                # Handle wildcard patterns - get the most recent
                import glob
                matching_files = glob.glob(path_pattern)
                if matching_files:
                    # Sort by modification time, get most recent
                    training_data_path = max(matching_files, key=lambda x: Path(x).stat().st_mtime)
                    break
            elif Path(path_pattern).exists():
                training_data_path = path_pattern
                break
        
        if not training_data_path:
            raise FileNotFoundError(f"Training data not found in expected locations: {training_paths}")
        
        logger.info(f"Using training data: {training_data_path}")
        logger.info(f"Training data file size: {Path(training_data_path).stat().st_size:,} bytes")
        
        # Initialize core components with unified mapper
        unified_mapping_path = str(classification_path / 'issue_category_mapping_diffs' / 'unified_issue_category_mapping.xlsx')
        
        # Verify unified mapping file exists
        if not Path(unified_mapping_path).exists():
            raise FileNotFoundError(f"Unified mapping file not found: {unified_mapping_path}")
        
        self.issue_mapper = UnifiedIssueCategoryMapper(
            training_data_path=training_data_path,
            mapping_file_path=unified_mapping_path
        )
        self.validator = ValidationEngine(training_data_path)
        self.data_analyzer = DataSufficiencyAnalyzer(training_data_path)
        
        # Sync ValidationEngine with complete issue mapper (critical for LLM prompts)
        self.validator.sync_with_issue_mapper(self.issue_mapper)
        
        logger.info(f"🔄 Loaded {len(self.issue_mapper.get_all_issue_types())} issue types from unified mapper")
        logger.info(f"🔄 Loaded {len(self.issue_mapper.get_all_categories())} categories")
        logger.info(f"🔄 ValidationEngine synced with {len(self.validator.valid_issue_types)} issue types")
        
        # Initialize classifiers based on enabled approaches
        enabled_approaches = self.config_manager.get_enabled_approaches()
        logger.info(f"Enabled approaches: {enabled_approaches}")
        
        if 'hybrid_rag' in enabled_approaches:
            config = self.config_manager.get_approach_config('hybrid_rag')
            
            # Fix: Ensure correct absolute path for index location
            absolute_index_path = str(classification_path / 'data' / 'embeddings' / 'rag_index')
            config['index_path'] = absolute_index_path
            
            self.hybrid_rag_classifier = HybridRAGClassifier(
                config=config,
                issue_mapper=self.issue_mapper,
                validator=self.validator,
                data_analyzer=self.data_analyzer
            )
            
            # Build or load vector index with comprehensive checks
            index_path = classification_path / 'data' / 'embeddings' / 'rag_index'
            faiss_file = index_path.with_suffix('.faiss')
            pkl_file = index_path.with_suffix('.pkl')
            
            # Check if index exists and is valid
            index_needs_rebuild = False
            if not faiss_file.exists() or not pkl_file.exists():
                index_needs_rebuild = True
                logger.info("📊 Vector index files missing - will build fresh index")
            else:
                # Check if training data is newer than index
                training_modified = Path(training_data_path).stat().st_mtime
                index_modified = faiss_file.stat().st_mtime
                if training_modified > index_modified:
                    index_needs_rebuild = True
                    logger.info("📊 Training data is newer than vector index - will rebuild")
                else:
                    logger.info("📊 Using existing vector index")
            
            if index_needs_rebuild:
                logger.info("🔨 Building vector index for RAG approach...")
                # Ensure embeddings directory exists
                index_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Build the index
                self.hybrid_rag_classifier.build_index(training_data_path, save_path=str(index_path))
                
                # Verify the built index
                if faiss_file.exists() and pkl_file.exists():
                    # Load and check the index
                    import pickle
                    with open(str(pkl_file), 'rb') as f:
                        metadata = pickle.load(f)
                    
                    if isinstance(metadata, dict) and 'texts' in metadata:
                        doc_count = len(metadata['texts'])
                    elif isinstance(metadata, list):
                        doc_count = len(metadata)
                    else:
                        doc_count = "unknown"
                    
                    logger.info(f"✅ Vector index built successfully with {doc_count} documents")
                    logger.info(f"📁 Index files: {faiss_file.stat().st_size:,} bytes (FAISS), {pkl_file.stat().st_size:,} bytes (metadata)")
                else:
                    logger.error("❌ Vector index build failed - files not created")
                    raise RuntimeError("Vector index build failed")
        
        if 'pure_llm' in enabled_approaches:
            config = self.config_manager.get_approach_config('pure_llm')
            if config.get('api_key'):
                self.pure_llm_classifier = PureLLMClassifier(
                    config=config,
                    issue_mapper=self.issue_mapper,
                    validator=self.validator,
                    data_analyzer=self.data_analyzer
                )
                logger.info("Pure LLM classifier initialized")
            else:
                logger.warning("Pure LLM classifier skipped - no API key configured")
        
        self.is_initialized = True
        logger.info("Hybrid RAG Classification System initialized successfully")
    
    def classify_document_by_id(self, document_id: str, **kwargs) -> ProcessingResult:
        """
        Classify a document by fetching its content from the database
        
        Args:
            document_id: Document ID to classify
            **kwargs: Classification options
                - approach: 'hybrid_rag' or 'pure_llm' (default: 'hybrid_rag')
                - confidence_threshold: Minimum confidence (default: 0.5)
                - max_results: Maximum results to return (default: 5)
                - include_justification: Include RAG evidence (default: True)
                - include_issue_types: Include issue types (default: True)
        
        Returns:
            ProcessingResult with classification data
        """
        if not self.is_initialized:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message=f"Classification service not initialized: {self.initialization_error}"
            )
        
        start_time = time.time()
        approach = kwargs.get('approach', 'hybrid_rag')
        
        try:
            # TODO: Fetch document content from Supabase database
            # For now, return error indicating this needs database integration
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message="Document database integration not yet implemented. Use classify_text method instead.",
                metadata={'document_id': document_id, 'approach': approach}
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Document classification failed for ID {document_id}: {e}")
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message=f"Document classification failed: {str(e)}",
                metadata={
                    'document_id': document_id,
                    'processing_time': processing_time,
                    'approach': approach
                }
            )
    
    def _format_issue_centric_response(self, raw_result, filtered_categories, filtered_issues, 
                                      approach, processing_time, overall_confidence):
        """
        Convert category-centric classification result to issue-centric format
        
        Uses the actual validated issues and categories from the hybrid RAG system
        to create an issue → categories mapping
        """
        issues = []
        issue_to_data = {}
        
        # Method 1: Use source_issues from filtered_categories (if available)
        for cat_info in filtered_categories:
            category = cat_info.get('category', '')
            justification = cat_info.get('justification', '')
            
            # Process source issues for this category
            for source_issue in cat_info.get('source_issues', []):
                issue_type = source_issue.get('issue_type', '')
                issue_confidence = source_issue.get('confidence', 0.0)
                issue_source = source_issue.get('source', 'unknown')
                evidence = source_issue.get('evidence', '')
                
                if issue_type not in issue_to_data:
                    issue_to_data[issue_type] = {
                        'categories': [],
                        'confidence': issue_confidence,
                        'justification': evidence or justification,
                        'source': issue_source
                    }
                
                # Add category to this issue (avoid duplicates)
                if category not in issue_to_data[issue_type]['categories']:
                    issue_to_data[issue_type]['categories'].append(category)
                
                # Use the highest confidence and most detailed justification
                if issue_confidence > issue_to_data[issue_type]['confidence']:
                    issue_to_data[issue_type]['confidence'] = issue_confidence
                    if evidence and len(evidence) > len(issue_to_data[issue_type]['justification']):
                        issue_to_data[issue_type]['justification'] = evidence
        
        # Method 2: Use issue mapper to get categories for each identified issue (backup method)
        if not issue_to_data and self.issue_mapper:
            for issue_info in filtered_issues:
                issue_type = issue_info.get('issue_type', '')
                issue_confidence = issue_info.get('confidence', 0.0)
                issue_source = issue_info.get('source', 'unknown')
                
                # Get categories for this issue from the mapper
                try:
                    categories = self.issue_mapper.get_categories_for_issue(issue_type)
                except:
                    categories = []  # Fallback if mapper method doesn't exist
                
                issue_to_data[issue_type] = {
                    'categories': categories,
                    'confidence': issue_confidence,
                    'justification': f"Identified {issue_type} issue with {issue_confidence:.0%} confidence",
                    'source': issue_source
                }
        
        # Method 3: Fallback - just use identified issues even without categories
        if not issue_to_data:
            for issue_info in filtered_issues:
                issue_type = issue_info.get('issue_type', '')
                issue_to_data[issue_type] = {
                    'categories': [],
                    'confidence': issue_info.get('confidence', 0.0),
                    'justification': '',
                    'source': issue_info.get('source', 'unknown')
                }
        
        # Convert to list format
        for issue_type, data in issue_to_data.items():
            issues.append({
                'issue_type': issue_type,
                'categories': data['categories'],
                'confidence': data['confidence'],
                'justification': data['justification'],
                'source': data['source']
            })
        
        # Sort by confidence descending
        issues.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'approach_used': approach,
            'processing_time': processing_time,
            'confidence_score': overall_confidence,
            'issues': issues,
            'total_issues': len(issues),
            'total_categories': len(set(cat for issue in issues for cat in issue['categories'])),
            'data_sufficiency_warnings': raw_result.get('data_sufficiency_warnings', []),
            'validation_report': raw_result.get('validation_report', {}),
            'llm_provider_used': raw_result.get('llm_provider_used')
        }

    def classify_text(self, subject: str, body: str = "", **kwargs) -> ProcessingResult:
        """
        Classify text content directly
        
        Args:
            subject: Subject/title of the document
            body: Body content of the document
            **kwargs: Classification options (same as classify_document_by_id)
        
        Returns:
            ProcessingResult with classification data
        """
        if not self.is_initialized:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message=f"Classification service not initialized: {self.initialization_error}"
            )
        
        start_time = time.time()
        approach = kwargs.get('approach', 'hybrid_rag')
        confidence_threshold = kwargs.get('confidence_threshold', 0.5)
        max_results = kwargs.get('max_results', 5)
        include_justification = kwargs.get('include_justification', True)
        include_issue_types = kwargs.get('include_issue_types', True)
        response_format = kwargs.get('format', 'category_centric')  # 'category_centric' or 'issue_centric'
        
        try:
            # Combine subject and body for classification
            text_content = f"Subject: {subject}\n\nBody: {body}" if body else f"Subject: {subject}"
            
            # Get appropriate classifier
            if approach == 'hybrid_rag' and self.hybrid_rag_classifier:
                classifier = self.hybrid_rag_classifier
            elif approach == 'pure_llm' and self.pure_llm_classifier:
                classifier = self.pure_llm_classifier
            else:
                available = []
                if self.hybrid_rag_classifier:
                    available.append('hybrid_rag')
                if self.pure_llm_classifier:
                    available.append('pure_llm')
                
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    error_message=f"Approach '{approach}' not available. Available: {available}"
                )
            
            # Perform classification
            raw_result = classifier.classify(text_content)
            
            # Filter and format results
            filtered_categories = []
            for cat_info in raw_result.get('categories', []):
                if cat_info.get('confidence', 0) >= confidence_threshold:
                    category_data = {
                        'category': cat_info.get('category', ''),
                        'confidence': cat_info.get('confidence', 0.0)
                    }
                    
                    if include_justification:
                        category_data['justification'] = cat_info.get('evidence', '')
                    
                    if include_issue_types:
                        category_data['issue_types'] = cat_info.get('issue_types', [])
                        category_data['source_issues'] = cat_info.get('source_issues', [])
                    
                    filtered_categories.append(category_data)
            
            # Limit results
            filtered_categories = filtered_categories[:max_results]
            
            # Filter identified issues
            filtered_issues = [
                {
                    'issue_type': issue.get('issue_type', ''),
                    'confidence': issue.get('confidence', 0.0),
                    'source': issue.get('source', 'unknown')
                }
                for issue in raw_result.get('identified_issues', [])
                if issue.get('confidence', 0) >= confidence_threshold
            ][:max_results]
            
            # Calculate overall confidence
            overall_confidence = max(
                [cat['confidence'] for cat in filtered_categories] +
                [issue['confidence'] for issue in filtered_issues] +
                [0.0]
            )
            
            processing_time = time.time() - start_time
            
            # Format result based on requested format
            if response_format == 'issue_centric':
                classification_data = self._format_issue_centric_response(
                    raw_result, filtered_categories, filtered_issues, 
                    approach, processing_time, overall_confidence
                )
            else:
                # Default category-centric format
                classification_data = {
                'approach_used': approach,
                'processing_time': processing_time,
                'confidence_score': overall_confidence,
                'categories': filtered_categories,
                'identified_issues': filtered_issues,
                'data_sufficiency_warnings': raw_result.get('data_sufficiency_warnings', []),
                'validation_report': raw_result.get('validation_report', {}),
                'llm_provider_used': raw_result.get('llm_provider_used')
            }
            
            logger.info(f"Classification completed: {len(filtered_categories)} categories, "
                       f"{len(filtered_issues)} issues, confidence: {overall_confidence:.3f}")
            
            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=classification_data,
                confidence=overall_confidence,
                metadata={
                    'text_length': len(text_content),
                    'processing_time': processing_time,
                    'approach': approach
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Text classification failed: {e}")
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message=f"Text classification failed: {str(e)}",
                metadata={
                    'processing_time': processing_time,
                    'approach': approach,
                    'text_length': len(f"{subject} {body}")
                }
            )
    
    def classify_batch(self, texts: List[Dict[str, str]], **kwargs) -> ProcessingResult:
        """
        Classify multiple texts in batch
        
        Args:
            texts: List of dicts with 'subject' and optional 'body' keys
            **kwargs: Classification options
        
        Returns:
            ProcessingResult with batch classification data
        """
        if not self.is_initialized:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message=f"Classification service not initialized: {self.initialization_error}"
            )
        
        start_time = time.time()
        results = []
        failed_count = 0
        
        for i, text_data in enumerate(texts):
            try:
                subject = text_data.get('subject', '')
                body = text_data.get('body', '')
                
                result = self.classify_text(subject, body, **kwargs)
                results.append({
                    'index': i,
                    'subject': subject,
                    'result': result.data if result.status == ProcessingStatus.SUCCESS else None,
                    'error': result.error_message if result.status == ProcessingStatus.ERROR else None,
                    'status': result.status.value
                })
                
                if result.status == ProcessingStatus.ERROR:
                    failed_count += 1
                    
            except Exception as e:
                results.append({
                    'index': i,
                    'subject': text_data.get('subject', ''),
                    'result': None,
                    'error': str(e),
                    'status': 'error'
                })
                failed_count += 1
        
        total_processing_time = time.time() - start_time
        success_count = len(results) - failed_count
        
        batch_data = {
            'total_items': len(texts),
            'successful_items': success_count,
            'failed_items': failed_count,
            'total_processing_time': total_processing_time,
            'average_processing_time': total_processing_time / len(texts) if texts else 0,
            'results': results
        }
        
        return ProcessingResult(
            status=ProcessingStatus.SUCCESS if failed_count == 0 else ProcessingStatus.PARTIAL,
            data=batch_data,
            metadata={
                'batch_size': len(texts),
                'success_rate': success_count / len(texts) if texts else 0
            }
        )
    
    def get_available_categories(self) -> ProcessingResult:
        """Get list of available categories"""
        if not self.is_initialized:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message=f"Classification service not initialized: {self.initialization_error}"
            )
        
        try:
            categories = list(self.issue_mapper.get_all_categories())
            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data={
                    'categories': sorted(categories),
                    'total_count': len(categories),
                    'standard_categories': [
                        "EoT", "Dispute Resolution", "Contractor's Obligations",
                        "Payments", "Authority's Obligations", "Change of Scope",
                        "Others", "Appointed Date"
                    ]
                }
            )
        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message=f"Failed to get categories: {str(e)}"
            )
    
    def get_available_issue_types(self) -> ProcessingResult:
        """Get list of available issue types"""
        if not self.is_initialized:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message=f"Classification service not initialized: {self.initialization_error}"
            )
        
        try:
            issue_types = list(self.issue_mapper.get_all_issue_types())
            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data={
                    'issue_types': sorted(issue_types),
                    'total_count': len(issue_types)
                }
            )
        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message=f"Failed to get issue types: {str(e)}"
            )
    
    def get_service_status(self) -> ProcessingResult:
        """Get classification service status and statistics"""
        status_data = {
            'service_name': 'Hybrid RAG Classification Service',
            'is_initialized': self.is_initialized,
            'initialization_error': self.initialization_error,
            'available_approaches': []
        }
        
        if self.is_initialized:
            if self.hybrid_rag_classifier:
                status_data['available_approaches'].append('hybrid_rag')
            if self.pure_llm_classifier:
                status_data['available_approaches'].append('pure_llm')
            
            status_data.update({
                'total_issue_types': len(self.issue_mapper.get_all_issue_types()),
                'total_categories': len(self.issue_mapper.get_all_categories()),
                'training_data_samples': len(self.data_analyzer.df) if hasattr(self.data_analyzer, 'df') else 0
            })
        
        return ProcessingResult(
            status=ProcessingStatus.SUCCESS if self.is_initialized else ProcessingStatus.ERROR,
            data=status_data
        )


# Service instance factory
_classification_service_instance = None

def get_classification_service(config_service=None):
    """Get or create the classification service instance"""
    global _classification_service_instance
    if _classification_service_instance is None:
        _classification_service_instance = HybridRAGClassificationService(config_service)
    return _classification_service_instance