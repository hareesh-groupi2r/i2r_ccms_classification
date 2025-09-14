#!/usr/bin/env python3
"""
Unified PDF Processing System
Provides consistent processing pipeline for both single and batch PDF operations
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import json
from datetime import datetime

from classifier.config_manager import ConfigManager
from classifier.issue_mapper import IssueCategoryMapper
from classifier.validation import ValidationEngine
from classifier.data_sufficiency import DataSufficiencyAnalyzer
from classifier.pure_llm import PureLLMClassifier
from classifier.hybrid_rag import HybridRAGClassifier
from classifier.pdf_extractor import PDFExtractor
from classifier.category_normalizer import CategoryNormalizer
from extract_correspondence_content import CorrespondenceExtractor

logger = logging.getLogger(__name__)


class UnifiedPDFProcessor:
    """
    Unified PDF processing system that provides consistent pipeline for both single and batch operations.
    
    Features:
    - Consistent text preprocessing with CorrespondenceExtractor
    - Direct classifier instantiation (no API dependency)
    - Unified configuration management
    - Comprehensive error handling
    - Standardized result formatting
    """
    
    def __init__(self, config_path: str = None, batch_config_path: str = None):
        """
        Initialize the unified processor.
        
        Args:
            config_path: Path to main configuration file (config.yaml)
            batch_config_path: Path to batch configuration file (optional)
        """
        # Load configurations
        self.config_manager = ConfigManager(config_path or "config.yaml")
        self.config = self.config_manager.get_all_config()
        
        # Initialize components
        self._init_components()
        
        # Initialize classifiers based on configuration
        self.classifiers = {}
        self._init_classifiers()
        
        logger.info("UnifiedPDFProcessor initialized")
    
    def _init_components(self):
        """Initialize shared components."""
        training_data_path = Path(self.config['data']['training_data'])
        
        self.issue_mapper = IssueCategoryMapper(training_data_path)
        self.validator = ValidationEngine(training_data_path)
        self.data_analyzer = DataSufficiencyAnalyzer(training_data_path)
        self.pdf_extractor = PDFExtractor()
        self.correspondence_extractor = CorrespondenceExtractor()
        self.category_normalizer = CategoryNormalizer(strict_mode=False)
        
        logger.info("Unified processor components initialized")
    
    def _init_classifiers(self):
        """Initialize classifiers based on configuration."""
        enabled_approaches = self.config_manager.get_enabled_approaches()
        
        # Initialize Hybrid RAG if enabled
        if 'hybrid_rag' in enabled_approaches:
            config = self.config_manager.get_approach_config('hybrid_rag')
            classifier = HybridRAGClassifier(
                config=config,
                issue_mapper=self.issue_mapper,
                validator=self.validator,
                data_analyzer=self.data_analyzer
            )
            
            # Build or load index if needed
            index_path = Path('./data/embeddings/rag_index')
            if not index_path.with_suffix('.faiss').exists():
                logger.info("Building vector index for RAG approach...")
                training_path = self.config['data']['training_data']
                classifier.build_index(training_path, save_path=str(index_path))
                logger.info("Vector index built and saved")
            
            self.classifiers['hybrid_rag'] = classifier
            logger.info("✅ Hybrid RAG classifier initialized")
        
        # Initialize Pure LLM if enabled
        if 'pure_llm' in enabled_approaches:
            config = self.config_manager.get_approach_config('pure_llm')
            if config.get('api_key'):
                self.classifiers['pure_llm'] = PureLLMClassifier(
                    config=config,
                    issue_mapper=self.issue_mapper,
                    validator=self.validator,
                    data_analyzer=self.data_analyzer
                )
                logger.info("✅ Pure LLM classifier initialized")
            else:
                logger.warning("⚠️ Pure LLM classifier skipped - no API key")
        
        if not self.classifiers:
            raise ValueError("No classifiers enabled! Please enable at least one approach.")
        
        logger.info(f"🚀 Initialized {len(self.classifiers)} classifiers: {list(self.classifiers.keys())}")
    
    def process_single_pdf(self, 
                          pdf_path: str, 
                          approaches: List[str] = None,
                          confidence_threshold: float = 0.3,
                          max_pages: int = None) -> Dict:
        """
        Process a single PDF file with unified pipeline.
        
        Args:
            pdf_path: Path to PDF file
            approaches: List of approaches to use (default: all enabled)
            confidence_threshold: Minimum confidence threshold for results
            max_pages: Maximum pages to extract from PDF
            
        Returns:
            Unified processing results
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if approaches is None:
            approaches = list(self.classifiers.keys())
        
        # Validate requested approaches are available
        invalid_approaches = set(approaches) - set(self.classifiers.keys())
        if invalid_approaches:
            raise ValueError(f"Invalid approaches: {invalid_approaches}. Available: {list(self.classifiers.keys())}")
        
        logger.info(f"📄 Processing PDF: {pdf_path.name}")
        logger.info(f"📋 Using approaches: {approaches}")
        
        result = {
            'file_name': pdf_path.name,
            'file_path': str(pdf_path),
            'status': 'completed',
            'processing_time': 0,
            'approaches': {},
            'unified_results': {
                'categories': [],
                'issues': [],
                'confidence_score': 0.0
            },
            'extraction_info': {},
            'configuration': {
                'approaches_used': approaches,
                'confidence_threshold': confidence_threshold,
                'max_pages': max_pages
            }
        }
        
        start_time = time.time()
        
        try:
            # Step 1: Extract PDF content
            if max_pages:
                self.pdf_extractor = PDFExtractor(max_pages=max_pages)
            
            raw_text, extraction_method = self.pdf_extractor.extract_text(pdf_path)
            
            # Step 2: Extract correspondence content (standardized preprocessing)
            extraction_result = self.correspondence_extractor.extract_correspondence_content(raw_text)
            focused_content = f"Subject: {extraction_result['subject']}\n\nContent: {extraction_result['body']}"
            
            result['extraction_info'] = {
                'raw_length': len(raw_text),
                'focused_length': len(focused_content),
                'extraction_method': extraction_method,
                'correspondence_method': extraction_result['extraction_method'],
                'subject': extraction_result['subject'],
                'body': extraction_result['body']
            }
            
            # Step 3: Process with each requested approach
            all_categories = {}  # For unified results
            all_issues = {}     # For unified results
            
            for approach_name in approaches:
                classifier = self.classifiers[approach_name]
                logger.info(f"  🔍 Classifying with {approach_name.replace('_', ' ').title()}...")
                
                approach_start = time.time()
                approach_result = classifier.classify(focused_content, is_file_path=False)
                approach_time = time.time() - approach_start
                
                # Handle classification results
                if approach_result.get('status') == 'error':
                    # Handle errors (including LLM validation failures)
                    result['approaches'][approach_name] = {
                        'status': 'error',
                        'error_type': approach_result.get('error_type', 'unknown'),
                        'error_message': approach_result.get('message', 'Unknown error'),
                        'processing_time': approach_time,
                        'categories': [],
                        'issues': []
                    }
                    logger.warning(f"    ❌ {approach_name}: {approach_result.get('message', 'Unknown error')}")
                else:
                    # Filter results by confidence threshold
                    filtered_categories = [
                        cat for cat in approach_result.get('categories', [])
                        if cat.get('confidence', 0) >= confidence_threshold
                    ]
                    
                    filtered_issues = [
                        issue for issue in approach_result.get('identified_issues', [])
                        if issue.get('confidence', 0) >= confidence_threshold
                    ]
                    
                    result['approaches'][approach_name] = {
                        'status': 'success',
                        'categories': filtered_categories,
                        'issues': filtered_issues,
                        'processing_time': approach_time,
                        'provider_used': approach_result.get('llm_provider_used', 'unknown'),
                        'full_result': approach_result
                    }
                    
                    # Collect for unified results
                    for cat in filtered_categories:
                        cat_name = cat.get('category', '')
                        if cat_name not in all_categories or all_categories[cat_name]['confidence'] < cat.get('confidence', 0):
                            all_categories[cat_name] = {
                                'category': cat_name,
                                'confidence': cat.get('confidence', 0),
                                'source_approach': approach_name
                            }
                    
                    for issue in filtered_issues:
                        issue_type = issue.get('issue_type', '')
                        if issue_type not in all_issues or all_issues[issue_type]['confidence'] < issue.get('confidence', 0):
                            all_issues[issue_type] = {
                                'issue_type': issue_type,
                                'confidence': issue.get('confidence', 0),
                                'source_approach': approach_name
                            }
                    
                    logger.info(f"    ✅ {approach_name}: {len(filtered_categories)} categories, {len(filtered_issues)} issues")
            
            # Step 4: Create unified results (best results across all approaches)
            result['unified_results']['categories'] = sorted(
                list(all_categories.values()),
                key=lambda x: x['confidence'],
                reverse=True
            )
            
            result['unified_results']['issues'] = sorted(
                list(all_issues.values()),
                key=lambda x: x['confidence'],
                reverse=True
            )
            
            # Calculate overall confidence score
            all_confidences = []
            all_confidences.extend([cat['confidence'] for cat in result['unified_results']['categories']])
            all_confidences.extend([issue['confidence'] for issue in result['unified_results']['issues']])
            
            result['unified_results']['confidence_score'] = max(all_confidences) if all_confidences else 0.0
            
            result['processing_time'] = time.time() - start_time
            
            logger.info(f"📊 Processing completed: {len(result['unified_results']['categories'])} categories, "
                      f"{len(result['unified_results']['issues'])} issues "
                      f"(confidence: {result['unified_results']['confidence_score']:.3f})")
            
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
            result['processing_time'] = time.time() - start_time
            logger.error(f"❌ Processing failed: {e}")
            raise
        
        return result
    
    def process_batch_pdfs(self, 
                          pdf_folder: str,
                          approaches: List[str] = None,
                          confidence_threshold: float = 0.3,
                          max_pages: int = None,
                          ground_truth_file: str = None,
                          output_folder: str = None) -> Dict:
        """
        Process multiple PDFs using unified pipeline.
        
        Args:
            pdf_folder: Path to folder containing PDFs
            approaches: List of approaches to use
            confidence_threshold: Minimum confidence threshold
            max_pages: Maximum pages to extract per PDF
            ground_truth_file: Optional ground truth for evaluation
            output_folder: Output folder for results
            
        Returns:
            Batch processing results
        """
        pdf_folder = Path(pdf_folder)
        if not pdf_folder.exists():
            raise FileNotFoundError(f"PDF folder not found: {pdf_folder}")
        
        pdf_files = list(pdf_folder.glob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in {pdf_folder}")
        
        logger.info(f"📦 Processing {len(pdf_files)} PDFs from {pdf_folder}")
        
        batch_results = {
            'folder': str(pdf_folder),
            'total_files': len(pdf_files),
            'processed_files': 0,
            'failed_files': 0,
            'results': [],
            'configuration': {
                'approaches': approaches or list(self.classifiers.keys()),
                'confidence_threshold': confidence_threshold,
                'max_pages': max_pages
            },
            'processing_stats': {
                'start_time': datetime.now(),
                'end_time': None,
                'total_processing_time': 0
            }
        }
        
        start_time = time.time()
        
        for i, pdf_file in enumerate(pdf_files, 1):
            logger.info(f"📄 [{i}/{len(pdf_files)}] Processing: {pdf_file.name}")
            
            try:
                result = self.process_single_pdf(
                    str(pdf_file),
                    approaches=approaches,
                    confidence_threshold=confidence_threshold,
                    max_pages=max_pages
                )
                batch_results['results'].append(result)
                batch_results['processed_files'] += 1
                
            except Exception as e:
                logger.error(f"❌ Failed to process {pdf_file.name}: {e}")
                batch_results['failed_files'] += 1
                
                # Add failed result placeholder
                batch_results['results'].append({
                    'file_name': pdf_file.name,
                    'status': 'failed',
                    'error': str(e),
                    'approaches': {}
                })
        
        batch_results['processing_stats']['end_time'] = datetime.now()
        batch_results['processing_stats']['total_processing_time'] = time.time() - start_time
        
        logger.info(f"🎉 Batch processing completed: {batch_results['processed_files']}/{batch_results['total_files']} successful")
        
        # Save results if output folder specified
        if output_folder:
            self._save_batch_results(batch_results, output_folder)
        
        return batch_results
    
    def _save_batch_results(self, results: Dict, output_folder: str):
        """Save batch results to files."""
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_path = output_path / f"unified_batch_results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save Excel results
        excel_path = output_path / f"unified_batch_results_{timestamp}.xlsx"
        self._save_results_excel(results, excel_path)
        
        logger.info(f"💾 Results saved to: {json_path} and {excel_path}")
    
    def _save_results_excel(self, results: Dict, excel_path: Path):
        """Save results to Excel format."""
        summary_data = []
        
        for result in results['results']:
            if result['status'] == 'completed':
                row = {
                    'File Name': result['file_name'],
                    'Subject': result.get('extraction_info', {}).get('subject', ''),
                    'Body': result.get('extraction_info', {}).get('body', ''),
                    'Processing Time (s)': f"{result['processing_time']:.2f}",
                    'Unified Categories': ', '.join([cat['category'] for cat in result['unified_results']['categories']]),
                    'Unified Issues': ', '.join([issue['issue_type'] for issue in result['unified_results']['issues']]),
                    'Confidence Score': f"{result['unified_results']['confidence_score']:.3f}",
                    'Approaches Used': ', '.join(result['configuration']['approaches_used'])
                }
            else:
                row = {
                    'File Name': result['file_name'],
                    'Status': result['status'],
                    'Error': result.get('error', '')
                }
            
            summary_data.append(row)
        
        # Create Excel file
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Summary sheet
            summary_df = pd.DataFrame(summary_data)
            summary_df = summary_df.sort_values('File Name')
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Processing stats
            stats_data = [{
                'Metric': k.replace('_', ' ').title(),
                'Value': str(v)
            } for k, v in results['processing_stats'].items()]
            
            stats_df = pd.DataFrame(stats_data)
            stats_df.to_excel(writer, sheet_name='Processing Stats', index=False)
    
    def get_available_approaches(self) -> List[str]:
        """Get list of available classification approaches."""
        return list(self.classifiers.keys())
    
    def get_system_info(self) -> Dict:
        """Get system information."""
        return {
            'available_approaches': self.get_available_approaches(),
            'total_issue_types': len(self.issue_mapper.get_all_issue_types()),
            'total_categories': len(self.issue_mapper.get_all_categories()),
            'training_samples': len(self.data_analyzer.df) if hasattr(self.data_analyzer, 'df') else 0
        }