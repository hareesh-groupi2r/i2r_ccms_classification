#!/usr/bin/env python3
"""
Analyze LOT-21 documents to capture LLM prompts for hallucination analysis
"""

import sys
import logging
from pathlib import Path
import time

# Add the classification system to path
sys.path.insert(0, str(Path(__file__).parent))

from classifier.config_manager import ConfigManager
from classifier.issue_mapper import IssueCategoryMapper  
from classifier.validation import ValidationEngine
from classifier.data_sufficiency import DataSufficiencyAnalyzer
from classifier.hybrid_rag import HybridRAGClassifier

# Configure logging to capture our analysis
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lot21_prompt_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def analyze_lot21_documents():
    """Process 2 LOT-21 documents to capture LLM prompts and vector search results"""
    
    logger.info("="*100)
    logger.info("STARTING LOT-21 DOCUMENT ANALYSIS FOR HALLUCINATION INVESTIGATION")
    logger.info("="*100)
    
    # Documents to analyze
    doc_paths = [
        "data/Lots21-27/Lot 21 to 23/LOT-21/1. AE to SPK ltr no 1963 dt 16.09.2017.pdf",
        "data/Lots21-27/Lot 21 to 23/LOT-21/10. 2917 - Final Payment Statement Submission (EPC10).pdf"
    ]
    
    try:
        # Initialize the classification system
        logger.info("Initializing Hybrid RAG Classification System...")
        
        config_manager = ConfigManager()
        if not config_manager.validate_config():
            raise RuntimeError("Configuration validation failed")
        
        # Determine training data path
        training_paths = [
            'data/synthetic/combined_training_data.xlsx',
            'data/raw/Consolidated_labeled_data.xlsx'
        ]
        
        training_data_path = None
        for path in training_paths:
            if Path(path).exists():
                training_data_path = path
                break
        
        if not training_data_path:
            raise FileNotFoundError(f"Training data not found in: {training_paths}")
        
        logger.info(f"Using training data: {training_data_path}")
        
        # Initialize core components
        issue_mapper = IssueCategoryMapper(training_data_path)
        validator = ValidationEngine(training_data_path)
        data_analyzer = DataSufficiencyAnalyzer(training_data_path)
        
        # Initialize hybrid RAG classifier
        config = config_manager.get_approach_config('hybrid_rag')
        classifier = HybridRAGClassifier(
            config=config,
            issue_mapper=issue_mapper,
            validator=validator,
            data_analyzer=data_analyzer
        )
        
        # Build or load index
        index_path = Path('data/embeddings/rag_index')
        if not index_path.with_suffix('.faiss').exists():
            logger.info("Building vector index...")
            classifier.build_index(training_data_path, save_path=str(index_path))
        else:
            logger.info("Using existing vector index")
        
        # Process each document
        for i, doc_path in enumerate(doc_paths, 1):
            logger.info(f"\n{'#'*100}")
            logger.info(f"PROCESSING DOCUMENT {i}/2: {doc_path}")
            logger.info(f"{'#'*100}")
            
            if not Path(doc_path).exists():
                logger.error(f"Document not found: {doc_path}")
                continue
            
            start_time = time.time()
            
            # Classify the document
            result = classifier.classify(doc_path, is_file_path=True)
            
            processing_time = time.time() - start_time
            
            logger.info(f"\n{'*'*80}")
            logger.info(f"DOCUMENT {i} CLASSIFICATION RESULTS")
            logger.info(f"{'*'*80}")
            logger.info(f"Status: {result.get('status', 'unknown')}")
            logger.info(f"Processing Time: {processing_time:.2f} seconds")
            logger.info(f"Extraction Method: {result.get('extraction_method', 'unknown')}")
            logger.info(f"Search Results Used: {result.get('search_results_used', 0)}")
            
            if 'identified_issues' in result:
                logger.info(f"\nFinal Identified Issues: {len(result['identified_issues'])}")
                for j, issue in enumerate(result['identified_issues'][:5], 1):
                    logger.info(f"  {j}. {issue.get('issue_type', 'Unknown')} "
                              f"(confidence: {issue.get('confidence', 0):.3f})")
            
            if 'categories' in result:
                logger.info(f"\nFinal Categories: {len(result['categories'])}")
                for j, cat in enumerate(result['categories'][:5], 1):
                    logger.info(f"  {j}. {cat.get('category', 'Unknown')} "
                              f"(confidence: {cat.get('confidence', 0):.3f})")
            
            logger.info(f"{'*'*80}")
            
            # Add delay between documents to separate logs clearly
            time.sleep(2)
        
        logger.info(f"\n{'='*100}")
        logger.info("LOT-21 DOCUMENT ANALYSIS COMPLETED")
        logger.info("="*100)
        logger.info("Check 'lot21_prompt_analysis.log' for detailed LLM prompts and responses")
        logger.info("="*100)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    analyze_lot21_documents()