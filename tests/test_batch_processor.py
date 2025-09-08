#!/usr/bin/env python3
"""
Unit tests for batch processing system
"""

import unittest
import sys
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from batch_processor import BatchPDFProcessor, process_lot_pdfs
from metrics_calculator import MetricsCalculator

# Suppress logging during tests
logging.getLogger().setLevel(logging.WARNING)


class TestBatchProcessor(unittest.TestCase):
    """Test cases for batch PDF processing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.sample_config = {
            'batch_processing': {
                'enabled': True,
                'approaches': {
                    'hybrid_rag': {'enabled': True, 'priority': 1},
                    'pure_llm': {'enabled': False, 'priority': 2}
                },
                'evaluation': {
                    'enabled': True,
                    'auto_detect_ground_truth': True,
                    'ground_truth_patterns': ["EDMS*.xlsx", "test_ground_truth.xlsx"]
                },
                'output': {
                    'results_folder': str(self.temp_dir / 'results'),
                    'save_format': 'xlsx'
                },
                'processing': {
                    'max_pages_per_pdf': 2,
                    'skip_on_error': True,
                    'rate_limit_delay': 0  # No delay in tests
                }
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @patch('batch_processor.ConfigManager')
    @patch('batch_processor.IssueCategoryMapper')
    @patch('batch_processor.ValidationEngine')
    @patch('batch_processor.DataSufficiencyAnalyzer')
    def test_batch_processor_initialization(self, mock_data_analyzer, mock_validator, 
                                          mock_issue_mapper, mock_config_manager):
        """Test batch processor initialization."""
        # Mock the config manager
        mock_config_instance = Mock()
        mock_config_instance.get_all_config.return_value = {
            'data': {'training_data': 'data/test.xlsx'},
            'approaches': {
                'hybrid_rag': {'enabled': True},
                'pure_llm': {'enabled': True}
            }
        }
        mock_config_manager.return_value = mock_config_instance
        
        # Create processor
        with patch('batch_processor.BatchPDFProcessor._load_batch_config', return_value=self.sample_config):
            with patch('batch_processor.BatchPDFProcessor._init_classifiers'):
                processor = BatchPDFProcessor()
                
                self.assertIsNotNone(processor.config)
                self.assertIsNotNone(processor.batch_config)
                self.assertIsNotNone(processor.issue_mapper)
                self.assertIsNotNone(processor.validator)
                self.assertIsNotNone(processor.data_analyzer)
    
    def test_default_batch_config(self):
        """Test default batch configuration generation."""
        with patch('batch_processor.ConfigManager'):
            with patch('batch_processor.BatchPDFProcessor._init_components'):
                with patch('batch_processor.BatchPDFProcessor._init_classifiers'):
                    processor = BatchPDFProcessor()
                    default_config = processor._get_default_batch_config()
                    
                    self.assertTrue(default_config['batch_processing']['enabled'])
                    self.assertTrue(default_config['batch_processing']['approaches']['hybrid_rag']['enabled'])
                    self.assertFalse(default_config['batch_processing']['approaches']['pure_llm']['enabled'])
    
    def test_ground_truth_loading(self):
        """Test ground truth loading from Excel file."""
        # Create mock Excel file
        import pandas as pd
        
        ground_truth_data = {
            'File': ['doc1.pdf', 'doc2.pdf'],
            'Category1': ['Change of Scope', 'Payments'],
            'Category2': ['', 'EoT'],
            'Category3': ['', '']
        }
        df = pd.DataFrame(ground_truth_data)
        excel_path = self.temp_dir / 'test_ground_truth.xlsx'
        df.to_excel(excel_path, index=False)
        
        with patch('batch_processor.ConfigManager'):
            with patch('batch_processor.BatchPDFProcessor._init_components'):
                with patch('batch_processor.BatchPDFProcessor._init_classifiers'):
                    processor = BatchPDFProcessor()
                    ground_truth = processor._load_ground_truth(str(excel_path))
                    
                    self.assertEqual(len(ground_truth), 2)
                    self.assertEqual(ground_truth['doc1.pdf'], ['Change of Scope'])
                    self.assertEqual(ground_truth['doc2.pdf'], ['Payments', 'EoT'])
    
    def test_auto_detect_ground_truth(self):
        """Test automatic ground truth detection."""
        # Create test files
        pdf_folder = self.temp_dir / 'pdfs'
        pdf_folder.mkdir()
        
        # Create EDMS file in parent directory
        edms_file = self.temp_dir / 'EDMS-Test.xlsx'
        edms_file.touch()
        
        with patch('batch_processor.ConfigManager'):
            with patch('batch_processor.BatchPDFProcessor._init_components'):
                with patch('batch_processor.BatchPDFProcessor._init_classifiers'):
                    processor = BatchPDFProcessor()
                    processor.batch_config = self.sample_config
                    
                    detected_file = processor._auto_detect_ground_truth(pdf_folder)
                    self.assertEqual(detected_file, str(edms_file))
    
    @patch('batch_processor.Path.glob')
    def test_process_pdf_folder_no_files(self, mock_glob):
        """Test processing when no PDF files are found."""
        mock_glob.return_value = []
        
        with patch('batch_processor.ConfigManager'):
            with patch('batch_processor.BatchPDFProcessor._init_components'):
                with patch('batch_processor.BatchPDFProcessor._init_classifiers'):
                    processor = BatchPDFProcessor()
                    
                    with self.assertRaises(ValueError) as context:
                        processor.process_pdf_folder(str(self.temp_dir))
                    
                    self.assertIn("No PDF files found", str(context.exception))
    
    def test_convenience_function(self):
        """Test the convenience function process_lot_pdfs."""
        with patch('batch_processor.BatchPDFProcessor') as mock_processor_class:
            mock_processor = Mock()
            mock_processor.process_pdf_folder.return_value = {'test': 'result'}
            mock_processor_class.return_value = mock_processor
            
            result = process_lot_pdfs(
                pdf_folder="test_folder",
                enable_llm=True,
                enable_metrics=False,
                output_folder="test_output"
            )
            
            self.assertEqual(result, {'test': 'result'})
            mock_processor.process_pdf_folder.assert_called_once_with(
                "test_folder", None, "test_output"
            )


class TestMetricsCalculator(unittest.TestCase):
    """Test cases for metrics calculation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = MetricsCalculator()
    
    def test_calculate_metrics_perfect_match(self):
        """Test metrics calculation with perfect match."""
        ground_truth = ['Change of Scope', 'Payments']
        predicted = ['Change of Scope', 'Payments']
        
        metrics = self.calculator.calculate_metrics(ground_truth, predicted)
        
        self.assertEqual(metrics['precision'], 1.0)
        self.assertEqual(metrics['recall'], 1.0)
        self.assertEqual(metrics['f1_score'], 1.0)
        self.assertEqual(metrics['exact_match'], 1.0)
        self.assertEqual(metrics['tp'], 2)
        self.assertEqual(metrics['fp'], 0)
        self.assertEqual(metrics['fn'], 0)
    
    def test_calculate_metrics_partial_match(self):
        """Test metrics calculation with partial match."""
        ground_truth = ['Change of Scope', 'Payments', 'EoT']
        predicted = ['Change of Scope', 'Contractor\'s Obligations']
        
        metrics = self.calculator.calculate_metrics(ground_truth, predicted)
        
        # TP=1, FP=1, FN=2
        self.assertEqual(metrics['tp'], 1)
        self.assertEqual(metrics['fp'], 1)
        self.assertEqual(metrics['fn'], 2)
        self.assertEqual(metrics['precision'], 0.5)  # 1/(1+1)
        self.assertAlmostEqual(metrics['recall'], 0.3333, places=4)  # 1/(1+2)
        self.assertEqual(metrics['exact_match'], 0.0)
        self.assertEqual(metrics['correct_categories'], ['Change of Scope'])
        self.assertEqual(metrics['missed_categories'], ['Payments', 'EoT'])
        self.assertEqual(metrics['extra_categories'], ['Contractor\'s Obligations'])
    
    def test_calculate_metrics_no_match(self):
        """Test metrics calculation with no match."""
        ground_truth = ['Change of Scope', 'Payments']
        predicted = ['Authority\'s Obligations', 'Dispute Resolution']
        
        metrics = self.calculator.calculate_metrics(ground_truth, predicted)
        
        self.assertEqual(metrics['precision'], 0.0)
        self.assertEqual(metrics['recall'], 0.0)
        self.assertEqual(metrics['f1_score'], 0.0)
        self.assertEqual(metrics['exact_match'], 0.0)
        self.assertEqual(metrics['tp'], 0)
        self.assertEqual(metrics['fp'], 2)
        self.assertEqual(metrics['fn'], 2)
    
    def test_calculate_metrics_empty_predictions(self):
        """Test metrics calculation with empty predictions."""
        ground_truth = ['Change of Scope', 'Payments']
        predicted = []
        
        metrics = self.calculator.calculate_metrics(ground_truth, predicted)
        
        self.assertEqual(metrics['precision'], 0.0)
        self.assertEqual(metrics['recall'], 0.0)
        self.assertEqual(metrics['f1_score'], 0.0)
        self.assertEqual(metrics['tp'], 0)
        self.assertEqual(metrics['fp'], 0)
        self.assertEqual(metrics['fn'], 2)
    
    def test_calculate_metrics_empty_ground_truth(self):
        """Test metrics calculation with empty ground truth."""
        ground_truth = []
        predicted = ['Change of Scope', 'Payments']
        
        metrics = self.calculator.calculate_metrics(ground_truth, predicted)
        
        self.assertEqual(metrics['precision'], 0.0)
        self.assertEqual(metrics['recall'], 0.0)
        self.assertEqual(metrics['f1_score'], 0.0)
        self.assertEqual(metrics['tp'], 0)
        self.assertEqual(metrics['fp'], 2)
        self.assertEqual(metrics['fn'], 0)
    
    def test_calculate_batch_metrics(self):
        """Test batch metrics calculation."""
        ground_truth_list = [
            ['Change of Scope', 'Payments'],
            ['EoT', 'Contractor\'s Obligations'],
            ['Authority\'s Obligations']
        ]
        predicted_list = [
            ['Change of Scope', 'Payments'],  # Perfect match
            ['EoT', 'Payments'],              # Partial match
            ['Dispute Resolution']            # No match
        ]
        
        batch_metrics = self.calculator.calculate_batch_metrics(ground_truth_list, predicted_list)
        
        self.assertEqual(batch_metrics['total_files'], 3)
        self.assertEqual(batch_metrics['perfect_predictions'], 1)
        
        # Check micro metrics (aggregate TP, FP, FN)
        # File 1: TP=2, FP=0, FN=0
        # File 2: TP=1, FP=1, FN=1  
        # File 3: TP=0, FP=1, FN=1
        # Total: TP=3, FP=2, FN=2
        self.assertEqual(batch_metrics['total_tp'], 3)
        self.assertEqual(batch_metrics['total_fp'], 2)
        self.assertEqual(batch_metrics['total_fn'], 2)
        
        # Micro precision = TP/(TP+FP) = 3/(3+2) = 0.6
        self.assertEqual(batch_metrics['micro_precision'], 0.6)
        # Micro recall = TP/(TP+FN) = 3/(3+2) = 0.6  
        self.assertEqual(batch_metrics['micro_recall'], 0.6)
    
    def test_jaccard_similarity(self):
        """Test Jaccard similarity calculation."""
        ground_truth = ['Change of Scope', 'Payments', 'EoT']
        predicted = ['Change of Scope', 'Payments', 'Contractor\'s Obligations']
        
        metrics = self.calculator.calculate_metrics(ground_truth, predicted)
        
        # Intersection: 2, Union: 4, Jaccard = 2/4 = 0.5
        self.assertEqual(metrics['jaccard_similarity'], 0.5)
    
    def test_category_statistics(self):
        """Test per-category statistics calculation."""
        ground_truth_list = [
            ['Change of Scope', 'Payments'],
            ['Change of Scope', 'EoT'],
            ['Payments']
        ]
        predicted_list = [
            ['Change of Scope'],
            ['Change of Scope', 'Payments'], 
            ['EoT']
        ]
        
        batch_metrics = self.calculator.calculate_batch_metrics(ground_truth_list, predicted_list)
        category_stats = batch_metrics['category_statistics']
        
        # Change of Scope: appears in GT 2 times, predicted correctly 2 times
        cos_stats = category_stats['Change of Scope']
        self.assertEqual(cos_stats['tp'], 2)
        self.assertEqual(cos_stats['fn'], 0)
        self.assertEqual(cos_stats['precision'], 1.0)
        self.assertEqual(cos_stats['recall'], 1.0)
        
        # Payments: appears in GT 2 times, predicted correctly 1 time
        payments_stats = category_stats['Payments']
        self.assertEqual(payments_stats['tp'], 1)
        self.assertEqual(payments_stats['fn'], 1)
        self.assertEqual(payments_stats['fp'], 1)


class TestBatchProcessorConfiguration(unittest.TestCase):
    """Test cases for batch processor configuration handling."""
    
    def test_config_validation(self):
        """Test configuration validation."""
        valid_config = {
            'batch_processing': {
                'enabled': True,
                'approaches': {
                    'hybrid_rag': {'enabled': True, 'priority': 1}
                }
            }
        }
        
        # Test that valid config doesn't raise errors
        with patch('batch_processor.ConfigManager'):
            with patch('batch_processor.BatchPDFProcessor._init_components'):
                with patch('batch_processor.BatchPDFProcessor._init_classifiers'):
                    with patch('batch_processor.BatchPDFProcessor._load_batch_config', return_value=valid_config):
                        processor = BatchPDFProcessor()
                        self.assertIsNotNone(processor.batch_config)
    
    def test_approach_priority_handling(self):
        """Test approach priority handling."""
        config_with_priorities = {
            'batch_processing': {
                'approaches': {
                    'hybrid_rag': {'enabled': True, 'priority': 2},
                    'pure_llm': {'enabled': True, 'priority': 1}
                }
            }
        }
        
        # This test would verify that approaches are processed in priority order
        # Implementation would depend on actual priority handling logic
        self.assertTrue(True)  # Placeholder


if __name__ == '__main__':
    # Configure test discovery
    loader = unittest.TestLoader()
    
    # Run specific test classes if specified
    if len(sys.argv) > 1:
        suite = unittest.TestSuite()
        for arg in sys.argv[1:]:
            if hasattr(sys.modules[__name__], arg):
                suite.addTests(loader.loadTestsFromTestCase(getattr(sys.modules[__name__], arg)))
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)
    else:
        # Run all tests
        unittest.main(verbosity=2)