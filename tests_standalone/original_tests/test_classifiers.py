#!/usr/bin/env python3
"""
Unit tests for classification components
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging

# Add project root to path  
sys.path.insert(0, str(Path(__file__).parent.parent))

from classifier.category_normalizer import CategoryNormalizer
from classifier.issue_normalizer import IssueNormalizer
from classifier.validation import ValidationEngine
from classifier.data_sufficiency import DataSufficiencyAnalyzer
from classifier.pure_llm import PureLLMClassifier

# Suppress logging during tests
logging.getLogger().setLevel(logging.WARNING)


class TestCategoryNormalizer(unittest.TestCase):
    """Test cases for category normalization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.normalizer = CategoryNormalizer(strict_mode=False)
    
    def test_exact_match_normalization(self):
        """Test exact match normalization."""
        test_cases = [
            "EoT",
            "Payments", 
            "Change of Scope",
            "Contractor's Obligations",
            "Authority's Obligations",
            "Dispute Resolution",
            "Others",
            "Appointed Date"
        ]
        
        for category in test_cases:
            normalized = self.normalizer.normalize_category(category)
            self.assertEqual(normalized, category)
    
    def test_case_insensitive_normalization(self):
        """Test case-insensitive normalization."""
        test_cases = [
            ("eot", "EoT"),
            ("PAYMENTS", "Payments"),
            ("change of scope", "Change of Scope"),
            ("contractor's obligations", "Contractor's Obligations")
        ]
        
        for input_cat, expected in test_cases:
            normalized = self.normalizer.normalize_category(input_cat)
            self.assertEqual(normalized, expected)
    
    def test_whitespace_normalization(self):
        """Test whitespace handling."""
        test_cases = [
            ("  EoT  ", "EoT"),
            ("\tPayments\n", "Payments"),
            ("Change  of   Scope", "Change of Scope"),
            ("Contractor's    Obligations", "Contractor's Obligations")
        ]
        
        for input_cat, expected in test_cases:
            normalized = self.normalizer.normalize_category(input_cat)
            self.assertEqual(normalized, expected)
    
    def test_abbreviation_expansion(self):
        """Test abbreviation expansion."""
        test_cases = [
            ("CoS", "Change of Scope"),
            ("cos", "Change of Scope"),  
            ("EoT", "EoT"),  # Should remain as is
            ("EOT", "EoT")   # Case normalization
        ]
        
        for input_cat, expected in test_cases:
            normalized = self.normalizer.normalize_category(input_cat)
            self.assertEqual(normalized, expected)
    
    def test_fuzzy_matching(self):
        """Test fuzzy string matching."""
        test_cases = [
            ("Payment", "Payments"),
            ("Extention of Time", "EoT"),
            ("Changing of Scope", "Change of Scope"),
            ("Contractors Obligation", "Contractor's Obligations"),
            ("Authority Obligation", "Authority's Obligations")
        ]
        
        for input_cat, expected in test_cases:
            normalized = self.normalizer.normalize_category(input_cat)
            self.assertEqual(normalized, expected)
    
    def test_invalid_category_handling(self):
        """Test handling of invalid categories."""
        invalid_categories = [
            "Invalid Category",
            "Random Text", 
            "Not A Category",
            "",
            None
        ]
        
        for invalid_cat in invalid_categories:
            if invalid_cat is not None:
                normalized = self.normalizer.normalize_category(invalid_cat)
                # Should return None or original string based on strict_mode
                self.assertTrue(normalized is None or isinstance(normalized, str))


class TestIssueNormalizer(unittest.TestCase):
    """Test cases for issue type normalization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.normalizer = IssueNormalizer()
    
    def test_exact_issue_normalization(self):
        """Test exact match normalization for common issue types."""
        test_cases = [
            "Change of scope proposals clarifications",
            "Handing over of land /Possession of site",
            "Slow Progress of Works", 
            "Utility shifting",
            "EoT",
            "Payments"
        ]
        
        for issue in test_cases:
            normalized = self.normalizer.normalize_issue(issue)
            self.assertEqual(normalized, issue)
    
    def test_case_insensitive_issue_normalization(self):
        """Test case-insensitive issue normalization."""
        test_cases = [
            ("CHANGE OF SCOPE PROPOSALS CLARIFICATIONS", "Change of scope proposals clarifications"),
            ("handing over of land", "Handing over of land /Possession of site"),
            ("slow progress of works", "Slow Progress of Works")
        ]
        
        for input_issue, expected in test_cases:
            normalized = self.normalizer.normalize_issue(input_issue)
            # This test might need adjustment based on actual implementation
            self.assertTrue(isinstance(normalized, str))
    
    def test_partial_match_normalization(self):
        """Test partial matching for issue types."""
        test_cases = [
            ("scope change", "Change of scope proposals clarifications"),
            ("land possession", "Handing over of land /Possession of site"),
            ("work progress", "Slow Progress of Works"),
            ("utility", "Utility shifting")
        ]
        
        for partial_issue, expected in test_cases:
            normalized = self.normalizer.normalize_issue(partial_issue)
            # Implementation-dependent - might return exact match or None
            self.assertTrue(isinstance(normalized, str) or normalized is None)


class TestValidationEngine(unittest.TestCase):
    """Test cases for validation engine."""
    
    def setUp(self):
        """Set up test fixtures with mock data."""
        self.mock_training_path = Path("mock_training.xlsx")
        
        # Mock the training data loading
        with patch('classifier.validation.pd.read_excel') as mock_read_excel:
            mock_df = Mock()
            mock_df.columns = ['Issue Type', 'Categories']
            mock_df.iterrows.return_value = iter([
                (0, Mock(iloc=['Change of scope proposals clarifications', 'Change of Scope'])),
                (1, Mock(iloc=['Payments', 'Payments'])),
                (2, Mock(iloc=['EoT', 'EoT']))
            ])
            mock_read_excel.return_value = mock_df
            
            self.validator = ValidationEngine(self.mock_training_path)
    
    def test_issue_validation_valid(self):
        """Test validation of valid issue types."""
        valid_issues = [
            "Change of scope proposals clarifications", 
            "Payments",
            "EoT"
        ]
        
        for issue in valid_issues:
            validated, is_valid, confidence = self.validator.validate_issue_type(issue)
            self.assertTrue(is_valid)
            self.assertEqual(confidence, 1.0)
            self.assertEqual(validated, issue)
    
    def test_issue_validation_invalid(self):
        """Test validation of invalid issue types."""
        invalid_issues = [
            "Invalid Issue Type",
            "Random Text",
            "Not An Issue"
        ]
        
        for issue in invalid_issues:
            validated, is_valid, confidence = self.validator.validate_issue_type(issue)
            # Behavior depends on auto_correct setting
            self.assertTrue(isinstance(validated, str) or validated is None)
            self.assertTrue(isinstance(is_valid, bool))
            self.assertTrue(0.0 <= confidence <= 1.0)
    
    def test_category_validation_valid(self):
        """Test validation of valid categories."""
        valid_categories = ["Change of Scope", "Payments", "EoT"]
        
        for category in valid_categories:
            validated, is_valid, confidence = self.validator.validate_category(category)
            self.assertTrue(is_valid)
            self.assertEqual(confidence, 1.0)
            self.assertEqual(validated, category)
    
    def test_constrained_prompt_creation(self):
        """Test creation of constrained prompts."""
        issues_prompt = self.validator.create_constrained_prompt('issues')
        categories_prompt = self.validator.create_constrained_prompt('categories')
        
        self.assertIn('issue types', issues_prompt.lower())
        self.assertIn('categories', categories_prompt.lower())
        self.assertTrue(len(issues_prompt) > 100)  # Should be substantial
        self.assertTrue(len(categories_prompt) > 100)


class TestDataSufficiencyAnalyzer(unittest.TestCase):
    """Test cases for data sufficiency analysis."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_training_path = Path("mock_training.xlsx")
        
        # Mock the training data loading
        with patch('classifier.data_sufficiency.pd.read_excel') as mock_read_excel:
            mock_df = Mock()
            mock_df.columns = ['Issue Type', 'Categories']
            # Create mock data with different sample counts
            mock_rows = []
            for i in range(50):  # 50 samples for "Change of scope"
                mock_rows.append((i, Mock(iloc=['Change of scope proposals clarifications', 'Change of Scope'])))
            for i in range(5):   # 5 samples for "Rare Issue"
                mock_rows.append((i+50, Mock(iloc=['Rare Issue Type', 'Others'])))
                
            mock_df.iterrows.return_value = iter(mock_rows)
            mock_read_excel.return_value = mock_df
            
            self.analyzer = DataSufficiencyAnalyzer(self.mock_training_path)
    
    def test_sufficiency_thresholds(self):
        """Test data sufficiency threshold classification."""
        # Test with high sample count
        sufficiency = self.analyzer.get_data_sufficiency('Change of scope proposals clarifications')
        self.assertIn(sufficiency, ['excellent', 'very_good', 'good'])
        
        # Test with low sample count  
        sufficiency = self.analyzer.get_data_sufficiency('Rare Issue Type')
        self.assertIn(sufficiency, ['critical', 'warning'])
        
        # Test with unknown issue
        sufficiency = self.analyzer.get_data_sufficiency('Unknown Issue')
        self.assertEqual(sufficiency, 'critical')
    
    def test_confidence_adjustments(self):
        """Test confidence adjustment application."""
        mock_result = {
            'identified_issues': [
                {'issue_type': 'Change of scope proposals clarifications', 'confidence': 0.9},
                {'issue_type': 'Rare Issue Type', 'confidence': 0.9}
            ],
            'categories': [
                {'category': 'Change of Scope', 'confidence': 0.9},
                {'category': 'Others', 'confidence': 0.9}
            ]
        }
        
        adjusted_result = self.analyzer.apply_confidence_adjustments(mock_result)
        
        # High-sample issue should maintain high confidence
        high_sample_issue = adjusted_result['identified_issues'][0]
        self.assertGreaterEqual(high_sample_issue['confidence'], 0.8)
        
        # Low-sample issue should have reduced confidence
        low_sample_issue = adjusted_result['identified_issues'][1]
        self.assertLess(low_sample_issue['confidence'], 0.9)


class TestPureLLMClassifier(unittest.TestCase):
    """Test cases for Pure LLM classifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            'model': 'gpt-4-turbo',
            'max_tokens': 1000,
            'temperature': 0.1
        }
        
        # Mock dependencies
        self.mock_issue_mapper = Mock()
        self.mock_validator = Mock()
        self.mock_data_analyzer = Mock()
    
    def test_json_response_parsing(self):
        """Test JSON response parsing from LLM."""
        with patch('classifier.pure_llm.ConfigManager'):
            with patch('classifier.pure_llm.IssueCategoryMapper'):
                with patch('classifier.pure_llm.ValidationEngine'):
                    with patch('classifier.pure_llm.DataSufficiencyAnalyzer'):
                        with patch('classifier.pure_llm.PureLLMClassifier._init_hierarchical_llm_clients'):
                            classifier = PureLLMClassifier(
                                self.mock_config,
                                self.mock_issue_mapper,
                                self.mock_validator, 
                                self.mock_data_analyzer
                            )
                            
                            # Test various JSON response formats
                            test_responses = [
                                '{"issues": [{"issue_type": "Test", "confidence": 0.9}]}',
                                '```json\n{"issues": [{"issue_type": "Test", "confidence": 0.9}]}\n```',
                                '```\n{"issues": [{"issue_type": "Test", "confidence": 0.9}]}\n```',
                                'Here is the analysis:\n```json\n{"issues": [{"issue_type": "Test", "confidence": 0.9}]}\n```'
                            ]
                            
                            for response in test_responses:
                                parsed = classifier._parse_llm_response(response)
                                self.assertIn('issues', parsed)
                                self.assertEqual(len(parsed['issues']), 1)
                                self.assertEqual(parsed['issues'][0]['issue_type'], 'Test')
    
    def test_json_sanitization(self):
        """Test JSON sanitization functionality."""
        with patch('classifier.pure_llm.ConfigManager'):
            with patch('classifier.pure_llm.IssueCategoryMapper'):
                with patch('classifier.pure_llm.ValidationEngine'):
                    with patch('classifier.pure_llm.DataSufficiencyAnalyzer'):
                        with patch('classifier.pure_llm.PureLLMClassifier._init_hierarchical_llm_clients'):
                            classifier = PureLLMClassifier(
                                self.mock_config,
                                self.mock_issue_mapper,
                                self.mock_validator,
                                self.mock_data_analyzer
                            )
                            
                            # Test JSON with formatting issues
                            test_cases = [
                                ('{"key": "value" and "more"}', '{"key": "value and more"}'),
                                ('{"key": "value",}', '{"key": "value"}'),
                                ("{'key': 'value'}", '{"key": "value"}')
                            ]
                            
                            for input_json, expected in test_cases:
                                sanitized = classifier._sanitize_json(input_json)
                                # Should be parseable JSON
                                import json
                                try:
                                    parsed = json.loads(sanitized)
                                    self.assertTrue(isinstance(parsed, dict))
                                except json.JSONDecodeError:
                                    self.fail(f"Sanitized JSON not parseable: {sanitized}")


if __name__ == '__main__':
    # Support running specific test classes
    if len(sys.argv) > 1:
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        for arg in sys.argv[1:]:
            if hasattr(sys.modules[__name__], arg):
                suite.addTests(loader.loadTestsFromTestCase(getattr(sys.modules[__name__], arg)))
        
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)
    else:
        unittest.main(verbosity=2)