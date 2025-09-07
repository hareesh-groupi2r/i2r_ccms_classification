"""
Issue-Category Mapper Module
Maps identified issue types to their corresponding categories based on training data
"""

from typing import Dict, List, Set, Tuple
import pandas as pd
from collections import defaultdict
import json
from pathlib import Path
import logging
from .category_normalizer import CategoryNormalizer
from .issue_normalizer import IssueTypeNormalizer

logger = logging.getLogger(__name__)


class IssueCategoryMapper:
    """
    Builds and manages the mapping between issue types and categories.
    Handles the many-to-many relationship where one issue type can map to multiple categories.
    """
    
    def __init__(self, training_data_path: str = None):
        """
        Initialize the mapper with training data.
        
        Args:
            training_data_path: Path to the Excel file containing training data
        """
        self.issue_to_categories = defaultdict(set)
        self.category_frequencies = defaultdict(int)
        self.issue_frequencies = defaultdict(int)
        self.stats = {}
        
        # Initialize normalizers
        self.category_normalizer = CategoryNormalizer(strict_mode=False)
        self.issue_normalizer = IssueTypeNormalizer()
        
        if training_data_path:
            self._build_mapping(training_data_path)
    
    def _build_mapping(self, data_path: str):
        """
        Analyze training data to build issue-category relationships.
        
        Args:
            data_path: Path to the training data Excel file
        """
        try:
            df = pd.read_excel(data_path)
            logger.info(f"Loaded {len(df)} training samples from {data_path}")
            
            for _, row in df.iterrows():
                issue_type_raw = str(row['issue_type']).strip()
                categories_str = str(row['category'])
                
                # Normalize issue type first
                issue_type, status, confidence = self.issue_normalizer.normalize_issue_type(issue_type_raw)
                if status != 'exact' and status != 'rejected':
                    logger.debug(f"Normalized issue type '{issue_type_raw}' to '{issue_type}' (status: {status})")
                
                if not issue_type:
                    logger.warning(f"Row: Invalid issue type '{issue_type_raw}' was rejected")
                    continue
                
                # Use the category normalizer to handle all category variations
                categories = self.category_normalizer.parse_and_normalize_categories(categories_str)
                
                # Log any normalization issues
                if not categories:
                    logger.warning(f"Row: No valid categories found in '{categories_str}' for issue '{issue_type}'")
                
                # Build mapping dictionary
                self.issue_to_categories[issue_type].update(categories)
                
                # Track frequencies for confidence scoring
                self.issue_frequencies[issue_type] += 1
                for category in categories:
                    self.category_frequencies[category] += 1
            
            # Convert sets to lists for consistency
            self.issue_to_categories = {
                issue: list(cats) 
                for issue, cats in self.issue_to_categories.items()
            }
            
            # Calculate mapping statistics
            self.calculate_mapping_stats()
            
            logger.info(f"Built mapping for {len(self.issue_to_categories)} issue types "
                       f"and {len(self.category_frequencies)} categories")
            
            # Log normalization statistics
            cat_stats = self.category_normalizer.get_stats()
            issue_stats = self.issue_normalizer.get_stats()
            
            if cat_stats['rejected'] > 0 or cat_stats['issue_type_as_category'] > 0:
                logger.warning(f"Category normalization: {cat_stats['rejected']} rejected, "
                              f"{cat_stats['issue_type_as_category']} issue types used as categories")
            
            if issue_stats['normalized'] > 0:
                logger.info(f"Issue type normalization: {issue_stats['normalized']} normalized out of {issue_stats['total_processed']} processed")
            
        except Exception as e:
            logger.error(f"Error building mapping from {data_path}: {e}")
            raise
    
    def get_categories_for_issue(self, issue_type: str, 
                                 confidence_threshold: float = 0.0) -> List[Tuple[str, float]]:
        """
        Get categories for a given issue type with confidence scores.
        
        Args:
            issue_type: The issue type to look up
            confidence_threshold: Minimum confidence to include a category
            
        Returns:
            List of tuples (category, confidence_score)
        """
        if issue_type not in self.issue_to_categories:
            logger.warning(f"Issue type '{issue_type}' not found in mapping")
            return []
        
        categories = self.issue_to_categories[issue_type]
        
        # Calculate confidence based on frequency in training data
        results = []
        for category in categories:
            confidence = self._calculate_confidence(issue_type, category)
            if confidence >= confidence_threshold:
                results.append((category, confidence))
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def _calculate_confidence(self, issue_type: str, category: str) -> float:
        """
        Calculate confidence score for issue-category mapping.
        
        Args:
            issue_type: The issue type
            category: The category
            
        Returns:
            Confidence score between 0 and 1
        """
        issue_freq = self.issue_frequencies.get(issue_type, 0)
        category_freq = self.category_frequencies.get(category, 0)
        
        if issue_freq == 0 or category_freq == 0:
            return 0.0
        
        # Base confidence on co-occurrence frequency
        base_confidence = 0.7  # Base confidence for known mappings
        
        # Boost confidence for frequently seen patterns
        freq_boost = min(0.3, issue_freq / 100)
        
        return min(1.0, base_confidence + freq_boost)
    
    def calculate_mapping_stats(self):
        """
        Generate statistics about issue-category mappings.
        
        Returns:
            Dictionary containing mapping statistics
        """
        if not self.issue_to_categories:
            self.stats = {
                'total_issue_types': 0,
                'total_unique_categories': 0,
                'avg_categories_per_issue': 0,
                'max_categories_per_issue': 0,
                'min_categories_per_issue': 0
            }
            return self.stats
        
        all_categories = set()
        category_counts = []
        
        for issue, categories in self.issue_to_categories.items():
            all_categories.update(categories)
            category_counts.append(len(categories))
        
        self.stats = {
            'total_issue_types': len(self.issue_to_categories),
            'total_unique_categories': len(all_categories),
            'avg_categories_per_issue': sum(category_counts) / len(category_counts),
            'max_categories_per_issue': max(category_counts),
            'min_categories_per_issue': min(category_counts),
            'issues_with_single_category': sum(1 for c in category_counts if c == 1),
            'issues_with_multiple_categories': sum(1 for c in category_counts if c > 1)
        }
        
        logger.info(f"Mapping statistics: {self.stats}")
        return self.stats
    
    def map_issues_to_categories(self, identified_issues: List[Dict]) -> List[Dict]:
        """
        Map identified issues to their categories with aggregated confidence.
        
        Args:
            identified_issues: List of dicts with 'issue_type' and 'confidence'
        
        Returns:
            List of dicts with 'category', 'confidence', 'source_issues'
        """
        category_scores = defaultdict(lambda: {'confidence': 0, 'sources': []})
        
        for issue in identified_issues:
            issue_type = issue.get('issue_type')
            issue_confidence = issue.get('confidence', 1.0)
            
            # Get categories for this issue
            categories = self.get_categories_for_issue(issue_type)
            
            for category, mapping_confidence in categories:
                # Combine confidences: issue identification * mapping confidence
                combined_confidence = issue_confidence * mapping_confidence
                
                # Keep the highest confidence if multiple issues map to same category
                if category_scores[category]['confidence'] < combined_confidence:
                    category_scores[category]['confidence'] = combined_confidence
                
                category_scores[category]['sources'].append({
                    'issue_type': issue_type,
                    'confidence': combined_confidence
                })
        
        # Convert to list format
        results = [
            {
                'category': category,
                'confidence': data['confidence'],
                'source_issues': data['sources']
            }
            for category, data in category_scores.items()
        ]
        
        return sorted(results, key=lambda x: x['confidence'], reverse=True)
    
    def save_mapping(self, output_path: str):
        """
        Save the mapping to a JSON file for later use.
        
        Args:
            output_path: Path where to save the mapping
        """
        mapping_data = {
            'issue_to_categories': self.issue_to_categories,
            'statistics': self.stats,
            'issue_frequencies': dict(self.issue_frequencies),
            'category_frequencies': dict(self.category_frequencies)
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(mapping_data, f, indent=2)
        
        logger.info(f"Saved mapping to {output_path}")
    
    def load_mapping(self, mapping_path: str):
        """
        Load a previously saved mapping from JSON file.
        
        Args:
            mapping_path: Path to the saved mapping file
        """
        try:
            with open(mapping_path, 'r') as f:
                mapping_data = json.load(f)
            
            self.issue_to_categories = mapping_data['issue_to_categories']
            self.stats = mapping_data['statistics']
            self.issue_frequencies = defaultdict(int, mapping_data.get('issue_frequencies', {}))
            self.category_frequencies = defaultdict(int, mapping_data.get('category_frequencies', {}))
            
            logger.info(f"Loaded mapping from {mapping_path}")
            
        except Exception as e:
            logger.error(f"Error loading mapping from {mapping_path}: {e}")
            raise
    
    def get_all_issue_types(self) -> List[str]:
        """
        Get list of all known issue types.
        
        Returns:
            List of issue type strings
        """
        return list(self.issue_to_categories.keys())
    
    def get_all_categories(self) -> List[str]:
        """
        Get list of all known categories.
        
        Returns:
            List of category strings
        """
        return list(self.category_frequencies.keys())
    
    def __repr__(self):
        return (f"IssueCategoryMapper(issues={len(self.issue_to_categories)}, "
                f"categories={len(self.category_frequencies)})")