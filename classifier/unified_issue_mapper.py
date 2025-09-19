"""
Unified Issue-Category Mapper Module
Uses both training data and explicit mapping file for comprehensive issue→category mapping
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


class UnifiedIssueCategoryMapper:
    """
    Enhanced mapper that combines training data with explicit mapping file.
    Provides comprehensive issue-to-category mapping with precedence to explicit mappings.
    """
    
    def __init__(self, training_data_path: str = None, mapping_file_path: str = None):
        """
        Initialize the unified mapper.
        
        Args:
            training_data_path: Path to the Excel file containing training data
            mapping_file_path: Path to the explicit issue→category mapping file
        """
        self.issue_to_categories = defaultdict(list)
        self.category_frequencies = defaultdict(int)
        self.issue_frequencies = defaultdict(int)
        self.stats = {}
        
        # Initialize normalizers
        self.category_normalizer = CategoryNormalizer(strict_mode=False)
        self.issue_normalizer = IssueTypeNormalizer()
        
        # Build unified mapping
        self._build_unified_mapping(training_data_path, mapping_file_path)
    
    def _build_unified_mapping(self, training_data_path: str = None, mapping_file_path: str = None):
        """
        Build unified mapping from both training data and explicit mapping file.
        
        Args:
            training_data_path: Path to the training data Excel file
            mapping_file_path: Path to the explicit mapping file
        """
        unified_mapping = defaultdict(set)
        
        # Step 1: Load from training data
        if training_data_path and Path(training_data_path).exists():
            logger.info(f"Loading issues from training data: {training_data_path}")
            try:
                df = pd.read_excel(training_data_path)
                training_count = 0
                
                for _, row in df.iterrows():
                    issue_type_raw = str(row['issue_type']).strip()
                    categories_str = str(row['category'])
                    
                    # Normalize issue type
                    issue_type, status, confidence = self.issue_normalizer.normalize_issue_type(issue_type_raw)
                    if not issue_type:
                        continue
                    
                    # Parse and normalize categories
                    categories = self.category_normalizer.parse_and_normalize_categories(categories_str)
                    category_names = [cat[0] if isinstance(cat, tuple) else cat for cat in categories]
                    
                    unified_mapping[issue_type].update(category_names)
                    self.issue_frequencies[issue_type] += 1
                    training_count += 1
                
                logger.info(f"Loaded {training_count} issue mappings from training data")
                
            except Exception as e:
                logger.error(f"Error loading training data from {training_data_path}: {e}")
        
        # Step 2: Load from explicit mapping file (TAKES PRECEDENCE)
        if not mapping_file_path:
            # Default to unified mapping file if available
            project_root = Path(__file__).parent.parent
            default_mapping = project_root / "unified_issue_category_mapping.xlsx"
            if default_mapping.exists():
                mapping_file_path = str(default_mapping)
        
        if mapping_file_path and Path(mapping_file_path).exists():
            logger.info(f"Loading explicit mappings from: {mapping_file_path}")
            try:
                df = pd.read_excel(mapping_file_path)
                explicit_count = 0
                
                for _, row in df.iterrows():
                    issue_type = str(row['Issue_type']).strip()
                    categories_str = str(row['Categories'])
                    
                    if pd.notna(categories_str) and categories_str != 'nan':
                        # Parse comma-separated categories
                        categories = [cat.strip() for cat in categories_str.split(',')]
                        
                        # EXPLICIT MAPPING TAKES PRECEDENCE
                        # Replace any training data mapping with explicit mapping
                        unified_mapping[issue_type] = set(categories)
                        explicit_count += 1
                
                logger.info(f"Loaded {explicit_count} explicit issue mappings (overriding training data)")
                
            except Exception as e:
                logger.error(f"Error loading explicit mappings from {mapping_file_path}: {e}")
        
        # Convert to final format
        self.issue_to_categories = {
            issue: list(categories) 
            for issue, categories in unified_mapping.items() 
            if categories
        }
        
        # Calculate frequencies for confidence scoring
        for issue, categories in self.issue_to_categories.items():
            for category in categories:
                self.category_frequencies[category] += 1
        
        # Calculate stats
        self.calculate_mapping_stats()
        
        logger.info(f"Built unified mapping: {len(self.issue_to_categories)} issues → {len(self.category_frequencies)} categories")
    
    def get_categories_for_issue(self, issue_type: str, confidence_threshold: float = 0.0) -> List[Tuple[str, float]]:
        """
        Get categories for a given issue type with confidence scores.
        
        Args:
            issue_type: The issue type to look up
            confidence_threshold: Minimum confidence to include a category
            
        Returns:
            List of tuples (category, confidence_score)
        """
        # Try direct lookup first
        if issue_type in self.issue_to_categories:
            categories = self.issue_to_categories[issue_type]
            actual_issue_type = issue_type
        else:
            # Try normalization if direct lookup fails
            normalized_result = self.issue_normalizer.normalize_issue_type(issue_type)
            if normalized_result and len(normalized_result) >= 2:
                normalized_issue_type = normalized_result[0]
                if normalized_issue_type in self.issue_to_categories:
                    categories = self.issue_to_categories[normalized_issue_type]
                    actual_issue_type = normalized_issue_type
                    logger.info(f"Issue normalization helped: '{issue_type}' -> '{normalized_issue_type}'")
                else:
                    logger.warning(f"Issue type '{issue_type}' not found in unified mapping (normalized: '{normalized_issue_type}')")
                    return []
            else:
                logger.warning(f"Issue type '{issue_type}' not found in unified mapping")
                return []
        
        # Calculate confidence based on frequency and mapping quality
        results = []
        for category in categories:
            confidence = self._calculate_confidence(actual_issue_type, category)
            if confidence >= confidence_threshold:
                results.append((category, confidence))
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def _calculate_confidence(self, issue_type: str, category: str) -> float:
        """Calculate confidence score for issue-category mapping."""
        # For explicit mappings, use high base confidence
        base_confidence = 0.9  # High confidence for unified mappings
        
        # Slight boost for frequently seen patterns
        issue_freq = self.issue_frequencies.get(issue_type, 1)
        freq_boost = min(0.1, issue_freq / 100)
        
        return min(1.0, base_confidence + freq_boost)
    
    def map_issues_to_categories(self, identified_issues: List[Dict]) -> List[Dict]:
        """
        Map identified issues to their categories with aggregated confidence.
        THIS IS THE KEY METHOD THAT NEEDS TO RETURN ALL MAPPED CATEGORIES.
        
        Args:
            identified_issues: List of dicts with 'issue_type' and 'confidence'
        
        Returns:
            List of dicts with 'category', 'confidence', 'source_issues'
        """
        category_scores = defaultdict(lambda: {'confidence': 0, 'sources': []})
        
        logger.info(f"🗂️  Mapping {len(identified_issues)} issues to categories using unified mapping...")
        
        for issue in identified_issues:
            issue_type = issue.get('issue_type')
            issue_confidence = issue.get('confidence', 1.0)
            
            logger.info(f"   🔍 Processing issue: '{issue_type}' (confidence: {issue_confidence:.3f})")
            
            # Get ALL categories for this issue
            categories = self.get_categories_for_issue(issue_type)
            
            if categories:
                logger.info(f"       → Found {len(categories)} categories: {[cat[0] for cat in categories]}")
                
                for category, mapping_confidence in categories:
                    # Combine confidences: issue identification * mapping confidence
                    combined_confidence = issue_confidence * mapping_confidence
                    
                    # Keep the highest confidence if multiple issues map to same category
                    if category_scores[category]['confidence'] < combined_confidence:
                        category_scores[category]['confidence'] = combined_confidence
                    
                    category_scores[category]['sources'].append({
                        'issue_type': issue_type,
                        'confidence': combined_confidence,
                        'evidence': issue.get('evidence', ''),
                        'source': issue.get('source', 'unified_mapping')
                    })
            else:
                logger.warning(f"       → No categories found for '{issue_type}'")
        
        # Convert to list format
        results = []
        for category, data in category_scores.items():
            # Collect evidence from all source issues for this category
            evidence_list = []
            issue_types_list = []
            for source in data['sources']:
                if source.get('evidence'):
                    evidence_list.append(source['evidence'])
                if source.get('issue_type'):
                    issue_types_list.append(source['issue_type'])
            
            results.append({
                'category': category,
                'confidence': data['confidence'],
                'source_issues': data['sources'],
                'evidence': '; '.join(evidence_list) if evidence_list else '',
                'issue_types': list(set(issue_types_list))
            })
        
        logger.info(f"🗂️  Final mapping result: {len(results)} categories")
        for result in results:
            logger.info(f"       • {result['category']} (confidence: {result['confidence']:.3f})")
        
        return sorted(results, key=lambda x: x['confidence'], reverse=True)
    
    def calculate_mapping_stats(self):
        """Generate statistics about issue-category mappings."""
        if not self.issue_to_categories:
            self.stats = {
                'total_issue_types': 0,
                'total_unique_categories': 0,
                'avg_categories_per_issue': 0,
                'max_categories_per_issue': 0,
                'min_categories_per_issue': 0
            }
            return self.stats
        
        category_counts = [len(cats) for cats in self.issue_to_categories.values()]
        
        self.stats = {
            'total_issue_types': len(self.issue_to_categories),
            'total_unique_categories': len(self.category_frequencies),
            'avg_categories_per_issue': sum(category_counts) / len(category_counts),
            'max_categories_per_issue': max(category_counts),
            'min_categories_per_issue': min(category_counts),
            'issues_with_single_category': sum(1 for c in category_counts if c == 1),
            'issues_with_multiple_categories': sum(1 for c in category_counts if c > 1)
        }
        
        logger.info(f"Unified mapping statistics: {self.stats}")
        return self.stats
    
    def get_all_issue_types(self) -> List[str]:
        """Get list of all known issue types."""
        return list(self.issue_to_categories.keys())
    
    def get_all_categories(self) -> List[str]:
        """Get list of all known categories."""
        return list(self.category_frequencies.keys())
    
    def __repr__(self):
        return (f"UnifiedIssueCategoryMapper(issues={len(self.issue_to_categories)}, "
                f"categories={len(self.category_frequencies)})")