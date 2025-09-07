"""
Category Normalizer Module
Ensures all categories are normalized to the standard 8 categories
Handles variations, typos, and incorrect entries
"""

import logging
from typing import Optional, List, Tuple
import difflib
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class CategoryNormalizer:
    """
    Normalizes categories to the standard 8 categories defined for the system.
    Handles variations in capitalization, spelling, and common errors.
    """
    
    # The 8 standard categories that are allowed in the system
    STANDARD_CATEGORIES = [
        "EoT",                      # Extension of Time
        "Dispute Resolution",       # Dispute related issues
        "Contractor's Obligations", # Contractor responsibilities
        "Payments",                 # Payment related issues
        "Authority's Obligations",  # Authority responsibilities
        "Change of Scope",          # Scope changes
        "Others",                   # Miscellaneous
        "Appointed Date"            # Appointed date related
    ]
    
    # Mapping of common variations to standard categories
    CATEGORY_MAPPINGS = {
        # EoT variations
        'eot': 'EoT',
        'extension of time': 'EoT',
        'time extension': 'EoT',
        'eot proposal': 'EoT',
        'eot proposals': 'EoT',
        
        # Dispute Resolution variations
        'dispute resolution': 'Dispute Resolution',
        'disputes': 'Dispute Resolution',
        'dispute': 'Dispute Resolution',
        'conflict resolution': 'Dispute Resolution',
        
        # Contractor's Obligations variations
        'contractors obligation': "Contractor's Obligations",
        'contractor obligation': "Contractor's Obligations",
        "contractor's obligation": "Contractor's Obligations",
        "contractor's obligations": "Contractor's Obligations",
        'contractors obligations': "Contractor's Obligations",
        'contractor obligations': "Contractor's Obligations",
        'contractor responsibility': "Contractor's Obligations",
        'contractor responsibilities': "Contractor's Obligations",
        
        # Authority's Obligations variations
        'authoritys obligation': "Authority's Obligations",
        'authority obligation': "Authority's Obligations",
        "authority's obligation": "Authority's Obligations",
        "authority's obligations": "Authority's Obligations",
        'authoritys obligations': "Authority's Obligations",
        'authority obligations': "Authority's Obligations",
        'authority responsibility': "Authority's Obligations",
        'authority responsibilities': "Authority's Obligations",
        
        # Payments variations
        'payment': 'Payments',
        'payments': 'Payments',
        'payment issues': 'Payments',
        'payment delay': 'Payments',
        'payment delays': 'Payments',
        'financial': 'Payments',
        
        # Change of Scope variations
        'change of scope': 'Change of Scope',
        'cos': 'Change of Scope',
        'scope change': 'Change of Scope',
        'scope changes': 'Change of Scope',
        'change in scope': 'Change of Scope',
        'scope modification': 'Change of Scope',
        'scope modifications': 'Change of Scope',
        
        # Others variations
        'others': 'Others',
        'other': 'Others',
        'miscellaneous': 'Others',
        'misc': 'Others',
        'general': 'Others',
        
        # Appointed Date variations
        'appointed date': 'Appointed Date',
        'appointment date': 'Appointed Date',
        'commencement date': 'Appointed Date',
        'start date': 'Appointed Date',
    }
    
    # Known issue types that might be incorrectly used as categories
    KNOWN_ISSUE_TYPES = [
        'completion certificate',
        'payment delay',
        'extension of time proposal',
        'change of scope proposal',
        'utility shifting',
        'borrow area',
        'force majeure',
        'pandemic',
        'covid',
        'design drawings',
        'memorandum',
        'appendix'
    ]
    
    def __init__(self, strict_mode: bool = True, similarity_threshold: float = 0.7):
        """
        Initialize the CategoryNormalizer.
        
        Args:
            strict_mode: If True, reject categories that can't be normalized
            similarity_threshold: Minimum similarity for fuzzy matching (0-1)
        """
        self.strict_mode = strict_mode
        self.similarity_threshold = similarity_threshold
        
        # Statistics tracking
        self.normalization_stats = {
            'total_processed': 0,
            'exact_matches': 0,
            'normalized': 0,
            'fuzzy_matched': 0,
            'rejected': 0,
            'issue_type_as_category': 0
        }
        
        # Load custom mappings if they exist
        self.custom_mappings_path = Path('./data/config/category_mappings.json')
        self.load_custom_mappings()
    
    def normalize_category(self, category: str) -> Tuple[Optional[str], str, float]:
        """
        Normalize a category to one of the 8 standard categories.
        
        Args:
            category: The category to normalize
            
        Returns:
            Tuple of (normalized_category, status, confidence)
            - normalized_category: The standard category or None if rejected
            - status: 'exact', 'normalized', 'fuzzy', 'rejected', or 'issue_type'
            - confidence: Confidence score (0-1)
        """
        self.normalization_stats['total_processed'] += 1
        
        if not category:
            return None, 'rejected', 0.0
        
        # Clean the input
        category = str(category).strip()
        
        # Check for exact match with standard categories
        if category in self.STANDARD_CATEGORIES:
            self.normalization_stats['exact_matches'] += 1
            return category, 'exact', 1.0
        
        # Convert to lowercase for mapping lookup
        category_lower = category.lower()
        
        # Check if it's a known issue type being used as category
        if category_lower in self.KNOWN_ISSUE_TYPES:
            logger.warning(f"Issue type '{category}' was used as a category. Mapping to 'Others'.")
            self.normalization_stats['issue_type_as_category'] += 1
            return 'Others', 'issue_type', 0.7
        
        # Check against known mappings
        if category_lower in self.CATEGORY_MAPPINGS:
            normalized = self.CATEGORY_MAPPINGS[category_lower]
            self.normalization_stats['normalized'] += 1
            return normalized, 'normalized', 0.95
        
        # Try fuzzy matching
        closest_match = self._fuzzy_match_category(category)
        if closest_match:
            normalized, confidence = closest_match
            self.normalization_stats['fuzzy_matched'] += 1
            return normalized, 'fuzzy', confidence
        
        # If strict mode, reject unknown categories
        if self.strict_mode:
            logger.error(f"Rejected unknown category: '{category}'")
            self.normalization_stats['rejected'] += 1
            return None, 'rejected', 0.0
        
        # In non-strict mode, default to 'Others'
        logger.warning(f"Unknown category '{category}' defaulted to 'Others'")
        self.normalization_stats['normalized'] += 1
        return 'Others', 'normalized', 0.5
    
    def _fuzzy_match_category(self, category: str) -> Optional[Tuple[str, float]]:
        """
        Find the closest matching standard category using fuzzy matching.
        
        Args:
            category: The category to match
            
        Returns:
            Tuple of (matched_category, confidence) or None
        """
        category_lower = category.lower()
        
        # First try to match against all known variations
        all_variations = list(self.CATEGORY_MAPPINGS.keys())
        matches = difflib.get_close_matches(
            category_lower, 
            all_variations, 
            n=1, 
            cutoff=self.similarity_threshold
        )
        
        if matches:
            matched_variation = matches[0]
            similarity = difflib.SequenceMatcher(None, category_lower, matched_variation).ratio()
            standard_category = self.CATEGORY_MAPPINGS[matched_variation]
            logger.info(f"Fuzzy matched '{category}' to '{standard_category}' via '{matched_variation}' (similarity: {similarity:.2f})")
            return standard_category, similarity
        
        # Then try to match directly against standard categories
        matches = difflib.get_close_matches(
            category, 
            self.STANDARD_CATEGORIES, 
            n=1, 
            cutoff=self.similarity_threshold
        )
        
        if matches:
            matched_category = matches[0]
            similarity = difflib.SequenceMatcher(None, category, matched_category).ratio()
            logger.info(f"Fuzzy matched '{category}' to '{matched_category}' (similarity: {similarity:.2f})")
            return matched_category, similarity
        
        return None
    
    def normalize_categories_list(self, categories: List[str]) -> List[Tuple[str, float]]:
        """
        Normalize a list of categories.
        
        Args:
            categories: List of categories to normalize
            
        Returns:
            List of tuples (normalized_category, confidence)
        """
        normalized = []
        for category in categories:
            norm_cat, status, confidence = self.normalize_category(category)
            if norm_cat:
                normalized.append((norm_cat, confidence))
        
        # Remove duplicates, keeping highest confidence for each category
        category_dict = {}
        for cat, conf in normalized:
            if cat not in category_dict or conf > category_dict[cat]:
                category_dict[cat] = conf
        
        return [(cat, conf) for cat, conf in category_dict.items()]
    
    def parse_and_normalize_categories(self, categories_str: str) -> List[str]:
        """
        Parse a comma-separated string of categories and normalize them.
        
        Args:
            categories_str: Comma-separated categories string
            
        Returns:
            List of normalized categories
        """
        if not categories_str:
            return []
        
        # Split by comma and normalize each
        categories = []
        for cat in str(categories_str).split(','):
            cat = cat.strip()
            if cat:
                norm_cat, status, confidence = self.normalize_category(cat)
                if norm_cat and norm_cat not in categories:
                    categories.append(norm_cat)
        
        return categories
    
    def validate_data_quality(self, df) -> dict:
        """
        Validate the quality of categories in a dataframe.
        
        Args:
            df: Pandas dataframe with 'category' column
            
        Returns:
            Dictionary with validation results
        """
        import pandas as pd
        
        results = {
            'total_rows': len(df),
            'rows_with_issues': [],
            'category_distribution': {},
            'normalization_needed': []
        }
        
        for idx, row in df.iterrows():
            if pd.notna(row.get('category')):
                categories_str = str(row['category'])
                original_categories = [c.strip() for c in categories_str.split(',') if c.strip()]
                normalized_categories = self.parse_and_normalize_categories(categories_str)
                
                # Check if normalization changed anything
                if set(original_categories) != set(normalized_categories):
                    results['normalization_needed'].append({
                        'row': idx + 2,  # Excel row number
                        'original': original_categories,
                        'normalized': normalized_categories,
                        'issue_type': row.get('issue_type', '')
                    })
                
                # Check for potential issues
                for cat in original_categories:
                    cat_lower = cat.lower()
                    # Check if an issue type was used as category
                    if cat_lower in self.KNOWN_ISSUE_TYPES:
                        results['rows_with_issues'].append({
                            'row': idx + 2,
                            'issue': f"Issue type '{cat}' used as category",
                            'issue_type': row.get('issue_type', ''),
                            'category': categories_str
                        })
                    # Check if category is completely unknown
                    elif cat not in self.STANDARD_CATEGORIES and cat_lower not in self.CATEGORY_MAPPINGS:
                        norm_cat, status, _ = self.normalize_category(cat)
                        if status == 'rejected':
                            results['rows_with_issues'].append({
                                'row': idx + 2,
                                'issue': f"Unknown category '{cat}'",
                                'issue_type': row.get('issue_type', ''),
                                'category': categories_str
                            })
                
                # Track distribution
                for cat in normalized_categories:
                    results['category_distribution'][cat] = results['category_distribution'].get(cat, 0) + 1
        
        return results
    
    def get_stats(self) -> dict:
        """Get normalization statistics."""
        return self.normalization_stats.copy()
    
    def reset_stats(self):
        """Reset normalization statistics."""
        for key in self.normalization_stats:
            self.normalization_stats[key] = 0
    
    def save_custom_mappings(self, mappings: dict):
        """
        Save custom category mappings to a JSON file.
        
        Args:
            mappings: Dictionary of custom mappings to add
        """
        self.custom_mappings_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Merge with existing mappings
        all_mappings = self.CATEGORY_MAPPINGS.copy()
        all_mappings.update({k.lower(): v for k, v in mappings.items()})
        
        with open(self.custom_mappings_path, 'w') as f:
            json.dump(all_mappings, f, indent=2)
        
        # Update current mappings
        self.CATEGORY_MAPPINGS = all_mappings
        logger.info(f"Saved custom mappings to {self.custom_mappings_path}")
    
    def load_custom_mappings(self):
        """Load custom category mappings from JSON file if it exists."""
        if self.custom_mappings_path.exists():
            try:
                with open(self.custom_mappings_path, 'r') as f:
                    custom_mappings = json.load(f)
                self.CATEGORY_MAPPINGS.update(custom_mappings)
                logger.info(f"Loaded custom mappings from {self.custom_mappings_path}")
            except Exception as e:
                logger.error(f"Error loading custom mappings: {e}")
    
    def export_normalization_report(self, output_path: str = './data/reports/category_normalization.json'):
        """
        Export a detailed normalization report.
        
        Args:
            output_path: Path to save the report
        """
        report = {
            'standard_categories': self.STANDARD_CATEGORIES,
            'total_known_mappings': len(self.CATEGORY_MAPPINGS),
            'statistics': self.get_stats(),
            'known_issue_types': self.KNOWN_ISSUE_TYPES,
            'category_mappings_sample': dict(list(self.CATEGORY_MAPPINGS.items())[:20])
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Exported normalization report to {output_path}")
        return report