"""
Validation Engine Module
Prevents LLM hallucinations by enforcing strict allowlists for issue types and categories
"""

from typing import List, Dict, Set, Tuple, Optional
import difflib
import logging
import pandas as pd
import json
from .category_normalizer import CategoryNormalizer

logger = logging.getLogger(__name__)


class ValidationEngine:
    """
    Validates classification outputs against known issue types and categories.
    Prevents hallucinations by maintaining strict allowlists and auto-correcting invalid values.
    """
    
    def __init__(self, training_data_path: str = None):
        """
        Initialize validation engine with strict allowlists.
        
        Args:
            training_data_path: Path to training data to extract valid values
        """
        self.valid_issue_types = set()
        self.valid_categories = set()
        
        # Configuration for auto-correction
        self.similarity_threshold = 0.7  # Minimum similarity for auto-correction
        self.case_sensitive = False
        
        # Initialize category normalizer (must be done before loading values)
        self.normalizer = CategoryNormalizer(strict_mode=True, similarity_threshold=self.similarity_threshold)
        
        # Track validation statistics
        self.validation_stats = {
            'total_validations': 0,
            'corrections_made': 0,
            'rejections': 0,
            'hallucinations_detected': 0
        }
        
        if training_data_path:
            self._load_valid_values(training_data_path)
    
    def _load_valid_values(self, data_path: str):
        """
        Load the exhaustive list of valid issue types and categories from training data.
        
        Args:
            data_path: Path to the training data Excel file
        """
        try:
            df = pd.read_excel(data_path)
            
            # Extract all unique issue types (should be 130 total)
            self.valid_issue_types = set(df['issue_type'].dropna().unique())
            
            # Use only the standard 8 categories
            self.valid_categories = set(CategoryNormalizer.STANDARD_CATEGORIES)
            
            # Also run validation check on the data
            validation_results = self.normalizer.validate_data_quality(df)
            if validation_results['rows_with_issues']:
                logger.warning(f"Found {len(validation_results['rows_with_issues'])} rows with category issues during data load")
                for issue in validation_results['rows_with_issues'][:5]:  # Show first 5
                    logger.warning(f"  Row {issue['row']}: {issue['issue']}")
            
            logger.info(f"Loaded {len(self.valid_issue_types)} valid issue types")
            logger.info(f"Loaded {len(self.valid_categories)} valid categories")
            
        except Exception as e:
            logger.error(f"Error loading valid values from {data_path}: {e}")
            raise
    
    def sync_with_issue_mapper(self, issue_mapper):
        """
        Synchronize valid issue types with the unified issue mapper.
        This ensures the LLM validation uses the complete set of 194+ issue types.
        
        Args:
            issue_mapper: UnifiedIssueCategoryMapper instance
        """
        # Get all issue types from the unified mapper (194+)
        all_issue_types = issue_mapper.get_all_issue_types()
        all_categories = issue_mapper.get_all_categories()
        
        # Update the validation constraints
        self.valid_issue_types = set(all_issue_types)
        self.valid_categories = set(all_categories)
        
        logger.info(f"🔄 Synced ValidationEngine with UnifiedIssueCategoryMapper:")
        logger.info(f"   📋 Issue types: {len(self.valid_issue_types)} (was limited to training data only)")
        logger.info(f"   📋 Categories: {len(self.valid_categories)}")
        logger.info(f"   ✅ LLM prompts will now include complete issue type list")
    
    def validate_issue_type(self, issue_type: str, 
                           auto_correct: bool = True) -> Tuple[Optional[str], bool, float]:
        """
        Validate and optionally correct an issue type.
        
        Args:
            issue_type: The issue type to validate
            auto_correct: Whether to attempt auto-correction for invalid values
            
        Returns:
            Tuple of (validated_issue, is_valid, confidence)
        """
        self.validation_stats['total_validations'] += 1
        
        # Clean input
        issue_type = str(issue_type).strip()
        
        # Exact match
        if issue_type in self.valid_issue_types:
            return issue_type, True, 1.0
        
        # Case-insensitive match
        if not self.case_sensitive:
            issue_lower = issue_type.lower()
            for valid_issue in self.valid_issue_types:
                if valid_issue.lower() == issue_lower:
                    logger.debug(f"Case-corrected '{issue_type}' to '{valid_issue}'")
                    return valid_issue, True, 0.95
        
        # Auto-correction using fuzzy matching
        if auto_correct:
            closest_matches = difflib.get_close_matches(
                issue_type, 
                self.valid_issue_types, 
                n=1, 
                cutoff=self.similarity_threshold
            )
            
            if closest_matches:
                corrected = closest_matches[0]
                similarity = difflib.SequenceMatcher(
                    None, issue_type, corrected
                ).ratio()
                
                logger.warning(
                    f"Auto-corrected hallucinated issue '{issue_type}' to '{corrected}' "
                    f"(similarity: {similarity:.2f})"
                )
                
                self.validation_stats['corrections_made'] += 1
                self.validation_stats['hallucinations_detected'] += 1
                
                return corrected, False, similarity
        
        # No valid match found
        logger.error(f"Rejected hallucinated issue type: '{issue_type}'")
        self.validation_stats['rejections'] += 1
        self.validation_stats['hallucinations_detected'] += 1
        
        return None, False, 0.0
    
    def validate_category(self, category: str, 
                         auto_correct: bool = True) -> Tuple[Optional[str], bool, float]:
        """
        Validate and optionally correct a category.
        
        Args:
            category: The category to validate
            auto_correct: Whether to attempt auto-correction for invalid values
            
        Returns:
            Tuple of (validated_category, is_valid, confidence)
        """
        self.validation_stats['total_validations'] += 1
        
        # Use the normalizer for validation
        normalized_cat, status, confidence = self.normalizer.normalize_category(category)
        
        if normalized_cat:
            if status == 'exact':
                return normalized_cat, True, confidence
            elif status in ['normalized', 'fuzzy']:
                self.validation_stats['corrections_made'] += 1
                logger.debug(f"Normalized category '{category}' to '{normalized_cat}' (status: {status})")
                return normalized_cat, False, confidence
            elif status == 'issue_type':
                self.validation_stats['corrections_made'] += 1
                self.validation_stats['hallucinations_detected'] += 1
                logger.warning(f"Issue type '{category}' used as category, normalized to '{normalized_cat}'")
                return normalized_cat, False, confidence
        
        # No valid match found
        logger.error(f"Rejected invalid category: '{category}'")
        self.validation_stats['rejections'] += 1
        self.validation_stats['hallucinations_detected'] += 1
        
        return None, False, 0.0
    
    def validate_classification_output(self, classification: Dict) -> Dict:
        """
        Validate entire classification output and filter/correct invalid values.
        
        Args:
            classification: Classification output to validate
            
        Returns:
            Validated classification with report
        """
        validated = {
            'identified_issues': [],
            'categories': [],
            'validation_report': {
                'hallucinations_detected': False,
                'corrections_made': [],
                'rejections': [],
                'validation_status': 'clean'
            }
        }
        
        # Validate issues
        if 'identified_issues' in classification:
            for issue in classification['identified_issues']:
                issue_type = issue.get('issue_type', '')
                validated_issue, is_valid, confidence = self.validate_issue_type(
                    issue_type, auto_correct=True
                )
                
                if validated_issue:
                    issue_copy = issue.copy()
                    issue_copy['issue_type'] = validated_issue
                    issue_copy['confidence'] = issue.get('confidence', 1.0) * confidence
                    issue_copy['validation_status'] = 'valid' if is_valid else 'corrected'
                    validated['identified_issues'].append(issue_copy)
                    
                    if not is_valid:
                        validated['validation_report']['hallucinations_detected'] = True
                        validated['validation_report']['corrections_made'].append({
                            'type': 'issue',
                            'original': issue_type,
                            'corrected': validated_issue,
                            'confidence': confidence
                        })
                else:
                    validated['validation_report']['hallucinations_detected'] = True
                    validated['validation_report']['rejections'].append({
                        'type': 'issue',
                        'value': issue_type,
                        'reason': 'No valid match found'
                    })
        
        # Validate categories
        if 'categories' in classification:
            for category_info in classification['categories']:
                category = category_info.get('category', '')
                validated_cat, is_valid, confidence = self.validate_category(
                    category, auto_correct=True
                )
                
                if validated_cat:
                    cat_copy = category_info.copy()
                    cat_copy['category'] = validated_cat
                    cat_copy['confidence'] = category_info.get('confidence', 1.0) * confidence
                    cat_copy['validation_status'] = 'valid' if is_valid else 'corrected'
                    validated['categories'].append(cat_copy)
                    
                    if not is_valid:
                        validated['validation_report']['hallucinations_detected'] = True
                        validated['validation_report']['corrections_made'].append({
                            'type': 'category',
                            'original': category,
                            'corrected': validated_cat,
                            'confidence': confidence
                        })
                else:
                    validated['validation_report']['hallucinations_detected'] = True
                    validated['validation_report']['rejections'].append({
                        'type': 'category',
                        'value': category,
                        'reason': 'No valid match found'
                    })
        
        # Set validation status
        if validated['validation_report']['hallucinations_detected']:
            if validated['validation_report']['rejections']:
                validated['validation_report']['validation_status'] = 'rejected_items'
            else:
                validated['validation_report']['validation_status'] = 'corrected'
        
        return validated
    
    def create_constrained_prompt(self, prompt_type: str = 'issues') -> str:
        """
        Create prompt with explicit constraints to prevent hallucinations.
        
        Args:
            prompt_type: Either 'issues' or 'categories'
            
        Returns:
            Constraint prompt text
        """
        if prompt_type == 'issues':
            return f"""
STRICT INSTRUCTION: You MUST ONLY use issue types from this exact list. 
DO NOT create new issue types. If uncertain, choose the closest match from this list:

VALID ISSUE TYPES (ONLY use these):
{json.dumps(sorted(list(self.valid_issue_types)), indent=2)}

Any issue type not in this list will be rejected.
"""
        else:  # categories
            return f"""
STRICT INSTRUCTION: You MUST ONLY use categories from this exact list.
DO NOT create new categories. If uncertain, choose the closest match from this list:

VALID CATEGORIES (ONLY use these):
{json.dumps(sorted(list(self.valid_categories)), indent=2)}

Any category not in this list will be rejected.
"""
    
    def get_validation_stats(self) -> Dict:
        """
        Get validation statistics.
        
        Returns:
            Dictionary with validation statistics
        """
        return self.validation_stats.copy()
    
    def reset_stats(self):
        """Reset validation statistics."""
        self.validation_stats = {
            'total_validations': 0,
            'corrections_made': 0,
            'rejections': 0,
            'hallucinations_detected': 0
        }
    
    def add_valid_issue_type(self, issue_type: str):
        """
        Add a new valid issue type to the allowlist.
        
        Args:
            issue_type: Issue type to add
        """
        self.valid_issue_types.add(issue_type)
        logger.info(f"Added new valid issue type: {issue_type}")
    
    def add_valid_category(self, category: str):
        """
        Add a new valid category to the allowlist.
        
        Args:
            category: Category to add
        """
        self.valid_categories.add(category)
        logger.info(f"Added new valid category: {category}")
    
    def save_allowlists(self, output_path: str):
        """
        Save the allowlists to a JSON file.
        
        Args:
            output_path: Path where to save the allowlists
        """
        allowlists = {
            'valid_issue_types': sorted(list(self.valid_issue_types)),
            'valid_categories': sorted(list(self.valid_categories)),
            'total_issue_types': len(self.valid_issue_types),
            'total_categories': len(self.valid_categories)
        }
        
        with open(output_path, 'w') as f:
            json.dump(allowlists, f, indent=2)
        
        logger.info(f"Saved allowlists to {output_path}")
    
    def load_allowlists(self, allowlists_path: str):
        """
        Load allowlists from a JSON file.
        
        Args:
            allowlists_path: Path to the saved allowlists file
        """
        try:
            with open(allowlists_path, 'r') as f:
                allowlists = json.load(f)
            
            self.valid_issue_types = set(allowlists['valid_issue_types'])
            self.valid_categories = set(allowlists['valid_categories'])
            
            logger.info(f"Loaded allowlists from {allowlists_path}")
            
        except Exception as e:
            logger.error(f"Error loading allowlists from {allowlists_path}: {e}")
            raise
    
    def __repr__(self):
        return (f"ValidationEngine(issue_types={len(self.valid_issue_types)}, "
                f"categories={len(self.valid_categories)})")