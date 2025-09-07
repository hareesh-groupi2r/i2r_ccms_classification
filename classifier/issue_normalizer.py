"""
Issue Type Normalizer Module
Normalizes issue types to handle variations, typos, and inconsistencies
"""

import logging
import difflib
import json
from typing import Optional, Tuple, List
from pathlib import Path

logger = logging.getLogger(__name__)


class IssueTypeNormalizer:
    """
    Normalizes issue types to handle common variations like:
    - Case differences: "Utility Shifting" vs "Utility shifting"
    - Punctuation differences: "Safety measures during construction." vs "Safety measures during construction"
    - Spacing differences: "Borrow area" vs "Borrow area "
    - Singular/plural: "Extension of Time Proposals" vs "Extension of time proposal"
    - Minor typos: "Delay in construction activiites" vs "Delay in construction activities"
    """
    
    # Known normalizations based on the analysis
    NORMALIZATION_MAPPINGS = {
        # All the variations found in check_issue_types.py analysis
        
        # 1. Appointment of Safety & Proof Consultants
        "appointment of safety and proof consultants": "Appointment of Safety & Proof Consultants",
        
        # 2. Borrow area (remove trailing space, keep "Borrow area " as it has more samples)
        "borrow area": "Borrow area ",
        
        # 3-9. Change of scope variations (normalize to "Change of scope proposal" - most common)
        "change of scope proposals": "Change of scope proposal",
        "change of scope proposal clarifications": "Change of scope proposals clarifications",
        "change of scope proposals clarifications": "Change of scope proposals clarifications",
        
        # 10. Change of Scope approval (remove newline)
        "delay due to change of scope approval\n": "Delay due to Change of Scope approval",
        
        # 11-13. Construction delay typos
        "delay in construction activates": "Delay in construction activities",
        "delay in construction activiites": "Delay in construction activities",
        
        # 14. Extension of Time (normalize to plural - more common: 25 vs 5)
        "extension of time proposal": "Extension of Time Proposals",
        
        # 15-22. Handing over of land (normalize to most complete: "Handing over of land /Possession of site.  ")
        "handing over of land / possession of site": "Handing over of land /Possession of site.  ",
        "handing over of land /possession of site": "Handing over of land /Possession of site.  ",
        "handing over of land /possession of site.": "Handing over of land /Possession of site.  ",
        "handing over of land/ procession of site": "Handing over of land /Possession of site.  ",
        
        # 23. Force Majeure (keep the more common: "Intimation of Occurrence of Force majeure Events")
        "intimation of occurrence of force majeure events": "Intimation of Occurrence of Force majeure Events",
        
        # 24. Maintenance of diversion road
        "maintenance of diversion road / existing road": "Maintenance of diversion road/ existing road",
        
        # 25. Milestones (keep the more common)
        "notices for achievement / non achievement of milestones": "Notices for Achievement/ Non Achievement of Milestones",
        
        # 26. Occurrence of cyclones (remove trailing space)
        "occurrence of cyclones ": "Occurrence of cyclones",
        
        # 27. Pandemic (keep more common: "Outbreak of epidemic or pandemic" - 12 vs 9)
        "outbreak of epidemic or pandemic": "Outbreak of epidemic or pandemic",
        
        # 28. Permission for extracting soil (remove trailing space)
        "permission for extracting soil from minor irrigation tanks and ponds ": "Permission for extracting soil from minor irrigation tanks and ponds",
        
        # 29. Progress Review (keep the more common: "Progress Review" - 9 vs 4)
        "progress review": "Progress Review",
        
        # 30. Project Highway (remove trailing space)
        "providing right of way in terms of length of project highway ": "Providing Right of Way in terms of length of Project Highway",
        
        # 31. Safety measures (normalize to version with period)
        "safety measures during construction ": "Safety measures during construction.",
        
        # 32. Slow progress (keep the more common: "Slow Progress of Works " - 5 vs 4)
        "slow progress of works": "Slow Progress of Works ",
        
        # 33. Under utilization (fix typo: uitilisation -> utilisation)
        "under uitilisation / idling of resources": "Under utilisation / idling of resources",
        
        # 34. Utility shifting (normalize to lowercase - more common: 25 vs 5)
        "utility shifting": "Utility shifting",
    }
    
    def __init__(self, similarity_threshold: float = 0.92):
        """
        Initialize the Issue Type Normalizer.
        
        Args:
            similarity_threshold: Minimum similarity for fuzzy matching (0-1)
        """
        self.similarity_threshold = similarity_threshold
        
        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'exact_matches': 0,
            'normalized': 0,
            'fuzzy_matched': 0,
            'rejected': 0
        }
        
        # Load custom mappings if they exist
        self.custom_mappings_path = Path('./data/config/issue_mappings.json')
        self.load_custom_mappings()
    
    def normalize_issue_type(self, issue_type: str) -> Tuple[Optional[str], str, float]:
        """
        Normalize an issue type to handle variations.
        
        Args:
            issue_type: The issue type to normalize
            
        Returns:
            Tuple of (normalized_issue, status, confidence)
            - normalized_issue: The normalized issue type or None if rejected
            - status: 'exact', 'normalized', 'fuzzy', or 'rejected'
            - confidence: Confidence score (0-1)
        """
        self.stats['total_processed'] += 1
        
        if not issue_type:
            return None, 'rejected', 0.0
        
        # Clean the input (basic cleanup)
        original = issue_type
        cleaned = str(issue_type).strip()
        
        # Convert to lowercase for mapping lookup
        issue_lower = cleaned.lower()
        
        # Check against known mappings
        if issue_lower in self.NORMALIZATION_MAPPINGS:
            normalized = self.NORMALIZATION_MAPPINGS[issue_lower]
            self.stats['normalized'] += 1
            logger.debug(f"Normalized '{issue_type}' to '{normalized}'")
            return normalized, 'normalized', 0.95
        
        # Basic cleanup normalizations
        normalized = self._basic_cleanup(cleaned)
        if normalized != cleaned:
            self.stats['normalized'] += 1
            return normalized, 'normalized', 0.90
        
        # If no changes needed after cleanup, check if original was already clean
        if cleaned == original:
            self.stats['exact_matches'] += 1
            return cleaned, 'exact', 1.0
        else:
            # Basic cleanup was applied
            self.stats['normalized'] += 1
            return cleaned, 'normalized', 0.95
    
    def _basic_cleanup(self, issue_type: str) -> str:
        """
        Apply basic cleanup normalizations.
        
        Args:
            issue_type: The issue type to clean up
            
        Returns:
            Cleaned up issue type
        """
        # Remove trailing whitespace and normalize internal spacing
        cleaned = ' '.join(issue_type.split())
        
        # Remove trailing periods if they seem inconsistent
        # (Only remove if it's the only difference from a common pattern)
        
        # Handle common case inconsistencies
        # This is conservative - only fix obvious patterns
        
        return cleaned
    
    def fuzzy_match_similar_issues(self, issue_type: str, known_issues: List[str]) -> Optional[Tuple[str, float]]:
        """
        Find the closest matching issue type using fuzzy matching.
        
        Args:
            issue_type: The issue type to match
            known_issues: List of known issue types
            
        Returns:
            Tuple of (matched_issue, similarity) or None
        """
        if not known_issues:
            return None
        
        # Find closest matches
        matches = difflib.get_close_matches(
            issue_type.lower(),
            [issue.lower() for issue in known_issues],
            n=1,
            cutoff=self.similarity_threshold
        )
        
        if matches:
            # Find the original casing
            match_lower = matches[0]
            for original in known_issues:
                if original.lower() == match_lower:
                    similarity = difflib.SequenceMatcher(None, issue_type.lower(), match_lower).ratio()
                    return original, similarity
        
        return None
    
    def batch_normalize(self, issue_types: List[str]) -> List[Tuple[str, str, float]]:
        """
        Normalize a batch of issue types.
        
        Args:
            issue_types: List of issue types to normalize
            
        Returns:
            List of tuples (original, normalized, confidence)
        """
        results = []
        for issue_type in issue_types:
            normalized, status, confidence = self.normalize_issue_type(issue_type)
            results.append((issue_type, normalized or issue_type, confidence))
        
        return results
    
    def validate_data_quality(self, df) -> dict:
        """
        Validate the quality of issue types in a dataframe.
        
        Args:
            df: Pandas dataframe with 'issue_type' column
            
        Returns:
            Dictionary with validation results
        """
        import pandas as pd
        
        results = {
            'total_rows': len(df),
            'normalization_needed': [],
            'potential_duplicates': [],
            'statistics': {}
        }
        
        # Get all unique issue types
        unique_issues = df['issue_type'].dropna().unique()
        original_count = len(unique_issues)
        
        # Normalize all issue types
        normalized_mapping = {}
        for issue in unique_issues:
            normalized, status, confidence = self.normalize_issue_type(issue)
            if normalized and normalized != issue:
                normalized_mapping[issue] = {
                    'normalized': normalized,
                    'status': status,
                    'confidence': confidence
                }
        
        # Find potential duplicates after normalization
        normalized_issues = {}
        for issue in unique_issues:
            normalized, _, _ = self.normalize_issue_type(issue)
            if normalized not in normalized_issues:
                normalized_issues[normalized] = []
            normalized_issues[normalized].append(issue)
        
        # Report duplicates
        for normalized, originals in normalized_issues.items():
            if len(originals) > 1:
                total_samples = sum(df[df['issue_type'] == orig].shape[0] for orig in originals)
                results['potential_duplicates'].append({
                    'normalized': normalized,
                    'originals': originals,
                    'total_samples': total_samples,
                    'sample_counts': {orig: df[df['issue_type'] == orig].shape[0] for orig in originals}
                })
        
        results['normalization_needed'] = list(normalized_mapping.items())
        results['statistics'] = {
            'original_unique_count': original_count,
            'normalized_unique_count': len(normalized_issues),
            'reduction': original_count - len(normalized_issues),
            'normalization_rate': len(normalized_mapping) / original_count if original_count > 0 else 0
        }
        
        return results
    
    def get_stats(self) -> dict:
        """Get normalization statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset normalization statistics."""
        for key in self.stats:
            self.stats[key] = 0
    
    def save_custom_mappings(self, mappings: dict):
        """
        Save custom issue type mappings to a JSON file.
        
        Args:
            mappings: Dictionary of custom mappings to add
        """
        self.custom_mappings_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Merge with existing mappings
        all_mappings = self.NORMALIZATION_MAPPINGS.copy()
        all_mappings.update({k.lower(): v for k, v in mappings.items()})
        
        with open(self.custom_mappings_path, 'w') as f:
            json.dump(all_mappings, f, indent=2)
        
        # Update current mappings
        self.NORMALIZATION_MAPPINGS = all_mappings
        logger.info(f"Saved custom mappings to {self.custom_mappings_path}")
    
    def load_custom_mappings(self):
        """Load custom issue type mappings from JSON file if it exists."""
        if self.custom_mappings_path.exists():
            try:
                with open(self.custom_mappings_path, 'r') as f:
                    custom_mappings = json.load(f)
                self.NORMALIZATION_MAPPINGS.update(custom_mappings)
                logger.info(f"Loaded custom mappings from {self.custom_mappings_path}")
            except Exception as e:
                logger.error(f"Error loading custom mappings: {e}")
    
    def export_normalization_report(self, output_path: str = './data/reports/issue_normalization.json'):
        """
        Export a detailed normalization report.
        
        Args:
            output_path: Path to save the report
        """
        report = {
            'total_known_mappings': len(self.NORMALIZATION_MAPPINGS),
            'similarity_threshold': self.similarity_threshold,
            'statistics': self.get_stats(),
            'sample_mappings': dict(list(self.NORMALIZATION_MAPPINGS.items())[:10])
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Exported issue normalization report to {output_path}")
        return report