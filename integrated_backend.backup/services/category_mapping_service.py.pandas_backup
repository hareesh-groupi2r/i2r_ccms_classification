"""
Category Mapping Service
Modular service for issue classification and category mapping
"""

import pandas as pd
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path
from fuzzywuzzy import fuzz, process
import json

from .interfaces import ICategoryMappingService, ProcessingResult, ProcessingStatus
from .configuration_service import get_config_service


class CategoryMappingService(ICategoryMappingService):
    """Service for mapping issues to categories and managing classification rules"""
    
    def __init__(self, config_service=None):
        self.config_service = config_service or get_config_service()
        self.config = self.config_service.get_service_config("category_mapping")
        self.ref_data_config = self.config_service.get_service_config("reference_data")
        
        # Configuration
        self.mapping_file = self.config.get("mapping_file", "Issue_to_category_mapping.csv")
        self.default_category = self.config.get("default_category", "Others")
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.fuzzy_matching = self.config.get("fuzzy_matching", True)
        
        # Load mapping data
        self.issue_mappings: Dict[str, str] = {}
        self.category_to_issues: Dict[str, List[str]] = {}
        self.available_categories: Set[str] = set()
        self.available_issue_types: Set[str] = set()
        
        self._load_mapping_data()
    
    def _load_mapping_data(self):
        """Load issue to category mapping data"""
        try:
            # Try to load from reference data config first
            if self.ref_data_config.get("issue_mappings"):
                self.issue_mappings = self.ref_data_config["issue_mappings"].copy()
                self.available_categories = set(self.ref_data_config.get("available_categories", []))
                self.available_issue_types = set(self.ref_data_config.get("available_issue_types", []))
            
            # If no data from config, try to load from CSV file
            if not self.issue_mappings:
                config_dir = Path(self.config_service.config_dir)
                mapping_file_path = config_dir / self.mapping_file
                
                if mapping_file_path.exists():
                    df = pd.read_csv(mapping_file_path)
                    
                    # Validate required columns
                    if 'Issue_Type' not in df.columns or 'Mapped_Category' not in df.columns:
                        print(f"Warning: CSV file {mapping_file_path} missing required columns")
                        self._create_default_mappings()
                        return
                    
                    # Load mappings
                    self.issue_mappings = dict(zip(df['Issue_Type'], df['Mapped_Category']))
                    self.available_categories = set(df['Mapped_Category'].unique())
                    self.available_issue_types = set(df['Issue_Type'].unique())
                    
                    print(f"Loaded {len(self.issue_mappings)} issue mappings from {mapping_file_path}")
                else:
                    print(f"Warning: Mapping file not found: {mapping_file_path}")
                    self._create_default_mappings()
            
            # Build reverse mapping (category -> issue types)
            self._build_reverse_mapping()
            
            # Ensure default category exists
            if self.default_category not in self.available_categories:
                self.available_categories.add(self.default_category)
                
        except Exception as e:
            print(f"Error loading mapping data: {e}")
            self._create_default_mappings()
    
    def _create_default_mappings(self):
        """Create default issue to category mappings"""
        default_mappings = {
            "Appointment of Design Director": "Authority's Obligations",
            "Clearances from Environmental & Forest Departments": "Clearances & Approvals",
            "Land Acquisition": "Land & Property Issues",
            "Payment Issues": "Financial Issues",
            "Quality Control": "Quality & Technical Issues",
            "Schedule Delays": "Time & Schedule Issues",
            "Material Supply": "Supply Chain Issues",
            "Safety Concerns": "Safety & Compliance",
            "Contract Disputes": "Legal & Contractual Issues",
            "Change Orders": "Scope & Change Management"
        }
        
        self.issue_mappings = default_mappings
        self.available_categories = set(default_mappings.values())
        self.available_categories.add(self.default_category)
        self.available_issue_types = set(default_mappings.keys())
        
        self._build_reverse_mapping()
        print(f"Created {len(default_mappings)} default issue mappings")
    
    def _build_reverse_mapping(self):
        """Build category to issue types mapping"""
        self.category_to_issues = {}
        
        for issue_type, category in self.issue_mappings.items():
            if category not in self.category_to_issues:
                self.category_to_issues[category] = []
            self.category_to_issues[category].append(issue_type)
    
    def map_issue_to_category(self, issue_type: str, **kwargs) -> ProcessingResult:
        """
        Map issue type to category
        
        Args:
            issue_type: Issue type to map
            **kwargs: Mapping options
                - use_fuzzy_matching: Enable fuzzy string matching
                - min_confidence: Minimum confidence for fuzzy matching
                - case_sensitive: Case sensitive matching
        
        Returns:
            ProcessingResult with mapped category and confidence
        """
        try:
            if not issue_type or not issue_type.strip():
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    error_message="Empty issue type provided"
                )
            
            issue_type = issue_type.strip()
            use_fuzzy = kwargs.get("use_fuzzy_matching", self.fuzzy_matching)
            min_confidence = kwargs.get("min_confidence", self.confidence_threshold)
            case_sensitive = kwargs.get("case_sensitive", False)
            
            # Direct exact match
            if case_sensitive:
                mapped_category = self.issue_mappings.get(issue_type)
            else:
                # Case-insensitive exact match
                for issue, category in self.issue_mappings.items():
                    if issue.lower() == issue_type.lower():
                        mapped_category = category
                        break
                else:
                    mapped_category = None
            
            if mapped_category:
                return ProcessingResult(
                    status=ProcessingStatus.SUCCESS,
                    data=mapped_category,
                    confidence=1.0,
                    metadata={
                        "match_type": "exact",
                        "original_issue": issue_type
                    }
                )
            
            # Fuzzy matching if enabled
            if use_fuzzy and self.available_issue_types:
                fuzzy_result = self._fuzzy_match_issue(issue_type, min_confidence)
                if fuzzy_result.status == ProcessingStatus.SUCCESS:
                    matched_issue = fuzzy_result.data["matched_issue"]
                    confidence = fuzzy_result.confidence
                    category = self.issue_mappings[matched_issue]
                    
                    return ProcessingResult(
                        status=ProcessingStatus.SUCCESS,
                        data=category,
                        confidence=confidence,
                        metadata={
                            "match_type": "fuzzy",
                            "matched_issue": matched_issue,
                            "original_issue": issue_type,
                            "fuzzy_score": fuzzy_result.data["score"]
                        }
                    )
            
            # No match found, return default category
            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=self.default_category,
                confidence=0.1,  # Low confidence for default
                metadata={
                    "match_type": "default",
                    "original_issue": issue_type,
                    "reason": "No matching issue type found"
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message=f"Error mapping issue to category: {str(e)}"
            )
    
    def _fuzzy_match_issue(self, issue_type: str, min_confidence: float) -> ProcessingResult:
        """Perform fuzzy matching against known issue types"""
        try:
            # Use fuzzywuzzy to find best match
            best_match, score = process.extractOne(
                issue_type, 
                list(self.available_issue_types),
                scorer=fuzz.ratio
            )
            
            # Convert score to confidence (0-1)
            confidence = score / 100.0
            
            if confidence >= min_confidence:
                return ProcessingResult(
                    status=ProcessingStatus.SUCCESS,
                    data={
                        "matched_issue": best_match,
                        "score": score
                    },
                    confidence=confidence
                )
            else:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    error_message=f"Best fuzzy match '{best_match}' has low confidence: {confidence:.2f}",
                    metadata={"best_match": best_match, "score": score}
                )
                
        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message=f"Error in fuzzy matching: {str(e)}"
            )
    
    def get_available_categories(self, **kwargs) -> ProcessingResult:
        """
        Get list of available categories
        
        Args:
            **kwargs: Options
                - include_counts: Include count of issue types per category
                - sort_by: 'name' or 'count'
        
        Returns:
            ProcessingResult with list of categories
        """
        try:
            include_counts = kwargs.get("include_counts", False)
            sort_by = kwargs.get("sort_by", "name")
            
            if include_counts:
                categories_with_counts = []
                for category in self.available_categories:
                    count = len(self.category_to_issues.get(category, []))
                    categories_with_counts.append({
                        "category": category,
                        "issue_count": count
                    })
                
                # Sort by name or count
                if sort_by == "count":
                    categories_with_counts.sort(key=lambda x: x["issue_count"], reverse=True)
                else:
                    categories_with_counts.sort(key=lambda x: x["category"])
                
                return ProcessingResult(
                    status=ProcessingStatus.SUCCESS,
                    data=categories_with_counts,
                    metadata={"total_categories": len(categories_with_counts)}
                )
            else:
                categories_list = sorted(list(self.available_categories))
                return ProcessingResult(
                    status=ProcessingStatus.SUCCESS,
                    data=categories_list,
                    metadata={"total_categories": len(categories_list)}
                )
                
        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message=f"Error getting available categories: {str(e)}"
            )
    
    def get_issue_types_for_category(self, category: str, **kwargs) -> ProcessingResult:
        """
        Get issue types that map to a specific category
        
        Args:
            category: Category name
            **kwargs: Options
                - case_sensitive: Case sensitive category matching
        
        Returns:
            ProcessingResult with list of issue types
        """
        try:
            if not category or not category.strip():
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    error_message="Empty category provided"
                )
            
            category = category.strip()
            case_sensitive = kwargs.get("case_sensitive", False)
            
            # Find matching category
            matched_category = None
            if case_sensitive:
                if category in self.category_to_issues:
                    matched_category = category
            else:
                for cat in self.category_to_issues.keys():
                    if cat.lower() == category.lower():
                        matched_category = cat
                        break
            
            if not matched_category:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    error_message=f"Category '{category}' not found"
                )
            
            issue_types = self.category_to_issues[matched_category]
            
            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=issue_types,
                metadata={
                    "category": matched_category,
                    "issue_count": len(issue_types)
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message=f"Error getting issue types for category: {str(e)}"
            )
    
    def add_mapping(self, issue_type: str, category: str, **kwargs) -> ProcessingResult:
        """Add new issue to category mapping"""
        try:
            if not issue_type or not issue_type.strip():
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    error_message="Empty issue type provided"
                )
            
            if not category or not category.strip():
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    error_message="Empty category provided"
                )
            
            issue_type = issue_type.strip()
            category = category.strip()
            
            # Add mapping
            old_category = self.issue_mappings.get(issue_type)
            self.issue_mappings[issue_type] = category
            
            # Update sets
            self.available_issue_types.add(issue_type)
            self.available_categories.add(category)
            
            # Rebuild reverse mapping
            self._build_reverse_mapping()
            
            # Save to file if requested
            if kwargs.get("save_to_file", False):
                self._save_mappings_to_file()
            
            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data={
                    "issue_type": issue_type,
                    "category": category,
                    "previous_category": old_category,
                    "is_new_mapping": old_category is None
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message=f"Error adding mapping: {str(e)}"
            )
    
    def remove_mapping(self, issue_type: str, **kwargs) -> ProcessingResult:
        """Remove issue to category mapping"""
        try:
            if not issue_type or not issue_type.strip():
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    error_message="Empty issue type provided"
                )
            
            issue_type = issue_type.strip()
            
            if issue_type not in self.issue_mappings:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    error_message=f"Issue type '{issue_type}' not found in mappings"
                )
            
            # Remove mapping
            removed_category = self.issue_mappings.pop(issue_type)
            self.available_issue_types.discard(issue_type)
            
            # Rebuild reverse mapping
            self._build_reverse_mapping()
            
            # Check if category still has mappings
            if removed_category not in self.category_to_issues:
                self.available_categories.discard(removed_category)
            
            # Save to file if requested
            if kwargs.get("save_to_file", False):
                self._save_mappings_to_file()
            
            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data={
                    "removed_issue": issue_type,
                    "removed_category": removed_category
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message=f"Error removing mapping: {str(e)}"
            )
    
    def _save_mappings_to_file(self) -> bool:
        """Save current mappings to CSV file"""
        try:
            config_dir = Path(self.config_service.config_dir)
            mapping_file_path = config_dir / self.mapping_file
            
            # Create DataFrame
            df = pd.DataFrame([
                {"Issue_Type": issue, "Mapped_Category": category}
                for issue, category in self.issue_mappings.items()
            ])
            
            # Save to CSV
            df.to_csv(mapping_file_path, index=False)
            print(f"Saved {len(df)} mappings to {mapping_file_path}")
            
            return True
            
        except Exception as e:
            print(f"Error saving mappings to file: {e}")
            return False
    
    def get_mapping_statistics(self) -> ProcessingResult:
        """Get statistics about current mappings"""
        try:
            stats = {
                "total_issue_types": len(self.available_issue_types),
                "total_categories": len(self.available_categories),
                "total_mappings": len(self.issue_mappings),
                "categories_with_counts": {},
                "default_category": self.default_category,
                "configuration": {
                    "fuzzy_matching": self.fuzzy_matching,
                    "confidence_threshold": self.confidence_threshold,
                    "mapping_file": self.mapping_file
                }
            }
            
            # Category statistics
            for category, issues in self.category_to_issues.items():
                stats["categories_with_counts"][category] = len(issues)
            
            # Find most common category
            if stats["categories_with_counts"]:
                most_common = max(stats["categories_with_counts"].items(), key=lambda x: x[1])
                stats["most_common_category"] = {
                    "category": most_common[0],
                    "count": most_common[1]
                }
            
            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=stats
            )
            
        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message=f"Error getting mapping statistics: {str(e)}"
            )
    
    def bulk_classify_issues(self, issue_types: List[str], **kwargs) -> ProcessingResult:
        """Classify multiple issue types at once"""
        try:
            if not issue_types:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    error_message="No issue types provided"
                )
            
            results = []
            successful_mappings = 0
            
            for issue_type in issue_types:
                mapping_result = self.map_issue_to_category(issue_type, **kwargs)
                
                result_item = {
                    "issue_type": issue_type,
                    "status": mapping_result.status.value,
                    "category": mapping_result.data if mapping_result.status == ProcessingStatus.SUCCESS else None,
                    "confidence": mapping_result.confidence,
                    "match_type": mapping_result.metadata.get("match_type") if mapping_result.metadata else None
                }
                
                if mapping_result.status == ProcessingStatus.SUCCESS:
                    successful_mappings += 1
                else:
                    result_item["error"] = mapping_result.error_message
                
                results.append(result_item)
            
            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=results,
                confidence=successful_mappings / len(issue_types),
                metadata={
                    "total_issues": len(issue_types),
                    "successful_mappings": successful_mappings,
                    "failed_mappings": len(issue_types) - successful_mappings
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message=f"Error in bulk classification: {str(e)}"
            )