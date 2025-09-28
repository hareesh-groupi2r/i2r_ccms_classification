"""
Category Mapping Service - Pandas-free version
Modular service for issue classification and category mapping without pandas dependency
"""

import csv
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
        """Load issue to category mapping data using built-in CSV module"""
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
                    # Load CSV using built-in csv module instead of pandas
                    mappings_data = []
                    with open(mapping_file_path, 'r', encoding='utf-8') as csvfile:
                        reader = csv.DictReader(csvfile)
                        mappings_data = list(reader)
                    
                    # Validate required columns
                    if not mappings_data or 'Issue_Type' not in mappings_data[0] or 'Mapped_Category' not in mappings_data[0]:
                        print(f"Warning: CSV file {mapping_file_path} missing required columns")
                        self._create_default_mappings()
                        return
                    
                    # Load mappings
                    self.issue_mappings = {row['Issue_Type']: row['Mapped_Category'] for row in mappings_data}
                    self.available_categories = set(row['Mapped_Category'] for row in mappings_data)
                    self.available_issue_types = set(row['Issue_Type'] for row in mappings_data)
                    
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
    
    def _build_reverse_mapping(self):
        """Build reverse mapping from categories to issue types"""
        self.category_to_issues = {}
        for issue_type, category in self.issue_mappings.items():
            if category not in self.category_to_issues:
                self.category_to_issues[category] = []
            self.category_to_issues[category].append(issue_type)
    
    def classify_issue(self, issue_text: str) -> ProcessingResult:
        """Classify an issue text to a category"""
        try:
            if not issue_text or not issue_text.strip():
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    error="Empty issue text provided"
                )
            
            issue_text = issue_text.strip()
            
            # Direct match first
            if issue_text in self.issue_mappings:
                category = self.issue_mappings[issue_text]
                return ProcessingResult(
                    status=ProcessingStatus.SUCCESS,
                    data={
                        "category": category,
                        "confidence": 1.0,
                        "match_type": "exact"
                    }
                )
            
            # Fuzzy matching if enabled
            if self.fuzzy_matching and self.available_issue_types:
                best_match, confidence = process.extractOne(
                    issue_text, 
                    list(self.available_issue_types)
                )
                
                confidence_score = confidence / 100.0  # Convert to 0-1 scale
                
                if confidence_score >= self.confidence_threshold:
                    category = self.issue_mappings[best_match]
                    return ProcessingResult(
                        status=ProcessingStatus.SUCCESS,
                        data={
                            "category": category,
                            "confidence": confidence_score,
                            "match_type": "fuzzy",
                            "matched_issue": best_match
                        }
                    )
            
            # Default category if no good match
            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data={
                    "category": self.default_category,
                    "confidence": 0.0,
                    "match_type": "default"
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error=f"Classification error: {str(e)}"
            )
    
    def get_available_categories(self) -> List[str]:
        """Get list of available categories"""
        return sorted(list(self.available_categories))
    
    def get_available_issue_types(self) -> List[str]:
        """Get list of available issue types"""
        return sorted(list(self.available_issue_types))
    
    def add_mapping(self, issue_type: str, category: str) -> ProcessingResult:
        """Add a new issue type to category mapping"""
        try:
            self.issue_mappings[issue_type] = category
            self.available_issue_types.add(issue_type)
            self.available_categories.add(category)
            self._build_reverse_mapping()
            
            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data={"message": f"Added mapping: {issue_type} -> {category}"}
            )
        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error=f"Failed to add mapping: {str(e)}"
            )
    
    def save_mappings(self) -> ProcessingResult:
        """Save current mappings to CSV file using built-in csv module"""
        try:
            config_dir = Path(self.config_service.config_dir)
            config_dir.mkdir(parents=True, exist_ok=True)
            mapping_file_path = config_dir / self.mapping_file
            
            # Write to CSV using built-in csv module
            with open(mapping_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['Issue_Type', 'Mapped_Category']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for issue_type, category in self.issue_mappings.items():
                    writer.writerow({
                        'Issue_Type': issue_type,
                        'Mapped_Category': category
                    })
            
            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data={"message": f"Saved {len(self.issue_mappings)} mappings to {mapping_file_path}"}
            )
            
        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error=f"Failed to save mappings: {str(e)}"
            )
    
    def get_category_summary(self) -> Dict[str, int]:
        """Get summary of issues per category"""
        summary = {}
        for category in self.available_categories:
            summary[category] = len(self.category_to_issues.get(category, []))
        return summary


def get_category_mapping_service(config_service=None) -> CategoryMappingService:
    """Factory function to get CategoryMappingService instance"""
    return CategoryMappingService(config_service)