"""
Minimal Category Mapping Service - No pandas dependency
For quick testing without compilation issues
"""

from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path
import json

from .interfaces import ICategoryMappingService, ProcessingResult, ProcessingStatus
from .configuration_service import get_config_service


class CategoryMappingService(ICategoryMappingService):
    """Minimal service for basic category mapping without pandas"""
    
    def __init__(self, config_service=None):
        self.config_service = config_service or get_config_service()
        self.default_category = "Others"
        
        # Use hardcoded default mappings for now
        self.issue_mappings = {
            "Payment Issues": "Financial Issues",
            "Quality Control": "Quality & Technical Issues", 
            "Schedule Delays": "Time & Schedule Issues",
            "Material Supply": "Supply Chain Issues",
            "Contract Disputes": "Legal & Contractual Issues",
            "Safety Concerns": "Safety & Compliance",
            "Land Acquisition": "Land & Property Issues"
        }
        
        self.available_categories = set(self.issue_mappings.values())
        self.available_categories.add(self.default_category)
        self.available_issue_types = set(self.issue_mappings.keys())
    
    def map_issue_to_category(self, issue_type: str, **kwargs) -> ProcessingResult:
        """Map issue type to category with basic matching"""
        try:
            if not issue_type or not issue_type.strip():
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    error_message="Empty issue type provided"
                )
            
            issue_type = issue_type.strip()
            
            # Direct match
            if issue_type in self.issue_mappings:
                return ProcessingResult(
                    status=ProcessingStatus.SUCCESS,
                    data=self.issue_mappings[issue_type],
                    confidence=1.0
                )
            
            # Simple partial matching
            for issue, category in self.issue_mappings.items():
                if issue.lower() in issue_type.lower() or issue_type.lower() in issue.lower():
                    return ProcessingResult(
                        status=ProcessingStatus.SUCCESS,
                        data=category,
                        confidence=0.7
                    )
            
            # Default category
            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=self.default_category,
                confidence=0.1
            )
            
        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message=f"Error mapping issue: {str(e)}"
            )
    
    def get_available_categories(self, **kwargs) -> ProcessingResult:
        """Get available categories"""
        return ProcessingResult(
            status=ProcessingStatus.SUCCESS,
            data=sorted(list(self.available_categories))
        )


# Factory function
def get_category_mapping_service(config_service=None) -> CategoryMappingService:
    return CategoryMappingService(config_service)