"""
Data Sufficiency Analyzer Module
Identifies under-represented issue types and categories in training data
"""

from typing import Dict, List, Tuple
import pandas as pd
from collections import defaultdict
import logging
import json
from pathlib import Path
from .category_normalizer import CategoryNormalizer
from .issue_normalizer import IssueTypeNormalizer

logger = logging.getLogger(__name__)


class DataSufficiencyAnalyzer:
    """
    Analyzes training data sufficiency and provides warnings for under-represented classifications.
    Adjusts confidence scores based on available training data.
    """
    
    def __init__(self, training_data_path: str = None):
        """
        Initialize the analyzer with training data.
        
        Args:
            training_data_path: Path to the training data Excel file
        """
        self.df = None
        self.issue_counts = {}
        self.category_counts = {}
        
        # Initialize normalizers
        self.category_normalizer = CategoryNormalizer(strict_mode=False)
        self.issue_normalizer = IssueTypeNormalizer()
        
        # Sufficiency thresholds
        self.sufficiency_thresholds = {
            'critical': 5,   # < 5 samples: unreliable
            'warning': 10,   # 5-10 samples: low confidence
            'good': 20,      # 10-20 samples: acceptable
            'excellent': 50  # > 50 samples: high confidence
        }
        
        # Confidence adjustment multipliers
        self.confidence_adjustments = {
            'critical': 0.5,    # Halve confidence for very low data
            'warning': 0.7,     # Reduce confidence by 30%
            'good': 0.85,       # Reduce confidence by 15%
            'very_good': 0.95,  # Reduce confidence by 5%
            'excellent': 1.0    # No adjustment
        }
        
        if training_data_path:
            self._analyze_distribution(training_data_path)
    
    def _analyze_distribution(self, data_path: str):
        """
        Count samples for each issue type and category.
        
        Args:
            data_path: Path to the training data Excel file
        """
        try:
            self.df = pd.read_excel(data_path)
            logger.info(f"Loaded {len(self.df)} samples for sufficiency analysis")
            
            # Count issue types using the issue normalizer
            issue_counter = defaultdict(int)
            for issue_raw in self.df['issue_type'].dropna():
                normalized_issue, status, confidence = self.issue_normalizer.normalize_issue_type(str(issue_raw).strip())
                if normalized_issue:
                    issue_counter[normalized_issue] += 1
            self.issue_counts = dict(issue_counter)
            
            # Count categories using the category normalizer
            category_counter = defaultdict(int)
            for categories_str in self.df['category'].dropna():
                normalized_categories = self.category_normalizer.parse_and_normalize_categories(str(categories_str))
                for category in normalized_categories:
                    category_counter[category] += 1
            self.category_counts = dict(category_counter)
            
            logger.info(f"Analyzed {len(self.issue_counts)} issue types and "
                       f"{len(self.category_counts)} categories")
            
        except Exception as e:
            logger.error(f"Error analyzing data distribution from {data_path}: {e}")
            raise
    
    def get_sufficiency_level(self, count: int) -> str:
        """
        Determine sufficiency level based on sample count.
        
        Args:
            count: Number of training samples
            
        Returns:
            Sufficiency level string
        """
        if count < self.sufficiency_thresholds['critical']:
            return 'critical'
        elif count < self.sufficiency_thresholds['warning']:
            return 'warning'
        elif count < self.sufficiency_thresholds['good']:
            return 'good'
        elif count < self.sufficiency_thresholds['excellent']:
            return 'very_good'
        else:
            return 'excellent'
    
    def get_confidence_adjustment(self, item_type: str, item_name: str) -> float:
        """
        Get confidence adjustment multiplier based on data availability.
        
        Args:
            item_type: Either 'issue' or 'category'
            item_name: The specific issue type or category name
            
        Returns:
            Confidence adjustment multiplier (0-1)
        """
        if item_type == 'issue':
            count = self.issue_counts.get(item_name, 0)
        else:  # category
            count = self.category_counts.get(item_name, 0)
        
        level = self.get_sufficiency_level(count)
        return self.confidence_adjustments.get(level, 1.0)
    
    def generate_sufficiency_report(self) -> Dict:
        """
        Generate comprehensive data sufficiency report.
        
        Returns:
            Dictionary containing sufficiency analysis
        """
        if not self.issue_counts or not self.category_counts:
            logger.warning("No data loaded for sufficiency analysis")
            return {}
        
        report = {
            'summary': {
                'total_samples': len(self.df) if self.df is not None else 0,
                'unique_issue_types': len(self.issue_counts),
                'unique_categories': len(self.category_counts)
            },
            'critical_issues': [],
            'warning_issues': [],
            'good_issues': [],
            'critical_categories': [],
            'warning_categories': [],
            'good_categories': [],
            'recommendations': [],
            'statistics': {}
        }
        
        # Analyze issue types
        for issue, count in self.issue_counts.items():
            level = self.get_sufficiency_level(count)
            item_info = {
                'issue_type': issue,
                'sample_count': count,
                'sufficiency_level': level
            }
            
            if level == 'critical':
                item_info['status'] = 'CRITICAL - Needs immediate data collection'
                report['critical_issues'].append(item_info)
            elif level == 'warning':
                item_info['status'] = 'WARNING - Consider collecting more data'
                report['warning_issues'].append(item_info)
            elif level == 'good':
                item_info['status'] = 'GOOD - Acceptable data level'
                report['good_issues'].append(item_info)
        
        # Analyze categories
        for category, count in self.category_counts.items():
            level = self.get_sufficiency_level(count)
            item_info = {
                'category': category,
                'sample_count': count,
                'sufficiency_level': level
            }
            
            if level == 'critical':
                item_info['status'] = 'CRITICAL - Needs immediate data collection'
                report['critical_categories'].append(item_info)
            elif level == 'warning':
                item_info['status'] = 'WARNING - Consider collecting more data'
                report['warning_categories'].append(item_info)
            elif level == 'good':
                item_info['status'] = 'GOOD - Acceptable data level'
                report['good_categories'].append(item_info)
        
        # Generate recommendations
        if report['critical_issues'] or report['critical_categories']:
            report['recommendations'].append({
                'priority': 'URGENT',
                'message': f"{len(report['critical_issues'])} issue types and "
                          f"{len(report['critical_categories'])} categories have critically low data. "
                          f"Classification for these will be unreliable.",
                'action': 'Prioritize data collection for critical items immediately'
            })
        
        if report['warning_issues'] or report['warning_categories']:
            report['recommendations'].append({
                'priority': 'MEDIUM',
                'message': f"{len(report['warning_issues'])} issue types and "
                          f"{len(report['warning_categories'])} categories have low data. "
                          f"Consider prioritizing data collection for these.",
                'action': 'Plan data collection for warning-level items'
            })
        
        # Calculate statistics
        issue_counts_list = list(self.issue_counts.values())
        category_counts_list = list(self.category_counts.values())
        
        report['statistics'] = {
            'issues': {
                'mean_samples': sum(issue_counts_list) / len(issue_counts_list) if issue_counts_list else 0,
                'min_samples': min(issue_counts_list) if issue_counts_list else 0,
                'max_samples': max(issue_counts_list) if issue_counts_list else 0,
                'critical_percentage': (len(report['critical_issues']) / len(self.issue_counts) * 100) 
                                      if self.issue_counts else 0,
                'warning_percentage': (len(report['warning_issues']) / len(self.issue_counts) * 100)
                                     if self.issue_counts else 0
            },
            'categories': {
                'mean_samples': sum(category_counts_list) / len(category_counts_list) if category_counts_list else 0,
                'min_samples': min(category_counts_list) if category_counts_list else 0,
                'max_samples': max(category_counts_list) if category_counts_list else 0,
                'critical_percentage': (len(report['critical_categories']) / len(self.category_counts) * 100)
                                      if self.category_counts else 0,
                'warning_percentage': (len(report['warning_categories']) / len(self.category_counts) * 100)
                                     if self.category_counts else 0
            }
        }
        
        # Sort by sample count (ascending) to prioritize worst cases
        report['critical_issues'].sort(key=lambda x: x['sample_count'])
        report['warning_issues'].sort(key=lambda x: x['sample_count'])
        report['critical_categories'].sort(key=lambda x: x['sample_count'])
        report['warning_categories'].sort(key=lambda x: x['sample_count'])
        
        return report
    
    def apply_confidence_adjustments(self, classification: Dict) -> Dict:
        """
        Adjust classification confidence based on data sufficiency.
        
        Args:
            classification: Classification results to adjust
            
        Returns:
            Classification with adjusted confidences and warnings
        """
        adjusted = classification.copy()
        
        # Add data sufficiency warnings
        adjusted['data_sufficiency_warnings'] = []
        
        # Adjust issue confidences
        if 'identified_issues' in adjusted:
            for issue in adjusted['identified_issues']:
                issue_type = issue.get('issue_type', '')
                count = self.issue_counts.get(issue_type, 0)
                level = self.get_sufficiency_level(count)
                adjustment = self.get_confidence_adjustment('issue', issue_type)
                
                # Store original confidence
                if 'original_confidence' not in issue:
                    issue['original_confidence'] = issue.get('confidence', 1.0)
                
                # Apply adjustment
                issue['confidence'] = issue['original_confidence'] * adjustment
                issue['data_sufficiency'] = level
                issue['training_samples'] = count
                
                # Add warning if needed
                if level in ['critical', 'warning']:
                    adjusted['data_sufficiency_warnings'].append({
                        'type': 'issue',
                        'name': issue_type,
                        'level': level,
                        'sample_count': count,
                        'confidence_adjustment': adjustment,
                        'message': f"Low training data for '{issue_type}' ({count} samples) - "
                                  f"confidence reduced by {int((1-adjustment)*100)}%"
                    })
        
        # Adjust category confidences
        if 'categories' in adjusted:
            for category in adjusted['categories']:
                cat_name = category.get('category', '')
                count = self.category_counts.get(cat_name, 0)
                level = self.get_sufficiency_level(count)
                adjustment = self.get_confidence_adjustment('category', cat_name)
                
                # Store original confidence
                if 'original_confidence' not in category:
                    category['original_confidence'] = category.get('confidence', 1.0)
                
                # Apply adjustment
                category['confidence'] = category['original_confidence'] * adjustment
                category['data_sufficiency'] = level
                category['training_samples'] = count
                
                # Add warning if needed
                if level in ['critical', 'warning']:
                    adjusted['data_sufficiency_warnings'].append({
                        'type': 'category',
                        'name': cat_name,
                        'level': level,
                        'sample_count': count,
                        'confidence_adjustment': adjustment,
                        'message': f"Low training data for '{cat_name}' ({count} samples) - "
                                  f"confidence reduced by {int((1-adjustment)*100)}%"
                    })
        
        # Add summary warning if there are any issues
        if adjusted['data_sufficiency_warnings']:
            critical_count = sum(1 for w in adjusted['data_sufficiency_warnings'] 
                               if w['level'] == 'critical')
            warning_count = sum(1 for w in adjusted['data_sufficiency_warnings'] 
                              if w['level'] == 'warning')
            
            if critical_count > 0:
                logger.warning(f"Classification contains {critical_count} items with critical data insufficiency")
            if warning_count > 0:
                logger.warning(f"Classification contains {warning_count} items with warning-level data insufficiency")
        
        return adjusted
    
    def get_priority_items(self, top_n: int = 20) -> Dict[str, List]:
        """
        Get priority items for data collection.
        
        Args:
            top_n: Number of top priority items to return
            
        Returns:
            Dictionary with priority issue types and categories
        """
        # Sort by count (ascending) to get lowest first
        priority_issues = sorted(self.issue_counts.items(), key=lambda x: x[1])[:top_n]
        priority_categories = sorted(self.category_counts.items(), key=lambda x: x[1])[:top_n]
        
        return {
            'priority_issues': [
                {'issue_type': issue, 'current_samples': count, 'needed_samples': max(20 - count, 0)}
                for issue, count in priority_issues
            ],
            'priority_categories': [
                {'category': cat, 'current_samples': count, 'needed_samples': max(20 - count, 0)}
                for cat, count in priority_categories
            ]
        }
    
    def save_report(self, report: Dict, output_path: str):
        """
        Save sufficiency report to JSON file.
        
        Args:
            report: Report dictionary to save
            output_path: Path where to save the report
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Saved sufficiency report to {output_path}")
    
    def update_thresholds(self, thresholds: Dict[str, int]):
        """
        Update sufficiency thresholds.
        
        Args:
            thresholds: Dictionary with new threshold values
        """
        self.sufficiency_thresholds.update(thresholds)
        logger.info(f"Updated sufficiency thresholds: {self.sufficiency_thresholds}")
    
    def __repr__(self):
        return (f"DataSufficiencyAnalyzer(issues={len(self.issue_counts)}, "
                f"categories={len(self.category_counts)})")