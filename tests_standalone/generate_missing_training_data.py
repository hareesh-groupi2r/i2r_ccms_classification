#!/usr/bin/env python3
"""
Generate Missing Training Data for CCMS Classification
Addresses the 78 missing issue types that exist in mapping but have no training samples
"""

import sys
import os
from pathlib import Path
import pandas as pd
import json
from typing import List, Dict, Set
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from classifier.unified_issue_mapper import UnifiedIssueCategoryMapper
from classifier.data_sufficiency import DataSufficiencyAnalyzer
from synthetic_data_generator import SyntheticDataGenerator


class MissingDataGenerator:
    """
    Identifies and generates training data for missing issue types
    """
    
    def __init__(self):
        """Initialize the missing data generator"""
        self.training_path = 'data/synthetic/combined_training_data.xlsx'
        self.mapping_path = 'issue_category_mapping_diffs/unified_issue_category_mapping.xlsx'
        
        # Load components
        self.mapper = UnifiedIssueCategoryMapper(self.training_path, self.mapping_path)
        self.analyzer = DataSufficiencyAnalyzer(self.training_path)
        
        # Get the gap analysis
        self._analyze_gaps()
    
    def _analyze_gaps(self):
        """Analyze gaps between mapping and training data"""
        # Get all issue types from mapping
        self.all_mapping_issues = set(self.mapper.get_all_issue_types())
        
        # Get all issue types from training data
        self.training_issues = set(self.analyzer.df['issue_type'].unique())
        
        # Find missing issue types
        self.missing_issues = self.all_mapping_issues - self.training_issues
        
        # Get training data sufficiency report
        self.sufficiency_report = self.analyzer.generate_sufficiency_report()
        
        print(f"=== GAP ANALYSIS RESULTS ===")
        print(f"📊 Total issue types in mapping: {len(self.all_mapping_issues)}")
        print(f"📊 Total issue types in training: {len(self.training_issues)}")
        print(f"🔍 Missing issue types: {len(self.missing_issues)}")
        print(f"⚠️  Warning issues (low data): {len(self.sufficiency_report['warning_issues'])}")
    
    def categorize_missing_issues(self) -> Dict[str, List[str]]:
        """
        Categorize missing issues by business importance
        """
        # Define business priority categories
        high_priority_keywords = [
            'authority', 'contractor', 'obligation', 'payment', 'change of scope', 
            'eot', 'extension', 'delay', 'approval', 'clearance', 'appointed date'
        ]
        
        medium_priority_keywords = [
            'design', 'drawing', 'submission', 'inspection', 'mobilization',
            'safety', 'quality', 'construction', 'schedule'
        ]
        
        categorized = {
            'high_priority': [],
            'medium_priority': [], 
            'low_priority': []
        }
        
        for issue in self.missing_issues:
            issue_lower = issue.lower()
            
            if any(keyword in issue_lower for keyword in high_priority_keywords):
                categorized['high_priority'].append(issue)
            elif any(keyword in issue_lower for keyword in medium_priority_keywords):
                categorized['medium_priority'].append(issue)
            else:
                categorized['low_priority'].append(issue)
        
        return categorized
    
    def generate_synthetic_samples_for_missing(self, api_key: str, samples_per_issue: int = 8):
        """
        Generate synthetic training samples for missing issue types
        """
        if not api_key:
            print("❌ OpenAI API key required for synthetic data generation")
            return None
            
        # Initialize synthetic data generator
        generator = SyntheticDataGenerator(self.training_path, api_key)
        
        # Get categorized missing issues
        categorized = self.categorize_missing_issues()
        
        print(f"\n=== SYNTHETIC DATA GENERATION PLAN ===")
        print(f"🔥 High priority issues: {len(categorized['high_priority'])}")
        print(f"⚡ Medium priority issues: {len(categorized['medium_priority'])}")
        print(f"📝 Low priority issues: {len(categorized['low_priority'])}")
        print(f"📊 Target samples per issue: {samples_per_issue}")
        print(f"📊 Total samples to generate: {len(self.missing_issues) * samples_per_issue}")
        
        # Start with high priority issues
        all_generated_samples = []
        
        for priority, issues in categorized.items():
            if not issues:
                continue
                
            print(f"\n🔄 Generating samples for {priority} issues ({len(issues)} issues)...")
            
            for i, issue_type in enumerate(issues):
                print(f"  📝 Generating {samples_per_issue} samples for: {issue_type}")
                
                try:
                    # Get the category mapping for this issue
                    categories = self.mapper.get_categories_for_issue(issue_type)
                    if not categories:
                        categories = ['Others']  # Fallback
                    
                    # Generate synthetic samples using the correct method
                    primary_category = categories[0] if categories else 'Others'
                    samples = generator.generate_llm_synthetic_data(
                        issue_type=issue_type,
                        category=primary_category,
                        current_samples=0,
                        target_samples=samples_per_issue
                    )
                    
                    all_generated_samples.extend(samples)
                    print(f"    ✅ Generated {len(samples)} samples")
                    
                except Exception as e:
                    print(f"    ❌ Failed to generate samples for {issue_type}: {e}")
                    continue
        
        print(f"\n✅ Generated {len(all_generated_samples)} total synthetic samples")
        return all_generated_samples
    
    def save_enhanced_training_data(self, synthetic_samples: List[Dict], output_path: str = None):
        """
        Combine original training data with synthetic samples and save
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/synthetic/enhanced_training_data_{timestamp}.xlsx"
        
        # Load original training data
        original_df = pd.read_excel(self.training_path)
        
        # Convert synthetic samples to DataFrame
        synthetic_df = pd.DataFrame(synthetic_samples)
        
        # Combine datasets
        enhanced_df = pd.concat([original_df, synthetic_df], ignore_index=True)
        
        # Save enhanced dataset
        enhanced_df.to_excel(output_path, index=False)
        
        print(f"\n✅ Enhanced training data saved to: {output_path}")
        print(f"📊 Original samples: {len(original_df)}")
        print(f"📊 Synthetic samples: {len(synthetic_df)}")
        print(f"📊 Total samples: {len(enhanced_df)}")
        print(f"📊 Unique issue types: {enhanced_df['issue_type'].nunique()}")
        
        return output_path
    
    def generate_coverage_report(self) -> Dict:
        """
        Generate a comprehensive coverage report
        """
        categorized = self.categorize_missing_issues()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_mapping_issues': len(self.all_mapping_issues),
                'total_training_issues': len(self.training_issues),
                'missing_issues': len(self.missing_issues),
                'coverage_percentage': (len(self.training_issues) / len(self.all_mapping_issues)) * 100
            },
            'missing_by_priority': {
                'high_priority': categorized['high_priority'],
                'medium_priority': categorized['medium_priority'],
                'low_priority': categorized['low_priority']
            },
            'training_sufficiency': {
                'warning_issues': len(self.sufficiency_report['warning_issues']),
                'good_issues': len(self.sufficiency_report['good_issues'])
            }
        }
        
        return report


def main():
    """Main execution function"""
    print("🚀 CCMS Missing Training Data Generator")
    print("=" * 50)
    
    # Initialize generator
    generator = MissingDataGenerator()
    
    # Generate coverage report
    report = generator.generate_coverage_report()
    
    print(f"\n📊 COVERAGE REPORT:")
    print(f"  Total issue types in mapping: {report['summary']['total_mapping_issues']}")
    print(f"  Issue types with training data: {report['summary']['total_training_issues']}")
    print(f"  Missing issue types: {report['summary']['missing_issues']}")
    print(f"  Current coverage: {report['summary']['coverage_percentage']:.1f}%")
    
    print(f"\n🎯 MISSING ISSUES BY PRIORITY:")
    for priority, issues in report['missing_by_priority'].items():
        print(f"  {priority}: {len(issues)} issues")
        if issues:
            print(f"    Examples: {', '.join(issues[:3])}{'...' if len(issues) > 3 else ''}")
    
    # Save report
    report_path = f"data/reports/missing_data_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📝 Analysis report saved to: {report_path}")
    
    # Check for API key to generate samples
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print(f"\n🤖 OpenAI API key found. Ready to generate synthetic samples.")
        print(f"To generate samples, call: generate_synthetic_samples_for_missing()")
    else:
        print(f"\n⚠️  OpenAI API key not found. Set OPENAI_API_KEY to generate synthetic samples.")
    
    return generator


if __name__ == "__main__":
    generator = main()