#!/usr/bin/env python3
"""
Comprehensive analysis of all ground truth files vs mapper coverage
Searches for EDMS*.xlsx and LOT-*.xlsx files and analyzes missing mappings
"""

import pandas as pd
import sys
from pathlib import Path
import glob
import re
from collections import defaultdict

class CategoryNormalizer:
    """Normalize category names to standard format"""
    
    STANDARD_CATEGORIES = [
        "Authority's Obligations",
        "Contractor's Obligations", 
        "Change of Scope",
        "EoT",
        "Payments",
        "Dispute Resolution",
        "Appointed Date",
        "Completion Certificate",
        "Others"
    ]
    
    # Category normalization mapping
    CATEGORY_MAPPING = {
        # Authority variations
        "authority's obligations": "Authority's Obligations",
        "authoritys obligations": "Authority's Obligations", 
        "authority obligations": "Authority's Obligations",
        "authority's obigations": "Authority's Obligations",  # Common typo
        "authoritys obigations": "Authority's Obligations",
        "authority obigations": "Authority's Obligations",
        
        # Contractor variations
        "contractor's obligations": "Contractor's Obligations",
        "contractors obligations": "Contractor's Obligations",
        "contractor obligations": "Contractor's Obligations",
        "contactor's obligations": "Contractor's Obligations",  # Common typo
        "contactors obligations": "Contractor's Obligations",
        
        # Change of Scope variations
        "change of scope": "Change of Scope",
        "cos": "Change of Scope",
        "change of scope ": "Change of Scope",
        
        # EoT variations
        "eot": "EoT",
        "EOT": "EoT",
        "extension of time": "EoT",
        "eot ": "EoT",
        
        # Payments variations
        "payments": "Payments",
        "payment": "Payments",
        "payments ": "Payments",
        
        # Dispute Resolution variations
        "dispute resolution": "Dispute Resolution",
        "disputes": "Dispute Resolution",
        "dispute": "Dispute Resolution",
        "dispute resolution ": "Dispute Resolution",
        
        # Appointed Date variations
        "appointed date": "Appointed Date",
        "appointed date ": "Appointed Date",
        
        # Completion Certificate variations
        "completion certificate": "Completion Certificate",
        "completion certifcate": "Completion Certificate",  # Common typo
        "completion cert": "Completion Certificate",
        "completion certificate ": "Completion Certificate",
        
        # Others variations
        "others": "Others",
        "other": "Others",
        "others ": "Others"
    }
    
    @classmethod
    def normalize_category(cls, category_str):
        """Normalize a category string to standard format"""
        if pd.isna(category_str) or category_str == "":
            return None
            
        # Clean and normalize
        category_str = str(category_str).strip()
        
        # Try direct mapping first
        category_lower = category_str.lower()
        if category_lower in cls.CATEGORY_MAPPING:
            return cls.CATEGORY_MAPPING[category_lower]
        
        # Try exact match with standard categories
        if category_str in cls.STANDARD_CATEGORIES:
            return category_str
            
        # If no match found, return original but cleaned
        return category_str
    
    @classmethod
    def normalize_category_list(cls, category_str):
        """Normalize a comma-separated list of categories"""
        if pd.isna(category_str) or category_str == "":
            return []
            
        # Split by comma and normalize each
        categories = [cat.strip() for cat in str(category_str).split(',')]
        normalized = []
        
        for cat in categories:
            if cat:  # Skip empty strings
                norm_cat = cls.normalize_category(cat)
                if norm_cat and norm_cat not in normalized:
                    normalized.append(norm_cat)
        
        return sorted(normalized)

def find_ground_truth_files(base_dir):
    """Find all EDMS*.xlsx and LOT-*.xlsx files in directory and subdirectories"""
    
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"❌ Directory not found: {base_dir}")
        return []
    
    # Search patterns
    patterns = ["**/EDMS*.xlsx", "**/LOT-*.xlsx"]
    found_files = []
    
    for pattern in patterns:
        files = list(base_path.glob(pattern))
        found_files.extend(files)
    
    # Remove duplicates and sort
    found_files = sorted(list(set(found_files)))
    
    print(f"🔍 Found {len(found_files)} ground truth files:")
    for file in found_files:
        print(f"   📄 {file.relative_to(base_path)}")
    
    return found_files

def load_mapper_data():
    """Load and normalize mapper data"""
    
    mapper_path = "/Users/hareeshkb/work/Krishna/ccms_classification/unified_issue_category_mapping.xlsx"
    
    if not Path(mapper_path).exists():
        print(f"❌ Mapper file not found: {mapper_path}")
        return {}, {}
    
    try:
        df = pd.read_excel(mapper_path)
        
        # Find issue type column
        issue_col = None
        for col in ['issue_type', 'Issue_type', 'Issue Type', 'issue', 'Issue']:
            if col in df.columns:
                issue_col = col
                break
        
        if not issue_col:
            print(f"❌ No issue type column found in mapper")
            return {}, {}
        
        # Build mapper data
        mapper_issues = set(df[issue_col].dropna().unique())
        
        # Build issue-to-categories mapping
        mapper_mapping = {}
        if 'Categories' in df.columns:
            for _, row in df.iterrows():
                issue = row[issue_col]
                categories_str = row['Categories']
                if pd.notna(issue) and pd.notna(categories_str):
                    # Normalize categories
                    categories = CategoryNormalizer.normalize_category_list(categories_str)
                    if categories:
                        mapper_mapping[issue] = categories
        
        print(f"📊 Loaded {len(mapper_issues)} issue types from mapper")
        print(f"📊 Loaded {len(mapper_mapping)} issue-category mappings from mapper")
        
        # Export mapper data for reference
        export_mapper_data(df, issue_col)
        
        return mapper_issues, mapper_mapping
        
    except Exception as e:
        print(f"❌ Error reading mapper file: {e}")
        return {}, {}

def export_mapper_data(df, issue_col):
    """Export mapper data to Excel for reference"""
    
    try:
        # Create a clean export with normalized categories
        export_data = []
        
        for _, row in df.iterrows():
            issue = row[issue_col]
            categories_str = row.get('Categories', '')
            
            if pd.notna(issue):
                # Normalize categories
                if pd.notna(categories_str):
                    normalized_cats = CategoryNormalizer.normalize_category_list(categories_str)
                    categories_clean = ", ".join(normalized_cats) if normalized_cats else ""
                else:
                    categories_clean = ""
                
                export_data.append({
                    'Issue_Type': issue,
                    'Categories_Original': categories_str,
                    'Categories_Normalized': categories_clean,
                    'Category_Count': len(normalized_cats) if normalized_cats else 0
                })
        
        export_df = pd.DataFrame(export_data)
        export_path = "/Users/hareeshkb/work/Krishna/ccms_classification/mapper_reference_normalized.xlsx"
        export_df.to_excel(export_path, index=False)
        
        print(f"💾 EXPORTED MAPPER REFERENCE TO: {export_path}")
        
    except Exception as e:
        print(f"⚠️ Error exporting mapper reference: {e}")

def analyze_ground_truth_file(file_path, mapper_issues, mapper_mapping):
    """Analyze a single ground truth file"""
    
    try:
        # Try to read with header on row 3 first (like LOT-21)
        df = pd.read_excel(file_path, header=2)
        
        # Look for issues and category columns
        issues_col = None
        category_col = None
        
        # Common column names for issues
        issue_patterns = [
            'Issues discussed in the letter',
            'Issues discussed in the letter ',  # With trailing space
            'issue',
            'issues',
            'Issue Type',
            'Issue',
            'Problems'
        ]
        
        # Common column names for categories  
        category_patterns = [
            'Category',
            'category',
            'Categories',
            'Type',
            'Classification'
        ]
        
        # Find issues column
        for pattern in issue_patterns:
            if pattern in df.columns:
                issues_col = pattern
                break
        
        # Find category column
        for pattern in category_patterns:
            if pattern in df.columns:
                category_col = pattern
                break
        
        # If not found with row 3 header, try row 1
        if not issues_col or not category_col:
            df = pd.read_excel(file_path, header=0)
            
            # Try again
            for pattern in issue_patterns:
                if pattern in df.columns:
                    issues_col = pattern
                    break
            
            for pattern in category_patterns:
                if pattern in df.columns:
                    category_col = pattern
                    break
        
        if not issues_col or not category_col:
            return {
                'status': 'error',
                'error': f"Could not find issues or category columns. Available: {list(df.columns)}"
            }
        
        # Extract valid data
        valid_data = df[[issues_col, category_col]].dropna()
        
        if len(valid_data) == 0:
            return {
                'status': 'error', 
                'error': 'No valid issue-category pairs found'
            }
        
        # Analyze the data
        gt_issues = set(valid_data[issues_col].unique())
        
        # Build ground truth mapping with normalized categories
        gt_mapping = {}
        category_distribution = defaultdict(int)
        
        for _, row in valid_data.iterrows():
            issue = row[issues_col]
            category_str = row[category_col]
            
            # Normalize categories
            normalized_cats = CategoryNormalizer.normalize_category_list(category_str)
            
            if issue and normalized_cats:
                if issue not in gt_mapping:
                    gt_mapping[issue] = set()
                gt_mapping[issue].update(normalized_cats)
                
                # Count category usage
                for cat in normalized_cats:
                    category_distribution[cat] += 1
        
        # Find missing issues and mapping differences
        missing_issues = gt_issues - mapper_issues
        covered_issues = gt_issues & mapper_issues
        
        # Find mapping differences for covered issues
        mapping_differences = {}
        for issue in covered_issues:
            if issue in gt_mapping and issue in mapper_mapping:
                gt_cats = set(gt_mapping[issue])
                mapper_cats = set(mapper_mapping[issue])
                
                if gt_cats != mapper_cats:
                    mapping_differences[issue] = {
                        'ground_truth': sorted(list(gt_cats)),
                        'mapper': sorted(list(mapper_cats)),
                        'missing_in_mapper': sorted(list(gt_cats - mapper_cats)),
                        'extra_in_mapper': sorted(list(mapper_cats - gt_cats))
                    }
        
        return {
            'status': 'success',
            'file_name': file_path.name,
            'total_issues': len(gt_issues),
            'covered_issues': len(covered_issues),
            'missing_issues': len(missing_issues),
            'coverage_percent': (len(covered_issues) / len(gt_issues) * 100) if gt_issues else 0,
            'missing_issue_list': sorted(list(missing_issues)),
            'gt_mapping': {k: sorted(list(v)) for k, v in gt_mapping.items()},
            'mapping_differences': mapping_differences,
            'category_distribution': dict(category_distribution),
            'issues_col': issues_col,
            'category_col': category_col
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f"Error processing file: {str(e)}"
        }

def generate_comprehensive_report(analyses, mapper_issues, mapper_mapping):
    """Generate comprehensive analysis report"""
    
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE GROUND TRUTH ANALYSIS REPORT")
    print("="*80)
    
    # Summary statistics
    total_files = len(analyses)
    successful_files = len([a for a in analyses if a['status'] == 'success'])
    failed_files = total_files - successful_files
    
    print(f"\n📈 SUMMARY STATISTICS:")
    print(f"   📄 Total files processed: {total_files}")
    print(f"   ✅ Successfully analyzed: {successful_files}")
    print(f"   ❌ Failed to analyze: {failed_files}")
    
    if failed_files > 0:
        print(f"\n❌ FAILED FILES:")
        for analysis in analyses:
            if analysis['status'] == 'error':
                print(f"   • {analysis.get('file_name', 'Unknown')}: {analysis['error']}")
    
    # Process successful analyses
    successful_analyses = [a for a in analyses if a['status'] == 'success']
    
    if not successful_analyses:
        print("\n❌ No successful analyses to report")
        return
    
    # Coverage analysis
    print(f"\n📊 COVERAGE ANALYSIS BY FILE:")
    print("-" * 60)
    
    total_coverage_sum = 0
    for analysis in successful_analyses:
        coverage = analysis['coverage_percent']
        total_coverage_sum += coverage
        print(f"   📄 {analysis['file_name']}: {coverage:.1f}% "
              f"({analysis['covered_issues']}/{analysis['total_issues']} issues)")
    
    avg_coverage = total_coverage_sum / len(successful_analyses)
    print(f"\n   📈 Average coverage across all files: {avg_coverage:.1f}%")
    
    # Aggregate missing issues across all files
    all_missing_issues = set()
    missing_by_file = {}
    
    for analysis in successful_analyses:
        file_missing = set(analysis['missing_issue_list'])
        all_missing_issues.update(file_missing)
        if file_missing:
            missing_by_file[analysis['file_name']] = file_missing
    
    print(f"\n❌ MISSING ISSUES AGGREGATE ANALYSIS:")
    print("-" * 60)
    print(f"   📊 Total unique missing issues across all files: {len(all_missing_issues)}")
    
    # Show missing issues by frequency
    issue_frequency = defaultdict(int)
    issue_files = defaultdict(list)
    
    for analysis in successful_analyses:
        for issue in analysis['missing_issue_list']:
            issue_frequency[issue] += 1
            issue_files[issue].append(analysis['file_name'])
    
    # Sort by frequency
    sorted_missing = sorted(issue_frequency.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n🔥 TOP MISSING ISSUES (by frequency):")
    print("-" * 60)
    for issue, freq in sorted_missing[:15]:  # Top 15
        files_str = ", ".join(issue_files[issue][:3])  # Show first 3 files
        if len(issue_files[issue]) > 3:
            files_str += f" (+{len(issue_files[issue])-3} more)"
        print(f"   • '{issue}' (appears in {freq} files)")
        print(f"     Files: {files_str}")
    
    # Category distribution analysis
    print(f"\n📊 CATEGORY DISTRIBUTION ANALYSIS:")
    print("-" * 60)
    
    all_categories = defaultdict(int)
    for analysis in successful_analyses:
        for cat, count in analysis['category_distribution'].items():
            all_categories[cat] += count
    
    sorted_categories = sorted(all_categories.items(), key=lambda x: x[1], reverse=True)
    
    for cat, total_count in sorted_categories:
        print(f"   📂 {cat}: {total_count} occurrences across all files")
    
    # Mapping differences analysis
    print(f"\n🔄 MAPPING DIFFERENCES ANALYSIS:")
    print("-" * 60)
    
    all_mapping_diffs = {}
    for analysis in successful_analyses:
        for issue, diff in analysis['mapping_differences'].items():
            if issue not in all_mapping_diffs:
                all_mapping_diffs[issue] = {
                    'files': [],
                    'ground_truth_sets': [],
                    'mapper_cats': diff['mapper']
                }
            all_mapping_diffs[issue]['files'].append(analysis['file_name'])
            all_mapping_diffs[issue]['ground_truth_sets'].append(set(diff['ground_truth']))
    
    if all_mapping_diffs:
        print(f"   📊 Issues with mapping differences: {len(all_mapping_diffs)}")
        
        for issue, data in list(all_mapping_diffs.items())[:10]:  # Show top 10
            # Find consensus ground truth categories
            all_gt_cats = set()
            for cat_set in data['ground_truth_sets']:
                all_gt_cats.update(cat_set)
            
            print(f"\n   🔄 '{issue}' (appears in {len(data['files'])} files)")
            print(f"      Mapper categories: {data['mapper_cats']}")
            print(f"      Ground truth categories: {sorted(list(all_gt_cats))}")
            print(f"      Files: {', '.join(data['files'][:3])}")
    
    # Export comprehensive results
    export_comprehensive_results(successful_analyses, all_missing_issues, all_mapping_diffs)

def export_comprehensive_results(analyses, all_missing_issues, all_mapping_diffs):
    """Export comprehensive results to Excel files"""
    
    try:
        # Export 1: Missing issues summary
        missing_data = []
        for analysis in analyses:
            for issue in analysis['missing_issue_list']:
                # Try to get categories from ground truth mapping
                categories = analysis['gt_mapping'].get(issue, [])
                missing_data.append({
                    'File': analysis['file_name'],
                    'Missing_Issue': issue,
                    'Ground_Truth_Categories': ", ".join(categories),
                    'Source': 'ground_truth_analysis'
                })
        
        if missing_data:
            missing_df = pd.DataFrame(missing_data)
            missing_path = "/Users/hareeshkb/work/Krishna/ccms_classification/all_missing_issues_comprehensive.xlsx"
            missing_df.to_excel(missing_path, index=False)
            print(f"\n💾 EXPORTED MISSING ISSUES TO: {missing_path}")
            print(f"   📄 {len(missing_data)} missing issue-category mappings")
        
        # Export 2: File-by-file analysis
        file_analysis_data = []
        for analysis in analyses:
            file_analysis_data.append({
                'File_Name': analysis['file_name'],
                'Status': analysis['status'],
                'Total_Issues': analysis.get('total_issues', 0),
                'Covered_Issues': analysis.get('covered_issues', 0),
                'Missing_Issues': analysis.get('missing_issues', 0),
                'Coverage_Percent': analysis.get('coverage_percent', 0),
                'Issues_Column': analysis.get('issues_col', ''),
                'Category_Column': analysis.get('category_col', ''),
                'Top_Categories': ", ".join([f"{k}({v})" for k, v in 
                                           sorted(analysis.get('category_distribution', {}).items(), 
                                                 key=lambda x: x[1], reverse=True)[:5]])
            })
        
        file_df = pd.DataFrame(file_analysis_data)
        file_path = "/Users/hareeshkb/work/Krishna/ccms_classification/file_by_file_analysis.xlsx"
        file_df.to_excel(file_path, index=False)
        print(f"\n💾 EXPORTED FILE ANALYSIS TO: {file_path}")
        
        # Export 3: Mapping differences
        if all_mapping_diffs:
            mapping_diff_data = []
            for issue, data in all_mapping_diffs.items():
                # Aggregate ground truth categories across files
                all_gt_cats = set()
                for cat_set in data['ground_truth_sets']:
                    all_gt_cats.update(cat_set)
                
                mapping_diff_data.append({
                    'Issue_Type': issue,
                    'Files_Count': len(data['files']),
                    'Files': ", ".join(data['files']),
                    'Mapper_Categories': ", ".join(data['mapper_cats']),
                    'Ground_Truth_Categories': ", ".join(sorted(list(all_gt_cats))),
                    'Missing_In_Mapper': ", ".join(sorted(list(all_gt_cats - set(data['mapper_cats'])))),
                    'Extra_In_Mapper': ", ".join(sorted(list(set(data['mapper_cats']) - all_gt_cats)))
                })
            
            mapping_df = pd.DataFrame(mapping_diff_data)
            mapping_path = "/Users/hareeshkb/work/Krishna/ccms_classification/mapping_differences_analysis.xlsx"
            mapping_df.to_excel(mapping_path, index=False)
            print(f"\n💾 EXPORTED MAPPING DIFFERENCES TO: {mapping_path}")
        
    except Exception as e:
        print(f"\n⚠️ Error exporting comprehensive results: {e}")

def main():
    """Main analysis function"""
    
    # Configuration
    base_directory = "/Users/hareeshkb/work/Krishna/ccms_classification/data"
    
    print("🔍 COMPREHENSIVE GROUND TRUTH ANALYSIS")
    print("="*70)
    print(f"📂 Searching in: {base_directory}")
    
    # Find all ground truth files
    gt_files = find_ground_truth_files(base_directory)
    
    if not gt_files:
        print("❌ No ground truth files found")
        return
    
    # Load mapper data
    mapper_issues, mapper_mapping = load_mapper_data()
    
    if not mapper_issues:
        print("❌ Failed to load mapper data")
        return
    
    # Analyze each file
    print(f"\n🔄 ANALYZING {len(gt_files)} FILES...")
    print("-" * 70)
    
    analyses = []
    
    for i, file_path in enumerate(gt_files, 1):
        print(f"\n📄 [{i}/{len(gt_files)}] Analyzing: {file_path.name}")
        
        analysis = analyze_ground_truth_file(file_path, mapper_issues, mapper_mapping)
        analysis['file_path'] = str(file_path)
        analyses.append(analysis)
        
        if analysis['status'] == 'success':
            print(f"   ✅ Coverage: {analysis['coverage_percent']:.1f}% "
                  f"({analysis['covered_issues']}/{analysis['total_issues']} issues)")
            if analysis['missing_issues'] > 0:
                print(f"   📉 Missing: {analysis['missing_issues']} issues")
        else:
            print(f"   ❌ Error: {analysis['error']}")
    
    # Generate comprehensive report
    generate_comprehensive_report(analyses, mapper_issues, mapper_mapping)
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"📊 Processed {len(gt_files)} files")

if __name__ == "__main__":
    main()