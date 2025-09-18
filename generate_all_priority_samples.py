#!/usr/bin/env python3
"""
Generate All High Priority Missing Training Samples with Claude API
Optimized for efficient generation
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from claude_synthetic_generator import ClaudeSyntheticGenerator
from generate_missing_training_data import MissingDataGenerator


def generate_all_priority_samples():
    """Generate samples for all high priority missing issues efficiently"""
    
    print("🚀 Generating ALL High Priority Samples with Claude API")
    print("=" * 70)
    
    # Get high priority missing issues
    print("📊 Analyzing gaps...")
    missing_generator = MissingDataGenerator()
    categorized = missing_generator.categorize_missing_issues()
    high_priority_issues = categorized['high_priority']
    
    print(f"\n🔥 High Priority Issues ({len(high_priority_issues)}):")
    for i, issue in enumerate(high_priority_issues, 1):
        print(f"  {i:2d}. {issue}")
    
    # Initialize Claude generator
    print(f"\n🤖 Initializing Claude generator...")
    try:
        claude_generator = ClaudeSyntheticGenerator(
            training_path='data/synthetic/combined_training_data.xlsx',
            mapping_path='issue_category_mapping_diffs/unified_issue_category_mapping.xlsx'
        )
        print("✅ Claude generator ready")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return None
    
    # Generate samples efficiently
    print(f"\n🎯 Generation Plan:")
    print(f"   📊 Issues: {len(high_priority_issues)}")
    print(f"   📊 Samples per issue: 6")
    print(f"   📊 Total samples: {len(high_priority_issues) * 6}")
    print(f"   💰 Estimated cost: ~$8-12")
    print(f"   ⏱️  Estimated time: ~10-15 minutes")
    
    all_samples = []
    successful_issues = []
    failed_issues = []
    
    start_time = time.time()
    
    for i, issue_type in enumerate(high_priority_issues, 1):
        print(f"\n📝 [{i:2d}/{len(high_priority_issues)}] Generating for: {issue_type}")
        
        try:
            # Generate 6 samples per issue (good balance of coverage vs cost)
            samples = claude_generator.generate_samples_for_issue(issue_type, target_samples=6)
            
            if samples:
                all_samples.extend(samples)
                successful_issues.append(issue_type)
                print(f"    ✅ Generated {len(samples)} samples | Total: {len(all_samples)}")
            else:
                failed_issues.append(issue_type)
                print(f"    ❌ No samples generated")
            
            # Save progress every 3 issues
            if i % 3 == 0 and all_samples:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                progress_file = f"data/synthetic/progress_backup_{timestamp}.xlsx"
                claude_generator.save_samples(all_samples, progress_file)
                print(f"    💾 Progress saved: {len(all_samples)} samples")
            
            # Rate limiting - be respectful to API
            if i < len(high_priority_issues):
                time.sleep(1)  # 1 second between requests
                
        except Exception as e:
            failed_issues.append(issue_type)
            print(f"    ❌ Error: {str(e)[:100]}...")
            time.sleep(2)  # Longer wait on error
            continue
    
    elapsed_time = time.time() - start_time
    
    # Results summary
    print(f"\n{'='*70}")
    print(f"🎉 GENERATION COMPLETE!")
    print(f"⏱️  Total time: {elapsed_time/60:.1f} minutes")
    print(f"✅ Successful issues: {len(successful_issues)}/{len(high_priority_issues)}")
    print(f"📊 Total samples generated: {len(all_samples)}")
    
    if failed_issues:
        print(f"\n❌ Failed issues ({len(failed_issues)}):")
        for issue in failed_issues:
            print(f"   - {issue}")
    
    if all_samples:
        # Save final synthetic samples
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        synthetic_file = claude_generator.save_samples(
            all_samples, 
            f"data/synthetic/claude_priority_samples_{timestamp}.xlsx"
        )
        
        # Combine with training data
        enhanced_file = claude_generator.combine_with_training_data(
            all_samples,
            f"data/synthetic/enhanced_training_priority_{timestamp}.xlsx"
        )
        
        print(f"\n📁 FILES CREATED:")
        print(f"   🔬 Synthetic only: {synthetic_file}")
        print(f"   📚 Enhanced training: {enhanced_file}")
        
        # Quality check
        print(f"\n📊 QUALITY METRICS:")
        categories = set(s['category'] for s in all_samples)
        correspondence_types = set(s.get('correspondence_type', 'unknown') for s in all_samples)
        avg_subject_length = sum(len(s['subject']) for s in all_samples) / len(all_samples)
        avg_body_length = sum(len(s['body']) for s in all_samples) / len(all_samples)
        
        print(f"   📋 Categories covered: {len(categories)}")
        print(f"   📧 Correspondence types: {len(correspondence_types)}")
        print(f"   📝 Avg subject length: {avg_subject_length:.0f} chars")
        print(f"   📄 Avg body length: {avg_body_length:.0f} chars")
        
        print(f"\n🔄 NEXT STEPS:")
        print(f"1. Update integrated backend configuration:")
        print(f"   training_data_path = '{enhanced_file}'")
        print(f"2. Rebuild vector database index")
        print(f"3. Test classification coverage")
        print(f"4. Generate medium/low priority issues if needed")
        
        return enhanced_file
    else:
        print(f"\n❌ No samples were generated successfully")
        return None


if __name__ == "__main__":
    result = generate_all_priority_samples()