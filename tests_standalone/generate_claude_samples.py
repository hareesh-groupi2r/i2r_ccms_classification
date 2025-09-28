#!/usr/bin/env python3
"""
Generate High Priority Samples using Claude API
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from claude_synthetic_generator import ClaudeSyntheticGenerator
from generate_missing_training_data import MissingDataGenerator


def generate_high_priority_with_claude():
    """Generate samples for high priority missing issues using Claude"""
    
    print("🤖 Generating High Priority Samples with Claude API")
    print("=" * 60)
    
    # Get high priority missing issues
    print("📊 Analyzing missing issues...")
    missing_generator = MissingDataGenerator()
    categorized = missing_generator.categorize_missing_issues()
    high_priority_issues = categorized['high_priority']
    
    print(f"🔥 High priority issues identified: {len(high_priority_issues)}")
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
        print(f"❌ Failed to initialize Claude generator: {e}")
        return None
    
    # Generate samples
    print(f"\n🚀 Starting generation with Claude API...")
    print(f"📊 Target: 8 samples per issue = {len(high_priority_issues) * 8} total samples")
    print(f"💰 Estimated cost: ~$10-15 for Claude API")
    
    try:
        # Generate synthetic samples for high priority issues
        synthetic_samples = claude_generator.generate_priority_samples(
            priority_issues=high_priority_issues,
            samples_per_issue=8
        )
        
        if not synthetic_samples:
            print("❌ No samples were generated")
            return None
        
        # Save synthetic samples
        synthetic_file = claude_generator.save_samples(synthetic_samples)
        
        # Combine with training data
        enhanced_file = claude_generator.combine_with_training_data(synthetic_samples)
        
        print(f"\n🎉 SUCCESS!")
        print(f"📁 Synthetic samples: {synthetic_file}")
        print(f"📁 Enhanced training data: {enhanced_file}")
        print(f"📊 Generated samples: {len(synthetic_samples)}")
        print(f"🎯 Missing issues addressed: {len(set(s['issue_type'] for s in synthetic_samples))}")
        
        # Show sample quality
        print(f"\n📋 SAMPLE QUALITY CHECK:")
        for sample in synthetic_samples[:3]:
            print(f"  Issue: {sample['issue_type']}")
            print(f"  Subject: {sample['subject'][:80]}...")
            print(f"  Body length: {len(sample['body'])} chars")
            print()
        
        print(f"\n🔄 NEXT STEPS:")
        print(f"1. Update integrated backend to use: {enhanced_file}")
        print(f"2. Rebuild vector database with new training data")
        print(f"3. Test classification with enhanced coverage")
        
        return enhanced_file
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = generate_high_priority_with_claude()