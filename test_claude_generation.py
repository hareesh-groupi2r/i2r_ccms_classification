#!/usr/bin/env python3
"""
Test Claude API generation with a single issue type
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from claude_synthetic_generator import ClaudeSyntheticGenerator


def test_single_issue():
    """Test generation for a single issue type"""
    
    print("🧪 Testing Claude API with single issue")
    
    try:
        # Initialize generator
        generator = ClaudeSyntheticGenerator(
            training_path='data/synthetic/combined_training_data.xlsx',
            mapping_path='issue_category_mapping_diffs/unified_issue_category_mapping.xlsx'
        )
        
        # Test with one high-priority issue
        test_issue = "Delay in payment of stage payments"
        print(f"🎯 Testing with issue: {test_issue}")
        
        # Generate samples
        samples = generator.generate_samples_for_issue(test_issue, target_samples=3)
        
        if samples:
            print(f"\n✅ Generated {len(samples)} samples!")
            
            for i, sample in enumerate(samples, 1):
                print(f"\n--- Sample {i} ---")
                print(f"Subject: {sample['subject'][:100]}...")
                print(f"Body length: {len(sample['body'])} chars")
                print(f"Category: {sample['category']}")
                print(f"Type: {sample['correspondence_type']}")
            
            # Save test samples
            output_file = generator.save_samples(samples, "data/synthetic/test_claude_samples.xlsx")
            print(f"\n💾 Test samples saved to: {output_file}")
            
        else:
            print("❌ No samples generated")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_single_issue()