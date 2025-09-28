#!/usr/bin/env python3
"""
Generate High Priority Missing Training Samples
Focus on the 12 high-priority missing issue types first
"""

import os
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from generate_missing_training_data import MissingDataGenerator


def generate_high_priority_samples():
    """Generate samples for high priority missing issues only"""
    
    print("🎯 Generating High Priority Missing Training Samples")
    print("=" * 60)
    
    # Initialize generator
    generator = MissingDataGenerator()
    
    # Get categorized missing issues
    categorized = generator.categorize_missing_issues()
    high_priority_issues = categorized['high_priority']
    
    print(f"🔥 High priority issues to address: {len(high_priority_issues)}")
    
    for i, issue in enumerate(high_priority_issues, 1):
        print(f"  {i:2d}. {issue}")
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("\n❌ OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return None
    
    print(f"\n🤖 OpenAI API key found. Starting generation...")
    
    # Generate samples for high priority issues only
    print(f"\n🔄 Generating 8 samples each for {len(high_priority_issues)} high priority issues...")
    print(f"📊 Total samples to generate: {len(high_priority_issues) * 8}")
    print(f"💰 Estimated API cost: ~$15-25")
    
    # Auto-proceed for automated execution
    print("\n🚀 Auto-proceeding with generation...")
    
    # Generate samples
    try:
        # Focus only on high priority issues
        focused_categorized = {
            'high_priority': high_priority_issues,
            'medium_priority': [],
            'low_priority': []
        }
        
        # Temporarily modify the generator's categorization
        original_method = generator.categorize_missing_issues
        generator.categorize_missing_issues = lambda: focused_categorized
        
        # Generate synthetic samples
        synthetic_samples = generator.generate_synthetic_samples_for_missing(
            api_key=api_key,
            samples_per_issue=8
        )
        
        # Restore original method
        generator.categorize_missing_issues = original_method
        
        if synthetic_samples:
            # Save the enhanced dataset
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = generator.save_enhanced_training_data(
                synthetic_samples, 
                f"data/synthetic/enhanced_training_high_priority_{timestamp}.xlsx"
            )
            
            print(f"\n✅ SUCCESS: Enhanced training data saved")
            print(f"📁 File: {output_path}")
            print(f"📊 Added {len(synthetic_samples)} synthetic samples")
            print(f"🎯 Addressed {len(high_priority_issues)} high priority missing issues")
            
            # Next steps guidance
            print(f"\n🔄 NEXT STEPS:")
            print(f"1. Update the integrated backend to use the new training file")
            print(f"2. Rebuild the vector database index")
            print(f"3. Test classification on documents with these issue types")
            print(f"4. Generate medium/low priority samples if needed")
            
            return output_path
            
        else:
            print("❌ No samples were generated")
            return None
            
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        return None


if __name__ == "__main__":
    generate_high_priority_samples()