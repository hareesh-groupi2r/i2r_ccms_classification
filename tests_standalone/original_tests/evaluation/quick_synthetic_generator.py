#!/usr/bin/env python3
"""
Quick Synthetic Data Generator
Simple script to generate synthetic data for critical issue types using OpenAI API
"""

import sys
from pathlib import Path
import pandas as pd
import os
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from synthetic_data_generator import SyntheticDataGenerator

def main():
    print("=" * 60)
    print("QUICK SYNTHETIC DATA GENERATOR")
    print("=" * 60)
    
    # Load environment
    load_dotenv()
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key.startswith('your-'):
        print("❌ OpenAI API key not found or invalid.")
        print("   Set your API key in .env file: OPENAI_API_KEY=sk-...")
        print("   Will use template-based generation only.")
        use_llm = False
    else:
        print("✅ OpenAI API key found. Using LLM + template generation.")
        use_llm = True
    
    # Initialize generator
    generator = SyntheticDataGenerator(
        training_data_path='data/raw/Consolidated_labeled_data.xlsx',
        openai_api_key=api_key if use_llm else None
    )
    
    print(f"\n📊 Data Analysis:")
    print(f"  Critical issues (<5 samples): {len(generator.critical_issues)}")
    print(f"  Warning issues (5-10 samples): {len(generator.warning_issues)}")
    
    # Ask user for generation parameters
    print(f"\n🎯 Generation Options:")
    
    if use_llm:
        print("  1. Quick generation (5 samples per critical issue, LLM + templates)")
        print("  2. Moderate generation (8 samples per critical issue, LLM + templates)")  
        print("  3. Full generation (10 samples per critical issue, LLM + templates)")
        print("  4. Template-only (no API costs)")
        
        # Auto-select Option 4: Template-only (no API costs)
        choice = "4"
        print("Auto-selecting Option 4: Template-only (no API costs)")
        print("📋 This regeneration will fix category normalization issues")
        
        if choice == "1":
            target_samples = 5
            use_llm = True
        elif choice == "2":
            target_samples = 8
            use_llm = True
        elif choice == "3":
            target_samples = 10
            use_llm = True
        elif choice == "4":
            target_samples = 8
            use_llm = False
        else:
            print("Invalid choice. Using quick generation.")
            target_samples = 5
            use_llm = True
    else:
        print("  Template-based generation only (no API costs)")
        target_samples = 8
        use_llm = False
    
    # Estimate cost for LLM generation
    if use_llm:
        estimated_samples = sum(max(0, target_samples - count) for _, count in generator.critical_issues[:20])  # Top 20 critical
        estimated_cost = (estimated_samples * 0.03)  # Rough estimate: $0.03 per sample
        print(f"\n💰 Estimated cost: ~${estimated_cost:.2f} for ~{estimated_samples} samples")
        
        # Auto-confirm since we're using template-only (no costs)
        print("Auto-confirming: Template generation has no costs")
        confirm = 'y'
    
    # Generate synthetic data
    print(f"\n🚀 Starting generation...")
    print(f"   Target samples per issue: {target_samples}")
    print(f"   Using LLM: {'Yes' if use_llm else 'No'}")
    print(f"   Using Templates: Yes")
    
    synthetic_df = generator.generate_synthetic_dataset(
        target_min_samples=target_samples,
        use_llm=use_llm,
        use_templates=True
    )
    
    if len(synthetic_df) > 0:
        # Show summary
        print(f"\n✅ Generation Complete!")
        print(f"   Total samples generated: {len(synthetic_df)}")
        print(f"   Issue types covered: {synthetic_df['issue_type'].nunique()}")
        print(f"   Categories covered: {synthetic_df['category'].nunique()}")
        
        # Show generation methods
        methods = synthetic_df['generation_method'].value_counts()
        print(f"\n📊 Generation Methods:")
        for method, count in methods.items():
            print(f"   {method}: {count} samples")
        
        # Show top issues generated
        print(f"\n🏆 Top Issue Types Generated:")
        top_issues = synthetic_df['issue_type'].value_counts().head(10)
        for issue, count in top_issues.items():
            print(f"   {issue[:50]:<50} {count} samples")
        
        # Save synthetic data
        output_dir = Path('./data/synthetic')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save synthetic-only data
        synthetic_path = output_dir / 'synthetic_samples.xlsx'
        generator.save_synthetic_data(synthetic_df, synthetic_path)
        
        # Create combined dataset with original data
        original_df = pd.read_excel('data/raw/Consolidated_labeled_data.xlsx')
        original_df['is_synthetic'] = False
        original_df['generation_method'] = 'original_data'
        
        combined_df = pd.concat([original_df, synthetic_df], ignore_index=True)
        combined_path = output_dir / 'combined_training_data.xlsx'
        combined_df.to_excel(combined_path, index=False)
        
        print(f"\n💾 Files Saved:")
        print(f"   Synthetic only: {synthetic_path}")
        print(f"   Combined data: {combined_path}")
        print(f"   Original: 523 samples")
        print(f"   Combined: {len(combined_df)} samples (+{len(synthetic_df)} synthetic)")
        
        # Create validation splits
        train_df, val_df = generator.create_overfitting_prevention_splits(synthetic_df)
        
        train_path = output_dir / 'training_set.xlsx'
        val_path = output_dir / 'validation_set.xlsx'
        
        train_df.to_excel(train_path, index=False)
        val_df.to_excel(val_path, index=False)
        
        print(f"\n📂 Training Splits Created:")
        print(f"   Training set: {train_path} ({len(train_df)} samples)")
        print(f"   Validation set: {val_path} ({len(val_df)} samples)")
        print(f"   Validation is {val_df['is_synthetic'].sum()/len(val_df)*100:.1f}% synthetic")
        
        # Show updated data sufficiency
        print(f"\n📈 Data Sufficiency Impact:")
        
        # Count critical issues after synthetic data
        original_critical = len(generator.critical_issues)
        
        # Simulate updated counts
        updated_counts = {}
        for issue_type in combined_df['issue_type'].unique():
            updated_counts[issue_type] = len(combined_df[combined_df['issue_type'] == issue_type])
        
        new_critical = sum(1 for count in updated_counts.values() if count < 5)
        new_warning = sum(1 for count in updated_counts.values() if 5 <= count < 10)
        new_good = sum(1 for count in updated_counts.values() if count >= 10)
        
        print(f"   Critical issues: {original_critical} → {new_critical} ({original_critical - new_critical} improved)")
        print(f"   Warning issues: {len(generator.warning_issues)} → {new_warning}")
        print(f"   Good issues: 12 → {new_good} (+{new_good - 12} improved)")
        
        print(f"\n🎯 Next Steps:")
        print(f"   1. Review generated samples for quality")
        print(f"   2. Use combined_training_data.xlsx for model training")
        print(f"   3. Always validate on real data only")
        print(f"   4. Monitor for overfitting during training")
        print(f"   5. Consider generating more data for remaining critical issues")
        
    else:
        print("❌ No synthetic data was generated.")
        print("   Check API key and network connection if using LLM generation.")


if __name__ == "__main__":
    main()