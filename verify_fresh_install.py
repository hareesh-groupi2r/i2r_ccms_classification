#!/usr/bin/env python3
"""
Verify Fresh Install Capabilities for CCMS Classification System
Tests that all required files exist and system can initialize properly
"""

import sys
import os
from pathlib import Path
import pandas as pd

def verify_fresh_install():
    """Verify all components for fresh install"""
    
    print("🔍 CCMS Classification System - Fresh Install Verification")
    print("=" * 65)
    
    issues = []
    warnings = []
    
    # 1. Check required training data files
    print("\n📊 Checking Training Data Files...")
    
    training_files = [
        "data/synthetic/combined_training_data.xlsx",
        "data/raw/Consolidated_labeled_data.xlsx"
    ]
    
    for file_path in training_files:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size
            print(f"  ✅ {file_path} ({size:,} bytes)")
            
            # Quick sample count check
            try:
                df = pd.read_excel(file_path)
                print(f"     📋 Contains {len(df)} training samples")
                print(f"     📋 Issue types: {df['issue_type'].nunique()}")
            except Exception as e:
                warnings.append(f"Could not read {file_path}: {e}")
        else:
            issues.append(f"Missing training data: {file_path}")
    
    # 2. Check unified mapping file
    print("\n🗂️  Checking Unified Mapping File...")
    
    mapping_file = "issue_category_mapping_diffs/unified_issue_category_mapping.xlsx"
    if Path(mapping_file).exists():
        size = Path(mapping_file).stat().st_size
        print(f"  ✅ {mapping_file} ({size:,} bytes)")
        
        try:
            # Check mapping content
            sys.path.insert(0, str(Path(__file__).parent))
            from classifier.unified_issue_mapper import UnifiedIssueCategoryMapper
            
            mapper = UnifiedIssueCategoryMapper(
                training_files[0] if Path(training_files[0]).exists() else training_files[1],
                mapping_file
            )
            all_issues = mapper.get_all_issue_types()
            all_categories = mapper.get_all_categories()
            
            print(f"     📋 Issue types: {len(all_issues)}")
            print(f"     📋 Categories: {len(all_categories)}")
            
        except Exception as e:
            warnings.append(f"Could not read unified mapping: {e}")
    else:
        issues.append(f"Missing unified mapping file: {mapping_file}")
    
    # 3. Check classifier modules
    print("\n🔧 Checking Classifier Modules...")
    
    classifier_modules = [
        "classifier/hybrid_rag.py",
        "classifier/validation.py",
        "classifier/unified_issue_mapper.py",
        "classifier/data_sufficiency.py",
        "classifier/config_manager.py"
    ]
    
    for module in classifier_modules:
        if Path(module).exists():
            print(f"  ✅ {module}")
        else:
            issues.append(f"Missing classifier module: {module}")
    
    # 4. Check integrated backend
    print("\n🏗️  Checking Integrated Backend...")
    
    backend_files = [
        "integrated_backend/services/hybrid_rag_classification_service.py",
        "integrated_backend/api/service_endpoints.py",
        "integrated_backend/api/app.py",
        "start_integrated_backend.sh"
    ]
    
    for file_path in backend_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            issues.append(f"Missing backend file: {file_path}")
    
    # 5. Check configuration
    print("\n⚙️  Checking Configuration...")
    
    config_files = [
        "config.yaml",
        "integrated_backend/config.yaml"
    ]
    
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"  ✅ {config_file}")
        else:
            warnings.append(f"Configuration file not found: {config_file}")
    
    # 6. Check vector database capability
    print("\n🗄️  Checking Vector Database Setup...")
    
    embeddings_dir = Path("data/embeddings")
    if embeddings_dir.exists():
        print(f"  ✅ Embeddings directory exists")
        
        # Check existing index files
        faiss_file = embeddings_dir / "rag_index.faiss"
        pkl_file = embeddings_dir / "rag_index.pkl"
        
        if faiss_file.exists() and pkl_file.exists():
            print(f"  📊 Existing vector index found")
            print(f"     FAISS: {faiss_file.stat().st_size:,} bytes")
            print(f"     Metadata: {pkl_file.stat().st_size:,} bytes")
        else:
            print(f"  📊 No existing vector index (will build on startup)")
    else:
        print(f"  📊 Embeddings directory missing (will create on startup)")
    
    # 7. Check enhanced features
    print("\n🚀 Checking Enhanced Features...")
    
    enhanced_files = [
        "claude_synthetic_generator.py",
        "generate_missing_training_data.py",
        "TECHNICAL_ENHANCEMENTS_DOCUMENTATION.md"
    ]
    
    for file_path in enhanced_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            warnings.append(f"Enhanced feature missing: {file_path}")
    
    # 8. Test basic imports
    print("\n🧪 Testing Basic Imports...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from classifier.config_manager import ConfigManager
        from classifier.unified_issue_mapper import UnifiedIssueCategoryMapper
        from classifier.validation import ValidationEngine
        from classifier.data_sufficiency import DataSufficiencyAnalyzer
        print("  ✅ Core classifier modules import successfully")
    except Exception as e:
        issues.append(f"Import error: {e}")
    
    # Summary
    print("\n" + "=" * 65)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 65)
    
    if not issues and not warnings:
        print("🎉 ALL CHECKS PASSED - System ready for fresh install!")
        print("\n🚀 Next Steps for New Team Members:")
        print("   1. pip install -r requirements.txt")
        print("   2. Set API keys in environment (optional for synthetic generation)")
        print("   3. ./start_integrated_backend.sh --start")
        print("   4. System will auto-build vector index on first startup")
        return True
        
    elif not issues:
        print("✅ SYSTEM READY with minor warnings")
        print(f"\n⚠️  Warnings ({len(warnings)}):")
        for warning in warnings:
            print(f"   - {warning}")
        print("\n✅ System will work correctly despite warnings")
        return True
        
    else:
        print("❌ CRITICAL ISSUES FOUND")
        print(f"\n🚨 Issues ({len(issues)}):")
        for issue in issues:
            print(f"   - {issue}")
        
        if warnings:
            print(f"\n⚠️  Warnings ({len(warnings)}):")
            for warning in warnings:
                print(f"   - {warning}")
        
        print("\n🔧 Please resolve critical issues before deployment")
        return False


if __name__ == "__main__":
    success = verify_fresh_install()
    sys.exit(0 if success else 1)