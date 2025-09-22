# Temporary Test Scripts

This directory contains temporary test scripts, demo utilities, and batch processing scripts.

## 🧪 Test Scripts
- `check_issue_types.py` - Check for issue type variations that need normalization
- `lot21_batch_test.py` - LOT-21 batch processing test with Priority 1-3 fixes
- `run_tests.py` - Test runner for CCMS classification system

## 🎯 Demo Scripts  
- `demonstrate_complete_normalization.py` - Complete normalization system demo
- `demonstrate_normalization.py` - Basic normalization demo
- `docker_demo.py` - Docker deployment capabilities demo

## 🔧 Utility Scripts
- `export_issue_category_mapping.py` - Export issue-to-category mappings to Excel
- `export_mapping_simple.py` - Simple mapping export utility
- `start_production.py` - Production startup script with health checks

## 📦 Batch Processing Scripts
- `process_lots21_27.sh` - Batch processing for Lots 21-27
- `process_single_lot.sh` - Single lot processing script  
- `process_specific_files.sh` - Specific file processing script

## 📝 Usage Note

These scripts are temporary utilities and can be:
1. **Integrated** into the main test suite if useful
2. **Documented** for future reference
3. **Removed** if no longer needed

Most functionality has been consolidated into:
- `backend/` for production code
- `tests_standalone/` for comprehensive testing
- `docs/` for documentation