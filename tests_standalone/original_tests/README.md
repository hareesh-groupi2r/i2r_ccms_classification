# CCMS Classification System - Test Suite

This directory contains comprehensive tests for the Contract Correspondence Multi-Category Classification System (CCMS).

## Test Organization

### 📁 **Unit Tests** (Root Level)
- `test_batch_processor.py` - Tests for batch PDF processing functionality
- `test_classifiers.py` - Tests for classification algorithms (Hybrid RAG, Pure LLM)
- `test_correspondence_extraction.py` - Tests for subject/body extraction from PDFs
- `test_pdf_extraction.py` - Tests for PDF text extraction methods

### 📁 **Integration Tests** (`integration/`)
- `test_augmentation.py` - Tests for data augmentation functionality
- `test_batch_simple.py` - Simple batch processing integration tests
- `test_category_normalizer.py` - Tests for category normalization logic
- `test_classifiers.py` - Integration tests for classifier components
- `test_complete_hierarchical_system.py` - End-to-end system tests
- `test_correspondence.py` - Integration tests for correspondence extraction
- `test_gemini_simple.py` - Tests for Gemini LLM integration
- `test_hierarchical_llm.py` - Tests for hierarchical LLM approach
- `test_issue_normalizer.py` - Tests for issue type normalization
- `test_llm_call.py` - Tests for LLM API calls
- `test_pdf_extraction.py` - Integration tests for PDF processing
- `test_production_api.py` - Tests for production API endpoints
- `test_provider_tracking.py` - Tests for LLM provider tracking
- `test_single_pdf.py` - Tests for single PDF processing
- `test_subject_patterns.py` - Tests for subject pattern matching

### 📁 **Evaluation Tests** (`evaluation/`)
- `evaluate_lot11.py` - Evaluation script for LOT-11 dataset
- `examine_lot11_data.py` - Data analysis script for LOT-11
- `quick_api_test.py` - Quick API functionality tests
- `quick_pdf_check.py` - Quick PDF processing verification
- `quick_synthetic_generator.py` - Synthetic data generation for testing
- `test_lot11_evaluation.py` - Formal tests for LOT-11 evaluation

### 📁 **Debug Scripts** (`debug_scripts/`)
- `debug_full_pipeline.py` - Debug the complete classification pipeline
- `debug_llm_response.py` - Debug LLM response parsing issues
- `debug_patterns.py` - Debug correspondence pattern matching
- `debug_production_flow.py` - Debug production workflow
- `debug_pure_llm.py` - Debug Pure LLM classifier

## Running Tests

### **Run All Tests**
```bash
python run_tests.py
```

### **Run Specific Test Categories**
```bash
# Unit tests only (default)
python run_tests.py --fast

# Integration tests (requires data files)
python run_tests.py --integration

# Run specific test class
python run_tests.py --test TestBatchProcessor

# Run specific test method
python run_tests.py --test TestBatchProcessor.test_process_single_pdf
```

### **Run with Coverage**
```bash
python run_tests.py --coverage
```

### **List Available Tests**
```bash
python run_tests.py --list
```

## Test Data Requirements

Some integration tests require:
- **LOT-11 Data**: `data/Lot-11/` directory with PDF files
- **Ground Truth Files**: Excel files with labeled data
- **Configuration Files**: `config.yaml` and `.env` file
- **Trained Models**: Embedding index in `data/embeddings/`

## Key Test Components

### **Batch Processing Tests**
- ✅ Single PDF processing with ground truth
- ✅ Batch processing with metrics calculation
- ✅ Excel output formatting and validation
- ✅ Ground truth auto-detection
- ✅ Confidence filtering (≥ 0.5)
- ✅ "Others" category exclusion from metrics

### **Classification Tests**
- ✅ Hybrid RAG approach with semantic search
- ✅ Pure LLM approach with various providers
- ✅ Category normalization and validation
- ✅ Issue type to category mapping
- ✅ Confidence score calculation

### **Content Extraction Tests**
- ✅ PDF text extraction (OCR and direct)
- ✅ Subject and body identification
- ✅ Correspondence pattern matching
- ✅ Content cleaning and normalization

### **Metrics and Evaluation Tests**
- ✅ Precision, Recall, F1-score calculation
- ✅ Multi-label classification metrics
- ✅ Ground truth comparison
- ✅ Excel results validation

## Contributing New Tests

When adding new tests:

1. **Unit Tests**: Add to root level for core functionality
2. **Integration Tests**: Add to `integration/` for component interaction
3. **Evaluation Tests**: Add to `evaluation/` for performance analysis
4. **Debug Scripts**: Add to `debug_scripts/` for troubleshooting

Follow naming convention: `test_*.py` for test files, `TestClassName` for test classes.

## Test Coverage Goals

- **Core Components**: >90% coverage
- **Integration Paths**: All major workflows covered
- **Error Handling**: Exception paths tested
- **Data Validation**: Input/output validation covered

Run `python run_tests.py --coverage` to generate coverage reports.