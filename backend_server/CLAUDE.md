# CCMS Classification Integrated Backend - Claude Context

## Project Overview
The CCMS (Contract Correspondence Management System) Classification Integrated Backend is a Flask-based API server that provides document classification services for legal contract correspondence. This backend replaces the original batch processing approach with a unified API-driven system.

## Architecture
- **Primary Server**: Flask application running on port 5001
- **Text Extraction**: Direct PDFExtractor using pdfplumber → OCR fallback per page
- **Classification**: Hybrid RAG (Retrieval-Augmented Generation) using FAISS vector index
- **Quality Control**: Conditional document quality filtering for problematic extractions
- **API Endpoints**: RESTful services for classification, OCR, and document processing

## Key Components

### 1. PDF Text Extraction (`classifier/pdf_extractor.py`)
- **Primary Method**: Hybrid extraction using pdfplumber for text + OCR for scanned pages
- **Page Limit**: Configurable max_pages (default: 2 for performance)
- **OCR Fallback**: Per-page OCR using Tesseract when text extraction insufficient
- **Resource Management**: Explicit image cleanup to prevent memory leaks
- **Methods**: `extract_text()` returns `(text, method)` tuple

### 2. Classification Service (`integrated_backend/api/service_endpoints.py`)
- **Direct Integration**: Uses PDFExtractor directly (not OCRService wrapper)
- **Quality Filtering**: Conditional filtering only for poor extractions (<200 chars or "scanned by")
- **Include Patterns**: Supports file filtering via fnmatch patterns
- **FAISS Integration**: Vector similarity search for document classification

### 3. Document Quality Checker (`classifier/document_quality.py`)
- **OCR Artifact Detection**: Patterns for "scanned by cam scanner" and similar artifacts
- **Content Validation**: Meaningful text thresholds and character analysis
- **Conditional Application**: Only applied when extraction results are poor

## API Endpoints

### Health & Status
- `GET /api/services/health` - Overall system health
- `GET /api/services/hybrid-rag-classification/status` - Classification service status

### Classification Services
- `POST /api/services/hybrid-rag-classification/classify-text` - Classify text content
- `POST /api/services/hybrid-rag-classification/classify-batch` - Batch file processing
- `POST /api/services/hybrid-rag-classification/process-folder` - Folder processing with enhanced logging
- `GET /api/services/hybrid-rag-classification/categories` - Available categories
- `GET /api/services/hybrid-rag-classification/issues` - Available issue types

### Document Services
- `POST /api/services/ocr/extract-text` - Extract text from documents
- `GET /api/services/ocr/methods` - Available OCR methods
- `POST /api/services/document-type/classify` - Document type classification

### LLM Services
- `POST /api/services/llm/extract-structured` - Structured data extraction
- `POST /api/services/llm/generate-text` - Text generation
- `POST /api/services/llm/summarize` - Document summarization

## Environment Configuration

### Required API Keys
- `CLAUDE_API_KEY` - Anthropic Claude API
- `OPENAI_API_KEY` - OpenAI GPT models
- `GOOGLE_API_KEY` - Google Gemini API

### Flask Settings
- `FLASK_HOST=0.0.0.0` - Listen on all interfaces
- `FLASK_PORT=5001` - Default port
- `FLASK_DEBUG=true` - Development mode
- `FLASK_USE_RELOADER=false` - Prevent dual processes

### Warning Suppression
- `TOKENIZERS_PARALLELISM=false` - Prevent HuggingFace warnings
- `PYTHONWARNINGS=ignore::UserWarning:multiprocessing.resource_tracker` - Suppress resource tracker warnings

## Startup and Management

### Start Server
```bash
# Auto-start with cleanup
./start_integrated_backend.sh --start

# Interactive mode
./start_integrated_backend.sh

# Check status only
./start_integrated_backend.sh --status
```

### Server Management
- **PID File**: `integrated_backend/server.pid`
- **Log File**: `integrated_backend/server.log`
- **Process Cleanup**: Automatic detection and cleanup of existing servers
- **Port Management**: Automatic port conflict resolution

## Key Fixes and Improvements

### Text Extraction Consistency
- **Issue**: Different extraction approaches between batch processor and integrated backend
- **Solution**: Replaced OCRService wrapper with direct PDFExtractor usage
- **Result**: Identical text extraction (2,904 chars for "11. 175_EPC-10 Letter.pdf")

### Process Management
- **Issue**: Dual Flask processes due to reloader
- **Solution**: Set `FLASK_USE_RELOADER=false` in environment
- **Result**: Single clean process with proper PID tracking

### Quality Filtering
- **Issue**: Aggressive quality filtering blocking valid documents
- **Solution**: Conditional filtering only for poor extractions (<200 chars or OCR artifacts)
- **Result**: Performance optimization while maintaining quality control

### Resource Management
- **Issue**: Memory leaks and resource tracker warnings
- **Solution**: Explicit image cleanup and environment variable configuration
- **Result**: Clean resource usage and suppressed warnings

## Testing and Validation

### Single File Testing
```bash
# Test specific problematic document
python -c "
import requests
response = requests.post('http://localhost:5001/api/services/hybrid-rag-classification/process-folder',
    json={'folder_path': '/path/to/folder', 'include_patterns': ['11. 175_EPC-10*']})
print(response.json())
"
```

### Health Check
```bash
curl http://localhost:5001/api/services/health
```

### Status Monitoring
```bash
curl http://localhost:5001/api/services/hybrid-rag-classification/status
```

## Performance Characteristics

### Text Extraction
- **Method**: `ocr_only(2pages)` for scanned documents
- **Processing Time**: ~8 seconds for 2-page documents
- **Memory Usage**: Optimized with explicit image cleanup
- **Character Extraction**: 2,904 characters for test document

### Classification
- **Quality Check**: Conditional filtering for poor extractions only
- **Vector Search**: FAISS-based similarity matching
- **Approaches**: Hybrid RAG with fallback options

## Common Commands

### Lint and Type Check
```bash
# Check if these commands exist in the project
npm run lint      # If Node.js project
npm run typecheck # If TypeScript
python -m flake8  # Python linting
python -m mypy    # Python type checking
```

### Testing
```bash
# Run integrated backend tests
python test_integrated_backend.py

# Run regression suite
python test_regression_suite.py
```

## Troubleshooting

### Common Issues
1. **Port 5001 in use**: Use `--start` flag to auto-cleanup
2. **API key errors**: Check `.env` file configuration
3. **Text extraction fails**: Verify Tesseract installation
4. **Memory issues**: Check image cleanup in PDFExtractor

### Debug Commands
```bash
# Check server logs
tail -f integrated_backend/server.log

# Check process status
./start_integrated_backend.sh --status

# Test single endpoint
curl -X POST http://localhost:5001/api/services/ocr/extract-text \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/test.pdf"}'
```

## Enhanced Logging Options

The process-folder endpoint now supports detailed per-PDF logging for step-by-step analysis:

### Logging Parameters
- `debug_logging: boolean` - Enable DEBUG level logging for all classifiers
- `per_file_logging: boolean` - Show detailed per-file processing summaries
- `log_text_extraction: boolean` - Log detailed text extraction process
- `log_classification_details: boolean` - Show detailed classification results

### Example Usage
```bash
curl -X POST http://localhost:5001/api/services/hybrid-rag-classification/process-folder \
  -H "Content-Type: application/json" \
  -d '{
    "folder_path": "/path/to/lot21", 
    "output_folder": "/tmp/lot21_test",
    "per_file_logging": true,
    "log_text_extraction": true,
    "log_classification_details": true,
    "include_patterns": ["*.pdf"]
  }'
```

### Logging Levels
1. **Standard** (default): Basic processing status per file
2. **Per-file** (`per_file_logging: true`): File-by-file summaries with processing times
3. **Text extraction** (`log_text_extraction: true`): Detailed text extraction process
4. **Classification details** (`log_classification_details: true`): Full classification process with category filtering

## Development Notes

### Code Patterns
- Always use direct PDFExtractor instead of OCRService wrapper
- Apply quality filtering conditionally based on extraction results
- Include fnmatch patterns for file filtering in batch operations
- Handle resource cleanup explicitly for images and processes
- Use enhanced logging options for debugging complex classification issues

### Testing Strategy
- Test with problematic documents (OCR artifacts, scanned content)
- Verify consistency with original batch processor results
- Monitor resource usage and memory leaks
- Validate all API endpoints and error handling
- Use detailed logging to trace processing steps for individual files

## Recent Changes
- Replaced OCRService with direct PDFExtractor (2025-09-16)
- Added conditional quality filtering (2025-09-16)
- Fixed Flask dual-process issue (2025-09-16)
- Enhanced startup script with comprehensive status reporting (2025-09-16)
- Added resource leak fixes and warning suppression (2025-09-16)