# CCMS Document Processing Services API

This API provides modular document processing services that can be invoked individually or through an orchestrator for complete document analysis.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables
```bash
export GOOGLE_API_KEY="your_gemini_api_key"
export DOCAI_PROJECT_ID="your_docai_project_id"
export DOCAI_PROCESSOR_ID="your_docai_processor_id"
```

### 3. Start the API Server
```bash
python app.py
```

The API will be available at `http://localhost:5000`

## API Overview

### Base URL: `/api/services`

### Available Services

1. **Document Type Classification** - Classify documents as contracts or correspondence
2. **OCR Service** - Extract text using multiple OCR backends (Document AI, PyPDF, Tesseract)
3. **LLM Service** - Extract structured data and classify content using Gemini
4. **Category Mapping** - Map issues to predefined categories with fuzzy matching
5. **Document Processing Orchestrator** - Coordinate all services for end-to-end processing

## API Endpoints

### Document Type Classification

#### Classify Document from File
```http
POST /api/services/document-type/classify
Content-Type: multipart/form-data

file: [PDF file]
pages_to_check: 5 (optional)
use_advanced_classification: true (optional)
confidence_threshold: 0.6 (optional)
```

#### Classify Document from Text
```http
POST /api/services/document-type/classify-text
Content-Type: application/json

{
  "text": "This Agreement is entered into...",
  "use_advanced_classification": true,
  "confidence_threshold": 0.6
}
```

### OCR Service

#### Extract Text from Document
```http
POST /api/services/ocr/extract
Content-Type: multipart/form-data

file: [PDF/Image file]
method: "auto" (optional: auto, docai, pypdf, tesseract)
fallback_on_error: true (optional)
page_range: "1-5" (optional: format "start-end")
```

#### Extract Text from Specific Pages
```http
POST /api/services/ocr/extract-pages
Content-Type: multipart/form-data

file: [PDF file]
page_numbers: "1,3,5" (comma-separated page numbers)
method: "pypdf" (optional)
```

#### Get Available OCR Methods
```http
GET /api/services/ocr/methods
```

### LLM Service

#### Extract Structured Data
```http
POST /api/services/llm/extract-structured
Content-Type: application/json

{
  "text": "Contract text content...",
  "schema": {
    "Project Name": "Official project title",
    "Contract Value": "Total contract amount",
    "Start Date": "Project start date"
  },
  "output_format": "json",
  "additional_instructions": "Focus on financial terms"
}
```

#### Classify Content
```http
POST /api/services/llm/classify-content
Content-Type: application/json

{
  "text": "Content to classify...",
  "options": ["Contract", "Correspondence", "Report"],
  "confidence_threshold": 0.7,
  "return_all_scores": false
}
```

### Category Mapping Service

#### Map Issue to Category
```http
POST /api/services/category-mapping/map-issue
Content-Type: application/json

{
  "issue_type": "Payment Issues",
  "use_fuzzy_matching": true,
  "min_confidence": 0.7
}
```

#### Bulk Classify Issues
```http
POST /api/services/category-mapping/bulk-classify
Content-Type: application/json

{
  "issues": ["Payment delay", "Quality control", "Schedule issues"],
  "use_fuzzy_matching": true,
  "min_confidence": 0.7
}
```

#### Get Available Categories
```http
GET /api/services/category-mapping/categories?include_counts=true&sort_by=name
```

### Document Processing Orchestrator

#### Full Document Processing
```http
POST /api/services/orchestrator/process-document
Content-Type: multipart/form-data

file: [Document file]
extract_structured_data: true (optional)
classify_issues: true (optional)
document_type: "contract" (optional: force document type)
extraction_method: "auto" (optional)
extraction_schema: '{"field": "description"}' (optional JSON string)
```

#### Partial Document Processing
```http
POST /api/services/orchestrator/process-partial
Content-Type: multipart/form-data

file: [Document file]
services: "document_type,ocr,llm" (comma-separated service list)
extraction_schema: '{"field": "description"}' (optional for LLM)
issues: "issue1,issue2" (optional for category_mapping)
```

## Example Usage

### Python Example

```python
import requests
import json

# Full document processing
with open('contract.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/services/orchestrator/process-document',
        files={'file': f},
        data={
            'extract_structured_data': 'true',
            'classify_issues': 'true',
            'extraction_schema': json.dumps({
                'Project Name': 'Official project name',
                'Contract Value': 'Total contract amount',
                'Contractor': 'Contractor name'
            })
        }
    )

result = response.json()
if result['success']:
    processing_results = result['data']
    print(f"Document Type: {processing_results['document_type']}")
    print(f"Structured Data: {processing_results['structured_data']}")
    print(f"Classifications: {processing_results['classifications']}")
else:
    print(f"Error: {result['error']}")
```

### JavaScript Example

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('extract_structured_data', 'true');

fetch('/api/services/orchestrator/process-document', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(result => {
    if (result.success) {
        console.log('Document Type:', result.data.document_type);
        console.log('Extracted Text Length:', result.data.extracted_text.length);
        console.log('Structured Data:', result.data.structured_data);
    } else {
        console.error('Error:', result.error);
    }
});
```

## Response Format

All endpoints return responses in this format:

```json
{
  "status": "success|error|partial",
  "success": true|false,
  "data": {...},
  "confidence": 0.85,
  "metadata": {...},
  "error": "error message (if failed)"
}
```

## Configuration

Services can be configured via environment variables or the configuration API:

### Get All Service Configurations
```http
GET /api/services/config/services
```

### Update Service Configuration
```http
PUT /api/services/config/services/llm
Content-Type: application/json

{
  "temperature": 0.2,
  "max_tokens": 5000,
  "retry_count": 3
}
```

## Health Check

Check the health of all services:

```http
GET /api/services/health
```

Returns service availability and status information.

## Error Handling

The API uses standard HTTP status codes:
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found
- `413`: File Too Large
- `500`: Internal Server Error

Error responses include descriptive error messages to help with debugging.

## File Size Limits

- Maximum file size: 200MB
- Supported formats: PDF, PNG, JPG, JPEG, TIFF

## Security Notes

- API keys should be set via environment variables
- File uploads are temporarily stored and automatically cleaned up
- No authentication is currently implemented (suitable for internal use)