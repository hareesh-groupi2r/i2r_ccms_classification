# Batch PDF Processing System

A configurable batch processing system for contract correspondence classification that supports multiple approaches and optional metrics evaluation.

## Features

✅ **Configurable Approaches**
- Hybrid RAG (Retrieval-Augmented Generation) - Always recommended
- Pure LLM - Can be enabled/disabled based on requirements
- Priority-based approach selection

✅ **Flexible Ground Truth Handling**
- Auto-detection of ground truth files (EDMS*.xlsx, ground_truth*.xlsx, *_labels.xlsx)
- Optional metrics computation with enable/disable flag
- Inference mode support (no ground truth required)

✅ **Scalable Processing**
- Batch processing of PDF folders
- Error handling with skip-on-error option
- Rate limiting to respect API quotas
- Progress tracking and detailed logging

✅ **Comprehensive Output**
- Excel results with multiple sheets (Summary, Metrics, Processing Stats)
- JSON output for programmatic access
- Individual file metrics and overall performance statistics

## Quick Start

### 1. Simple Processing (Recommended)

```python
from batch_processor import process_lot_pdfs

# Process with Hybrid RAG only (fastest, most reliable)
results = process_lot_pdfs(
    pdf_folder="data/Lot-11",
    enable_llm=False,  # Only Hybrid RAG
    enable_metrics=True,  # Auto-detect ground truth
    output_folder="results/batch_results"
)
```

### 2. Both Approaches Comparison

```python
# Process with both approaches for comparison
results = process_lot_pdfs(
    pdf_folder="data/Lot-11",
    ground_truth_file="data/EDMS-Lot 11.xlsx",  # Explicit ground truth
    enable_llm=True,  # Enable Pure LLM
    enable_metrics=True,
    output_folder="results/comparison"
)
```

### 3. Inference Mode (No Ground Truth)

```python
# Pure inference without evaluation
results = process_lot_pdfs(
    pdf_folder="data/new_documents",
    enable_llm=False,  # Only Hybrid RAG for speed
    enable_metrics=False,  # No ground truth available
    output_folder="results/inference"
)
```

## Advanced Configuration

### Custom Batch Processor

```python
from batch_processor import BatchPDFProcessor

# Initialize with custom configuration
processor = BatchPDFProcessor(
    config_path="config.yaml",
    batch_config_path="custom_batch_config.yaml"
)

# Process PDFs
results = processor.process_pdf_folder(
    pdf_folder="data/Lot-11",
    ground_truth_file="data/ground_truth.xlsx",
    output_folder="results/custom"
)
```

### Configuration Files

#### Main Configuration (`config.yaml`)
```yaml
approaches:
  pure_llm:
    enabled: true  # Can be overridden by batch config
    model: "gpt-4-turbo"
    max_tokens: 4096
    temperature: 0.1
    
  hybrid_rag:
    enabled: true  # Recommended to keep enabled
    embedding_model: "all-mpnet-base-v2"
    llm_model: "claude-sonnet-4-20250514"
    vector_db: "faiss"
```

#### Batch Configuration (`batch_config.yaml`)
```yaml
batch_processing:
  enabled: true
  approaches:
    hybrid_rag:
      enabled: true
      priority: 1  # Always run first
    pure_llm:
      enabled: false  # Toggle this based on needs
      priority: 2
      
  evaluation:
    enabled: true  # Enable metrics computation
    auto_detect_ground_truth: true
    ground_truth_patterns:
      - "EDMS*.xlsx"
      - "ground_truth*.xlsx" 
      - "*_labels.xlsx"
    
  processing:
    max_pages_per_pdf: 2
    skip_on_error: true
    rate_limit_delay: 3  # seconds between API calls
    
  output:
    results_folder: "results"
    save_format: "xlsx"
    include_confidence_scores: true
```

## Command Line Usage

```bash
# Basic usage
python batch_processor.py data/Lot-11

# With options
python batch_processor.py data/Lot-11 \
  --ground-truth data/EDMS-Lot-11.xlsx \
  --enable-llm \
  --output results/batch_run

# Inference mode
python batch_processor.py data/new_documents \
  --disable-metrics \
  --output results/inference
```

## Output Structure

### Excel Output
The system generates Excel files with multiple sheets:

1. **Summary Sheet** - File-by-file results with categories and metrics
2. **Overall Metrics Sheet** - Aggregated performance metrics by approach
3. **Processing Stats Sheet** - Processing time and success/failure statistics

### JSON Output
```json
{
  "processing_stats": {
    "total_files": 25,
    "processed_files": 24,
    "failed_files": 1,
    "start_time": "2024-01-15T10:30:00",
    "end_time": "2024-01-15T11:15:00"
  },
  "config": {
    "enabled_approaches": ["hybrid_rag"]
  },
  "results": [
    {
      "file_name": "document.pdf",
      "status": "completed",
      "approaches": {
        "hybrid_rag": {
          "categories": ["Change of Scope", "Payments"],
          "processing_time": 5.2,
          "provider_used": "RAG-based",
          "metrics": {
            "precision": 0.8,
            "recall": 1.0,
            "f1_score": 0.89
          }
        }
      },
      "ground_truth": ["Change of Scope", "Payments"]
    }
  ],
  "overall_metrics": {
    "hybrid_rag": {
      "micro_f1": 0.85,
      "macro_f1": 0.83,
      "exact_match_accuracy": 0.72
    }
  }
}
```

## Performance Recommendations

### For Production Use
- **Enable**: Hybrid RAG only (`enable_llm=False`)
- **Reason**: Faster, more reliable, lower API costs
- **Rate Limiting**: 3 seconds between files (default)

### For Research/Comparison
- **Enable**: Both approaches (`enable_llm=True`)
- **Reason**: Compare performance, validate results
- **Rate Limiting**: Consider longer delays due to LLM API calls

### For Inference/New Documents
- **Enable**: Hybrid RAG only (`enable_llm=False`)
- **Metrics**: Disabled (`enable_metrics=False`)
- **Reason**: No ground truth available, fastest processing

## Error Handling

The system includes robust error handling:
- **Skip on Error**: Continue processing other files if one fails
- **Detailed Logging**: Track processing status and errors
- **Graceful Degradation**: Fall back to available approaches
- **Recovery**: Retry failed files with different methods

## Metrics Explained

### Individual File Metrics
- **Precision**: Correct predictions / Total predictions
- **Recall**: Correct predictions / Total ground truth
- **F1-Score**: Harmonic mean of precision and recall
- **Exact Match**: Whether all categories match exactly

### Aggregated Metrics
- **Micro F1**: Aggregate TP/FP/FN across all files
- **Macro F1**: Average of individual file F1 scores
- **Exact Match Accuracy**: Percentage of perfect predictions

## Troubleshooting

### Common Issues

1. **No Ground Truth Detected**
   ```
   Solution: Ensure EDMS*.xlsx file exists in PDF folder or parent directories
   ```

2. **API Key Errors**
   ```
   Solution: Check .env file has ANTHROPIC_API_KEY or OPENAI_API_KEY set
   ```

3. **Memory Issues with Large Batches**
   ```
   Solution: Process in smaller batches or increase system memory
   ```

4. **Rate Limiting Errors**
   ```
   Solution: Increase rate_limit_delay in batch configuration
   ```

## Examples

See `example_batch_usage.py` for detailed usage examples including:
- Simple lot processing
- Approach comparison
- Advanced configuration
- Error handling
- Inference mode

## Integration

The batch processing system can be integrated into larger workflows:

```python
# In your workflow
from batch_processor import BatchPDFProcessor

def process_contract_lot(lot_name: str, pdf_folder: str):
    processor = BatchPDFProcessor()
    results = processor.process_pdf_folder(
        pdf_folder=pdf_folder,
        output_folder=f"results/{lot_name}"
    )
    return results['overall_metrics']
```