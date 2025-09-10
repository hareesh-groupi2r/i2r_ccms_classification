# Batch Processing Guide for Lots 21-27

This guide provides comprehensive instructions for processing the contract correspondence lots 21-27 using the CCMS classification system.

## Overview

The system provides two main scripts for batch processing:

1. **`process_lots21_27.sh`** - Process all lots (21-27) at once
2. **`process_single_lot.sh`** - Process individual lots with custom options

Both scripts automatically handle:
- PDF extraction and text processing
- Hybrid RAG classification (default)
- Optional Pure LLM classification
- Ground truth evaluation (where available)
- Metrics calculation and reporting
- Results export to Excel format

## Quick Start

### Process All Lots

```bash
# Process all lots with default settings (Hybrid RAG only)
./process_lots21_27.sh

# Process all lots with Pure LLM enabled (slower but comprehensive)
./process_lots21_27.sh --with-llm

# See what would be processed without running
./process_lots21_27.sh --dry-run
```

### Process Individual Lots

```bash
# Process LOT-21 (has ground truth)
./process_single_lot.sh 21

# Process LOT-24 with Pure LLM enabled
./process_single_lot.sh 24 --with-llm

# Process LOT-22 without metrics calculation
./process_single_lot.sh 22 --no-metrics

# Process with custom output directory
./process_single_lot.sh 25 --output my_results/lot25
```

## Lot Structure

The system processes the following lots from `data/Lots21-27/`:

### Lots 21-23 (Located in `Lot 21 to 23/`)
- **LOT-21**: 27 PDFs with ground truth (`LOT-21.xlsx`)
- **LOT-22**: 20 PDFs (no ground truth)
- **LOT-23**: 18 PDFs (no ground truth)

### Lots 24-27 (Located in `Lot 24 to 27/`)
- **LOT-24**: 30 PDFs (no ground truth)
- **LOT-25**: 30 PDFs (no ground truth)
- **LOT-26**: 30 PDFs (no ground truth)
- **LOT-27**: 30 PDFs (no ground truth)

**Total**: 185 PDF files across 7 lots

## Ground Truth and Evaluation

### Available Ground Truth
- **LOT-21**: `LOT-21.xlsx` contains manual classifications for evaluation
- **LOT-22 to LOT-27**: No ground truth files found

### Metrics Calculation
When ground truth is available or auto-detected, the system calculates:
- **Precision, Recall, F1-Score**: Classification accuracy metrics
- **Exact Match**: Percentage of perfectly classified documents
- **Category Statistics**: Per-category performance breakdown
- **Jaccard Similarity**: Set-based similarity measure

### Auto-Detection
The system automatically looks for ground truth files matching these patterns:
- `EDMS*.xlsx`
- `ground_truth*.xlsx`
- `*_labels.xlsx`

## Classification Approaches

### Hybrid RAG (Default)
- **Method**: Combines retrieval-augmented generation with similarity search
- **Speed**: Fast (~2-3 seconds per PDF)
- **Accuracy**: High for documents similar to training data
- **Use Case**: Production environments, large batches

### Pure LLM (Optional)
- **Method**: Direct LLM classification with hierarchical fallback (Gemini → OpenAI → Anthropic)
- **Speed**: Slower (~15-30 seconds per PDF)
- **Accuracy**: Often higher for complex or unusual documents
- **Use Case**: Maximum accuracy, smaller batches, research

## Output Structure

Results are saved to `results/` directory with the following structure:

```
results/
├── LOT-21/
│   ├── batch_results_20250910_143022.xlsx    # Main results file
│   ├── processing_summary.json               # Processing metadata
│   └── metrics_report.json                   # Detailed metrics (if ground truth available)
├── LOT-22/
│   └── batch_results_20250910_143145.xlsx
└── ... (other lots)
```

### Excel Results File Structure

Each results file contains multiple sheets:

1. **Results**: Main classification results
   - File name and path
   - Extracted categories
   - Confidence scores
   - Processing method and time
   - Ground truth (if available)
   - Metrics (if calculated)

2. **Metrics Summary** (if ground truth available):
   - Overall performance metrics
   - Per-category statistics
   - Confusion matrix data

3. **Processing Stats**:
   - Total files processed
   - Success/failure rates
   - Processing times
   - System information

## Advanced Usage

### Custom Configuration

Create a custom `batch_config.yaml` to override default settings:

```yaml
batch_processing:
  enabled: true
  approaches:
    hybrid_rag:
      enabled: true
      priority: 1
    pure_llm:
      enabled: false
      priority: 2
  evaluation:
    enabled: true
    auto_detect_ground_truth: true
    ground_truth_patterns: 
      - "EDMS*.xlsx"
      - "ground_truth*.xlsx"
  output:
    results_folder: "results"
    save_format: "xlsx"
  processing:
    max_pages_per_pdf: 2
    skip_on_error: true
    rate_limit_delay: 3
```

### Environment Variables

Set these variables for customization:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GEMINI_API_KEY="your-gemini-key"
```

### Processing Options

| Option | Description | Default |
|--------|-------------|---------|
| `--with-llm` | Enable Pure LLM approach | Disabled |
| `--no-metrics` | Disable metrics calculation | Enabled |
| `--output DIR` | Custom output directory | `results/LOT-XX` |

## Performance Estimates

### Hybrid RAG Processing (Default)
- **LOT-21** (27 PDFs): ~2-3 minutes
- **LOT-22** (20 PDFs): ~1-2 minutes  
- **LOT-23** (18 PDFs): ~1-2 minutes
- **LOT-24-27** (30 PDFs each): ~2-3 minutes each

**Total estimated time**: ~15-20 minutes for all lots

### Pure LLM Processing (--with-llm)
- **LOT-21** (27 PDFs): ~15-20 minutes
- **LOT-22** (20 PDFs): ~10-15 minutes
- **LOT-23** (18 PDFs): ~8-12 minutes
- **LOT-24-27** (30 PDFs each): ~15-25 minutes each

**Total estimated time**: ~2-3 hours for all lots

## Error Handling

The scripts include comprehensive error handling:

### Skip on Error (Default)
- Failed PDFs are logged and skipped
- Processing continues with remaining files
- Failed files are reported in final summary

### Common Issues and Solutions

1. **"Virtual environment not found"**
   ```bash
   # Ensure virtual environment is created
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **"No PDF files found"**
   - Check directory paths in the script
   - Verify PDF files exist in lot directories

3. **"API key not found"**
   ```bash
   # Set API keys in .env file
   echo "OPENAI_API_KEY=your-key" >> .env
   echo "ANTHROPIC_API_KEY=your-key" >> .env
   echo "GEMINI_API_KEY=your-key" >> .env
   ```

4. **Memory errors with large PDFs**
   - Reduce `max_pages_per_pdf` in configuration
   - Process lots individually instead of all at once

## Monitoring Progress

### Real-time Monitoring
```bash
# Watch processing logs
tail -f logs/batch_processing.log

# Monitor results directory
watch -n 5 'find results/ -name "*.xlsx" | wc -l'
```

### Progress Indicators
The scripts provide detailed progress information:
- Current lot being processed
- PDF files found per lot
- Processing time per file
- Success/failure counts
- Overall completion status

## Quality Assurance

### Validation Checks
- **PDF Extraction**: Verifies text extraction success
- **Classification Results**: Validates category format
- **Metrics Calculation**: Ensures ground truth alignment
- **Output Files**: Confirms Excel file creation

### Result Review
After processing, review:
1. **Processing Summary**: Check success rates
2. **Sample Results**: Spot-check classifications
3. **Metrics Report**: Evaluate performance (where available)
4. **Error Logs**: Investigate any failures

## Troubleshooting

### Debug Mode
Enable detailed logging:
```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export LOG_LEVEL=DEBUG
./process_single_lot.sh 21
```

### Manual Processing
If scripts fail, process manually:
```python
from batch_processor import process_lot_pdfs

results = process_lot_pdfs(
    pdf_folder="data/Lots21-27/Lot 21 to 23/LOT-21",
    ground_truth_file="data/Lots21-27/Lot 21 to 23/LOT-21/LOT-21.xlsx",
    enable_llm=False,
    enable_metrics=True,
    output_folder="results/LOT-21"
)
```

### Performance Optimization
For faster processing:
1. Use Hybrid RAG only (default)
2. Process lots in parallel on different machines
3. Reduce `max_pages_per_pdf` for very long documents
4. Increase `rate_limit_delay` if hitting API limits

## Integration with Service Management

Use the service management system for production processing:

```bash
# Start the service
./service.sh start

# Process via API
curl -X POST http://localhost:8000/classify/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["document content..."], "approach": "hybrid_rag"}'

# Monitor service
./service.sh status
./service.sh logs
```

## Best Practices

1. **Start Small**: Test with single lots before processing all
2. **Monitor Resources**: Watch CPU/memory usage during processing
3. **Backup Results**: Copy results before reprocessing
4. **Review Quality**: Manually review sample results
5. **Document Issues**: Log any problems for future reference

## Support

For issues or questions:
1. Check this documentation
2. Review error logs in `logs/` directory
3. Test with single PDFs first
4. Verify configuration files
5. Check API key availability and limits