# Hybrid RAG Flow - Complete Technical Deep Dive
## Contract Correspondence Classification System

---

## Slide 1: Overview - Hybrid RAG Architecture

### **Pipeline Flow Summary**
```
PDF Document → Text Extraction → Correspondence Extraction → 
Text Preprocessing → Chunking → Vector Search → 
Issue Aggregation → LLM Validation → Category Mapping → Final Results
```

### **Key Components**
- **Vector Database**: FAISS with sentence embeddings
- **Embedding Model**: `all-mpnet-base-v2` (768 dimensions)
- **LLM Validation**: Multi-provider fallback (Claude/GPT-4/Gemini)
- **Training Data**: 1000+ labeled correspondence examples

### **Core Advantage**
- Combines **retrieval precision** with **generative intelligence**
- Hierarchical fallback for reliability
- Configurable thresholds for precision/recall tuning

---

## Slide 2: Phase 1 - Document Processing & Text Extraction

### **PDF Text Extraction**
```python
# PDFExtractor with multiple strategies
extraction_methods = [
    "pypdf2",           # Primary: Fast, structure-preserving
    "pdfplumber",       # Fallback: Better for complex layouts  
    "pymupdf",          # Fallback: OCR-like capabilities
    "ocr_fallback"      # Final: For scanned documents
]
```

### **Correspondence Content Extraction**
```python
# CorrespondenceExtractor - Key step for focus
extraction_result = {
    'subject': extracted_subject,
    'body': extracted_body,
    'sender_info': parsed_sender,
    'extraction_method': method_used
}

# Create focused content for classification
focused_content = f"Subject: {subject}\n\nContent: {body}"
```

### **Configuration Points**
- **Max Pages**: `2` (config: `max_pages_per_pdf`)
- **Text Length**: No hard limit, but processed in chunks
- **Fallback Strategy**: Auto-detection across extraction methods

---

## Slide 3: Phase 2 - Text Preprocessing & Normalization

### **Text Cleaning Pipeline**
```python
class TextPreprocessor:
    def normalize_document(self, text):
        # 1. Clean special characters
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\'\"]', ' ', text)
        
        # 2. Fix OCR errors
        text = self.fix_ocr_errors(text)
        
        # 3. Normalize dates/amounts
        text = re.sub(r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b', r'\1/\2/\3', text)
        
        # 4. Preserve contract terms
        preserved_terms = {'shall', 'must', 'agreement', 'payment', ...}
```

### **Key Preprocessing Features**
- **Contract Term Preservation**: Important legal terms NOT removed as stopwords
- **OCR Error Correction**: `l→I`, `O→0` in numeric contexts
- **Sentence Boundary Detection**: NLTK punkt tokenizer
- **Lemmatization**: WordNet lemmatizer (not stemming)

### **Thresholds & Configuration**
- **Min Sentence Length**: 20 characters
- **Max Evidence Length**: 200 characters (truncated with `...`)

---

## Slide 4: Phase 3 - Chunking Strategy (Critical for Context)

### **Smart Document Chunking**
```python
def create_sliding_windows(text, window_size=3, overlap=1):
    sentences = extract_sentences(text)
    
    # Smart sizing for different document lengths
    if text_length <= window_size * 100:
        return [(entire_text, 0, text_length)]  # Single chunk for small docs
    else:
        # Sliding windows with overlap for context
        stride = window_size - overlap  # stride = 2 for window=3, overlap=1
```

### **Chunking Configuration**
- **Window Size**: `3 sentences` (config: `window_size`)
- **Overlap**: `1 sentence` (config: `overlap`)
- **Stride**: `2 sentences` (calculated: window_size - overlap)
- **Chunk Size**: ~300 characters per sentence × 3 = ~900 chars per chunk

### **Chunking Logic**
- **Small Documents** (< 300 chars): Single chunk (entire document)
- **Large Documents**: Multiple overlapping chunks
- **Overlap Purpose**: Ensures issue context not lost at chunk boundaries

---

## Slide 5: Phase 4 - Vector Embedding & Index Search

### **Embedding Process**
```python
# Sentence Transformer Model: all-mpnet-base-v2
embedding_model = SentenceTransformer('all-mpnet-base-v2')
embedding_dim = 768  # Fixed dimension

# Encode document chunks
query_embeddings = model.encode(chunk_texts)  # Shape: [n_chunks, 768]
```

### **FAISS Vector Search**
```python
# Search configuration
search_params = {
    'top_k': 12,                    # Retrieve top 12 similar examples
    'similarity_threshold': 0.20,   # Very low threshold for max recall
    'distance_metric': 'L2',        # Euclidean distance in FAISS
    'index_type': 'IndexFlatL2'     # Exhaustive search for accuracy
}

# Convert L2 distance to similarity score
similarity = 1 / (1 + l2_distance)
```

### **Critical Thresholds**
- **Top-K**: `12` results per chunk (config: `top_k`)
- **Similarity Threshold**: `0.20` (config: `similarity_threshold`)
- **Max Issues**: `10` aggregated issues (config: `max_issues`)

---

## Slide 6: Phase 5 - Issue Aggregation & Scoring

### **Per-Chunk Issue Extraction**
```python
for chunk_idx, (window_text, start_idx, end_idx) in enumerate(chunks):
    # Search vector DB for this chunk
    search_results = embeddings_manager.search(
        window_text, k=12, threshold=0.20
    )
    
    # Aggregate issue types from similar documents
    for result in search_results:
        issue_type = result['metadata']['issue_type']
        confidence = result['similarity']  # 0.20 to 1.0 range
        
        # Track highest confidence per issue type
        if confidence > issue_scores[issue_type]['confidence']:
            issue_scores[issue_type]['confidence'] = confidence
```

### **Confidence Boosting Rules**
```python
# Multiple occurrence boost
if search_count > 3:
    boosted_confidence = min(1.0, original_confidence * 1.2)

# Quality filtering with confidence decay
for rank, issue in enumerate(ranked_issues):
    decay_factor = 1.0 - (rank * 0.03)  # 3% decay per rank
    adjusted_confidence = issue['confidence'] * decay_factor
```

### **Aggregation Thresholds**
- **Multi-occurrence Boost**: 20% boost if issue found in >3 search results
- **Confidence Decay**: `3%` per ranking position (config: `confidence_decay`)
- **Quality Filter**: Generic evidence patterns penalized by 20%

---

## Slide 7: Phase 6 - LLM Validation (Critical Quality Gate)

### **LLM Validation Purpose**
```python
# Context construction for LLM
validation_context = {
    'document_excerpt': text[:2500],  # First 2500 chars
    'semantic_issues': top_7_issues,  # Top 7 from vector search
    'mapped_categories': categories,
    'validation_constraints': strict_rules
}
```

### **Multi-Provider Fallback Strategy**
```python
llm_hierarchy = [
    'claude-sonnet-4-20250514',  # Primary (Anthropic)
    'gpt-4-turbo',              # Fallback 1 (OpenAI)  
    'gemini-1.5-flash'          # Fallback 2 (Google)
]
```

### **LLM Validation Thresholds**
- **Min LLM Confidence**: `0.1` (config: `min_llm_confidence`)
- **Document Context**: `2500 characters` (config: `document_context_chars`)
- **Max Validated Issues**: `5` per document
- **Temperature**: `0.1` (low for consistency)
- **Max Tokens**: `1000` (JSON response)

### **Validation Rules Applied**
- Evidence must exist in **current document** (not training data)
- Issue must be **explicitly mentioned** or clearly implied
- Confidence **< 0.1 rejected** for this specific document
- **JSON-only response** format enforced

---

## Slide 8: Phase 7 - Advanced Confidence Filtering

### **Confidence Divergence Detection**
```python
def apply_confidence_filtering(issues):
    # Calculate confidence statistics
    top_20_percent = issues[:int(len(issues) * 0.2)]
    bottom_20_percent = issues[-int(len(issues) * 0.2):]
    
    avg_top = mean([i['confidence'] for i in top_20_percent])
    avg_bottom = mean([i['confidence'] for i in bottom_20_percent])
    
    divergence = avg_top - avg_bottom
    
    # Apply filtering if significant divergence detected
    if divergence >= 0.15:  # 15% threshold
        keep_top_percentage = 0.20  # Keep only top 20%
```

### **Source Priority Rules**
```python
# Priority order for multiple sources
source_priority = [
    'llm_validation',     # Highest priority
    'semantic_search',    # Second priority  
    'other_sources'       # Lowest priority
]
```

### **Filtering Thresholds**
- **Divergence Threshold**: `15%` gap between high/low confidence (config: `confidence_divergence_threshold`)
- **Top Percentage Filter**: Keep top `20%` when divergence detected (config: `top_percentage_filter`)
- **Quality Filter Threshold**: `0.06` (30% of similarity threshold)

---

## Slide 9: Phase 8 - Category Mapping & Validation

### **Issue to Category Mapping**
```python
class IssueCategoryMapper:
    def map_issues_to_categories(self, issues):
        # Load mapping from Excel: issues_to_category_mapping.xlsx
        for issue in issues:
            category = mapping_table.get(issue['issue_type'])
            mapped_categories.append({
                'category': category,
                'confidence': issue['confidence'] * mapping_confidence,
                'source_issue': issue['issue_type']
            })
```

### **Validation Engine**
```python
class ValidationEngine:
    def validate_issue_type(self, issue_type, auto_correct=True):
        # Check against allowlist
        if issue_type in valid_issue_types:
            return issue_type, True, 1.0
        
        # Auto-correction with fuzzy matching
        best_match, similarity = fuzzy_match(issue_type, valid_issue_types)
        if similarity > 0.7:  # 70% similarity threshold
            return best_match, False, similarity
```

### **Validation Thresholds**
- **Fuzzy Match Threshold**: `70%` similarity for auto-correction
- **Mapping Confidence**: Category mapping inherits issue confidence
- **Validation Status**: `valid`, `corrected`, or `invalid`

---

## Slide 10: Configuration Deep Dive

### **Key Configuration Parameters**
```yaml
hybrid_rag:
  # Vector Search Parameters
  top_k: 12                          # Results per chunk search
  similarity_threshold: 0.20         # Minimum similarity score
  max_issues: 10                     # Maximum aggregated issues
  
  # Chunking Parameters  
  window_size: 3                     # Sentences per chunk
  overlap: 1                         # Overlapping sentences
  
  # LLM Validation Parameters
  min_llm_confidence: 0.1            # Minimum LLM confidence
  document_context_chars: 2500       # Context sent to LLM
  
  # Quality Filtering Parameters
  confidence_decay: 0.03             # 3% decay per ranking position
  confidence_divergence_threshold: 0.15  # 15% gap for filtering
  top_percentage_filter: 0.20        # Keep top 20% when filtering
  
  # Advanced Features
  enable_confidence_filtering: true   # Enable divergence detection
```

### **Model Configuration**
```yaml
embedding_model: "all-mpnet-base-v2"    # 768-dim sentence embeddings
llm_model: "claude-sonnet-4-20250514"   # Primary LLM for validation
vector_db: "faiss"                      # IndexFlatL2 for accuracy
```

---

## Slide 11: Performance Metrics & Tuning Points

### **Precision vs Recall Tuning**
```python
# For Higher Precision (fewer false positives)
config_precision = {
    'similarity_threshold': 0.35,      # Raise threshold
    'min_llm_confidence': 0.3,         # Raise LLM confidence
    'enable_confidence_filtering': True, # Enable divergence filtering
    'top_percentage_filter': 0.15      # Keep only top 15%
}

# For Higher Recall (catch more issues)  
config_recall = {
    'similarity_threshold': 0.15,      # Lower threshold
    'min_llm_confidence': 0.05,        # Lower LLM confidence  
    'max_issues': 15,                  # Allow more issues
    'confidence_decay': 0.01           # Less ranking penalty
}
```

### **Processing Performance**
- **Average Processing Time**: ~15-30 seconds per document
- **Vector Search**: ~1-2 seconds for 12k training examples
- **LLM Validation**: ~10-20 seconds (depends on provider)
- **Memory Usage**: ~2GB for full index + models

### **Current Optimization Settings**
- **Balanced for Recall**: Threshold=0.20, Max Issues=10
- **Quality Filtering**: Enabled for precision when needed
- **Multi-provider Fallback**: Reliability over speed

---

## Slide 12: Error Handling & Fallback Strategy

### **Multi-Level Fallback Architecture**
```python
# LLM Provider Fallback Hierarchy
try:
    response = primary_llm.call(prompt)      # Claude Sonnet
except Exception as e1:
    try:
        response = fallback_openai.call(prompt)  # GPT-4 Turbo
    except Exception as e2:
        try:
            response = fallback_gemini.call(prompt)  # Gemini Flash
        except Exception as e3:
            # Fall back to semantic search only
            return semantic_results_without_llm_validation
```

### **Error Recovery Strategies**
- **LLM Validation Failure**: Use semantic search results as fallback
- **Vector Index Missing**: Return error with instructions to build index
- **Low Similarity Results**: Apply quality filtering, not complete rejection
- **Chunking Failure**: Fall back to entire document as single chunk

### **Reliability Features**
- **Provider Status Tracking**: Logs which LLM provider succeeded
- **Graceful Degradation**: System continues without LLM if all providers fail
- **Error Context**: Detailed error reporting for debugging
- **Processing Stats**: Track success/failure rates per component

---

## Slide 13: Real-World Example Walkthrough

### **Input Document**
```
Subject: Change of Scope - Additional Safety Measures Required

Dear Authority Engineer,

We need approval for additional safety barriers due to new site conditions. 
This requires design modifications and will impact the project timeline by 2 weeks.
Please advise on the approval process for this scope change.
```

### **Processing Flow**
1. **Text Extraction**: CorrespondenceExtractor identifies subject/body
2. **Chunking**: Single chunk (small document < 300 chars)
3. **Vector Search**: 12 similar examples found, top issues:
   - "Design & Drawings for COS works" (similarity: 0.75)
   - "Extension of Time" (similarity: 0.62)
   - "Authority Engineer" (similarity: 0.58)

4. **LLM Validation**: Claude confirms "Design & Drawings for COS works" and "Extension of Time"
5. **Category Mapping**: Maps to "Contractor's Obligations"
6. **Final Result**: 2 validated issues with confidence scores

### **Output Structure**
```json
{
  "identified_issues": [
    {"issue_type": "Design & Drawings for COS works", "confidence": 0.81},
    {"issue_type": "Extension of Time", "confidence": 0.67}
  ],
  "categories": [
    {"category": "Contractor's Obligations", "confidence": 0.74}
  ],
  "processing_time": 12.3
}
```

---

## Slide 14: Deployment & Monitoring Considerations

### **Production Deployment Requirements**
```yaml
# Resource Requirements
memory: "4GB minimum (8GB recommended)"
cpu: "4 cores minimum"  
storage: "10GB for models + index"
network: "Stable internet for LLM API calls"

# Dependencies
- sentence-transformers>=2.0
- faiss-cpu>=1.7 
- openai>=1.0
- anthropic>=0.3
```

### **Monitoring Metrics**
- **Processing Success Rate**: % of documents successfully processed
- **LLM Provider Usage**: Which providers are being used/failed
- **Confidence Distribution**: Track issue confidence scores over time
- **Category Distribution**: Monitor classification patterns
- **Processing Times**: Track performance bottlenecks

### **Maintenance Tasks**
- **Index Updates**: Re-build with new training data quarterly
- **Model Updates**: Monitor for new embedding models
- **Threshold Tuning**: Adjust based on precision/recall analysis
- **Provider Key Rotation**: Manage API keys securely

---

## Slide 15: Future Enhancements & Research Directions

### **Technical Improvements**
1. **Advanced Chunking**: Semantic chunking based on content similarity
2. **Dynamic Thresholds**: Auto-adjust thresholds based on document type
3. **Multi-Modal Processing**: Handle images, tables, diagrams
4. **Active Learning**: Use feedback to improve vector search quality

### **Performance Optimizations**
1. **GPU Acceleration**: FAISS GPU indices for faster search
2. **Caching Layer**: Redis cache for frequent document patterns
3. **Batch Processing**: Optimize for bulk document processing
4. **Streaming Processing**: Real-time document classification

### **Business Logic Enhancements**
1. **Hierarchical Classification**: Multi-level issue categorization
2. **Temporal Analysis**: Track issue trends over time
3. **Risk Scoring**: Assign priority scores to identified issues
4. **Integration APIs**: Webhook notifications for critical issues

### **Research Opportunities**
1. **Fine-tuned Embeddings**: Domain-specific embedding models
2. **Hybrid Retrieval**: Combine dense + sparse retrieval methods
3. **Explanation Generation**: Generate explanations for classifications
4. **Multi-language Support**: Process non-English correspondence

---

## Slide 16: Q&A and Technical Discussion Points

### **Common Questions & Answers**

**Q: Why use chunking instead of processing entire documents?**
A: Chunking prevents important context from being diluted in long documents and allows better similarity matching against training examples.

**Q: How do you handle conflicting results from different chunks?**
A: We aggregate by issue type, keeping highest confidence per issue, then apply LLM validation for final determination.

**Q: What happens if all LLM providers fail?**
A: System gracefully falls back to semantic search results only, with appropriate confidence adjustments and status flags.

**Q: How do you prevent hallucinations in LLM validation?**
A: Strict prompting requires evidence from current document only, with JSON-only responses and confidence thresholds.

### **Technical Deep-Dive Topics**
- Vector similarity mathematics and FAISS implementation details
- LLM prompt engineering techniques for structured output
- Confidence score calibration across different components
- Training data quality impact on retrieval performance

---

**Thank you for your attention!**
**Questions and Discussion Welcome**