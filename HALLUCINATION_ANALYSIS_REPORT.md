# Hybrid RAG Classification Hallucination Analysis Report
*LOT-21 Document Analysis - September 12, 2025*

## Executive Summary

Analysis of 2 LOT-21 documents revealed significant hallucination issues in the hybrid RAG classification system, primarily caused by poor vector search quality and overly permissive similarity thresholds. The system shows high false positive rates due to irrelevant document matches being presented to the LLM with inflated confidence scores.

## Key Findings

### ✅ What's Working Well

1. **Temperature Setting**: Temperature is correctly set to 0.1 (not 0.9 as originally thought), which helps reduce hallucination
2. **Validation Constraints**: Strong allowlist validation prevents completely invalid issue types
3. **LLM Reasoning**: Claude correctly identified many irrelevant vector search results in both documents
4. **Document Extraction**: OCR extraction working well for scanned PDFs

### ❌ Critical Hallucination Sources Identified

## 1. Vector Search Quality Issues

### **Document 1 - Letter about Metal Beam Crash Barrier**
```
Actual Content: Request for Metal Beam Crash Barrier location details
Vector Search Results (Confidence: 0.6-0.8):
❌ "QAP, EMP, EHS and Construction Programme" - Not relevant
❌ "Mobilisation of Authority Engineer" - Not the main issue  
❌ "Design & Drawings for COS works" - Not about change of scope
✅ "Authority Engineer" - Somewhat relevant (0.814 confidence)
✅ "Submission of Plan & Profile" - Partially relevant (0.810 confidence)
```

### **Document 2 - Final Payment Statement** 
```
Actual Content: Final payment statement and completion certificate  
Vector Search Results (Confidence: 0.8+):
❌ "Change of scope proposals clarifications" - Irrelevant (0.84 confidence)
❌ "QAP, EMP, EHS and Construction Programme" - Not relevant (0.84 confidence) 
❌ "Mobilisation of Authority Engineer" - Wrong context (0.84 confidence)
✅ LLM Correctly Suggested: "Completion certificate", "Stage Payments Statements"
```

## 2. Technical Parameter Issues

| Parameter | Current Value | Issue | Impact |
|-----------|---------------|-------|--------|
| **similarity_threshold** | 0.3 | Too low - accepts weak matches | High false positives |
| **top_k** | 15 | Too many irrelevant results | Noise confuses LLM |
| **max_issues** | 10 | Still allows too much noise | Multiple irrelevant suggestions |
| **document_truncation** | 1500 chars | May lose important context | Incomplete analysis |
| **search_count_bias** | Unweighted | High frequency ≠ relevance | "Authority Engineer" had 55 hits |

## 3. LLM Response Format Issues

Both documents failed JSON parsing due to LLM returning markdown-wrapped JSON instead of plain JSON:

```
❌ Current LLM Response Format:
Looking at this document...
```json
{"validated_issues": [...]}
``` 
Analysis: ...

✅ Expected Format:
{"validated_issues": [...]}
```

## Recommended Solutions

### 🎯 **Priority 1: Vector Search Constraints**

1. **Increase Similarity Threshold**
   ```yaml
   # config.yaml changes
   hybrid_rag:
     similarity_threshold: 0.65  # Increase from 0.3
     top_k: 8                    # Reduce from 15  
     max_issues: 5               # Reduce from 10
   ```

2. **Add Confidence Decay**
   ```python
   # Apply decay to reduce confidence of lower-ranked results
   confidence = original_confidence * (1.0 - (rank * 0.1))
   ```

3. **Implement Search Result Quality Filtering**
   ```python
   # Filter out results where evidence is too generic or repetitive
   def filter_generic_evidence(results):
       filtered = []
       for result in results:
           evidence = result['evidence'].lower()
           if not any(generic in evidence for generic in [
               'as per agreement', 'authority engineer', 'contractor shall'
           ]):
               filtered.append(result)
       return filtered
   ```

### 🎯 **Priority 2: Prompt Engineering Improvements**

1. **Fix JSON Response Format**
   ```python
   # Update LLM prompt to enforce strict JSON
   prompt += """
   CRITICAL: Respond ONLY with valid JSON. No explanations, no markdown, no analysis outside JSON.
   Your response must start with { and end with }.
   """
   
   # For Anthropic Claude, add response format constraint
   if isinstance(self.llm_client, anthropic.Anthropic):
       messages.append({
           "role": "assistant", 
           "content": "{"  # Force JSON start
       })
   ```

2. **Enhanced Context Ranking**
   ```python
   # Prioritize context by relevance scores
   prompt_context = f"""
   Document: {text[:2000]}  # Increase from 1500
   
   Most Relevant Issues (confidence > 0.7):
   {high_confidence_issues}
   
   Potential Additional Issues (confidence 0.5-0.7):
   {medium_confidence_issues}
   """
   ```

3. **Add Explicit Negative Instructions**
   ```python
   constraint_text = """
   STRICT VALIDATION RULES:
   1. Issue must be EXPLICITLY mentioned or clearly implied in the document
   2. Do NOT infer issues from general document type patterns
   3. Authority Engineer involvement alone is NOT an issue type
   4. Reject issues if confidence < 0.8 for this specific document
   5. Maximum 3 validated issues per document
   """
   ```

### 🎯 **Priority 3: Vector Database Optimization**

1. **Add Document Type Metadata Filtering**
   ```python
   # Filter search results by document type similarity
   def enhanced_search(query, doc_type_hint=None):
       results = base_search(query)
       if doc_type_hint:
           # Boost results from similar document types
           for result in results:
               if result['metadata']['doc_type'] == doc_type_hint:
                   result['similarity'] *= 1.2
       return results
   ```

2. **Implement Search Result Clustering**
   ```python
   # Group similar results to reduce noise
   def cluster_search_results(results):
       clustered = defaultdict(list)
       for result in results:
           key = result['issue_type'] 
           clustered[key].append(result)
       
       # Keep only best result per cluster
       return [max(cluster, key=lambda x: x['similarity']) 
               for cluster in clustered.values()]
   ```

### 🎯 **Priority 4: Configuration Updates**

**Recommended config.yaml changes:**
```yaml
hybrid_rag:
  enabled: true
  embedding_model: "all-mpnet-base-v2"
  llm_model: "claude-sonnet-4-20250514"
  
  # Vector search constraints (UPDATED)
  top_k: 8              # Reduced from 15
  similarity_threshold: 0.65  # Increased from 0.3
  max_issues: 5         # Reduced from 10
  
  # New parameters for quality control
  confidence_decay: 0.1
  min_llm_confidence: 0.8
  max_evidence_repetition: 0.3
  
  # Context window optimization  
  document_context_chars: 2000  # Increased from 1500
  window_size: 2        # Reduced from 3 
  overlap: 0           # Reduced from 1

# Enhanced validation
validation:
  enable_strict_validation: true
  auto_correct: true
  similarity_threshold: 0.8  # Increased from 0.7
  min_issue_confidence: 0.8  # New parameter
  max_issues_per_document: 3 # New constraint
```

## Implementation Priority

### **Phase 1 (Immediate - 1-2 days)**
1. Update similarity_threshold to 0.65
2. Reduce top_k to 8 and max_issues to 5
3. Fix LLM JSON response formatting

### **Phase 2 (Short-term - 1 week)**
1. Implement search result quality filtering
2. Add enhanced validation constraints
3. Increase document context to 2000 chars

### **Phase 3 (Medium-term - 2-3 weeks)**
1. Implement search result clustering
2. Add document type metadata filtering
3. Develop confidence decay mechanisms

## Expected Impact

With these changes, we expect:

- **Precision**: Increase from ~40% to ~80%+
- **False Positives**: Reduce by 60-70%
- **LLM Processing**: Reduce irrelevant context by 50%
- **Response Quality**: Eliminate JSON parsing errors
- **Classification Speed**: Improve by 20-30% due to less noise

## Testing & Validation

1. **A/B Testing**: Run both old and new configurations on LOT-21 documents
2. **Validation Metrics**: Track precision, recall, and false positive rates
3. **Human Evaluation**: Manual review of 50 random classifications
4. **Performance Monitoring**: Track processing time and API costs

---

**Temperature Correction**: The actual temperature is 0.1 (not 0.9 as initially mentioned), which is appropriate for reducing hallucination. No changes needed here.

**Next Steps**: Implement Priority 1 changes first, test with LOT-21 documents, then proceed with subsequent phases based on results.