# COMPREHENSIVE FIX PLAN: Evidence Attribution + Many-to-Many Mapping

## **EXECUTIVE SUMMARY**
The hybrid RAG classification system has two critical flaws:
1. **Evidence Attribution**: System uses evidence from vector DB training data instead of actual document being analyzed
2. **Many-to-Many Mapping**: Synthetic data generation only uses first category, breaking complex issue-category relationships

These flaws explain the 61-68% recall vs 80-85% target and poor Authority's Obligations detection.

## **PHASE 1: CRITICAL EVIDENCE ATTRIBUTION FIX**

### **1.1 Root Cause Analysis**
**Current Problem:**
- `classifier/hybrid_rag.py` lines 302-320 extract `reference_sentence` from vector DB metadata
- This becomes "evidence" for current document classification
- **Result**: Evidence has nothing to do with actual document being analyzed

**Example Found:**
- **Actual Document**: "...you have not given details of Metal Beam Crash Barrier locations..."
- **System Reports**: "We are herewith re-submitting drawings for Cattle Underpass..." (NOT in document!)

### **1.2 Evidence Extraction Logic Fix**
**Modify `_semantic_search_issues()` method:**
```python
# Current (WRONG):
'evidence': '; '.join(scores['evidence'][:3])  # From training data

# Fixed (CORRECT):
'current_document_evidence': extract_evidence_from_current_doc(text, issue_type),
'reference_evidence': '; '.join(scores['evidence'][:3])  # Keep as reference
```

### **1.3 Implement Current Document Evidence Extraction**
**New function needed:**
```python
def extract_evidence_from_current_doc(document_text, issue_type):
    """Extract actual evidence from current document for given issue type"""
    # Use text matching, keyword search, and pattern matching
    # Return direct quotes from current document
    # Return None if no supporting evidence found
```

### **1.4 Enhanced LLM Validation**
**Update LLM prompt with strict evidence requirements:**
```
CRITICAL EVIDENCE RULES:
1. Extract evidence ONLY from the provided document excerpt
2. Use direct quotes from the current document
3. Do NOT use external knowledge or reference documents
4. If no evidence exists in current document, mark confidence as 0
5. Provide exact text snippets that support each classification
```

## **PHASE 2: MANY-TO-MANY MAPPING PRESERVATION**

### **2.1 Synthetic Data Generation Fix**
**Critical Issue Found:**
- `synthetic_data_generator.py` line 100: `normalized_cats[0]` (only first category)
- **Comment confirms**: "For synthetic data generation, use single category"

**Fix Required:**
```python
# Current (WRONG):
normalized_category = normalized_cats[0]  # Only first category

# Fixed (CORRECT):
for normalized_category in normalized_cats:  # ALL categories
    # Generate samples for each category this issue type maps to
```

### **2.2 Enhanced Synthetic Data Generation**
**Multi-category sample generation:**
- Generate separate samples for each category an issue type maps to
- Include context clues showing why issue appears in each category
- Preserve training data complexity in synthetic data

### **2.3 Context-Aware Issue-Category Mapping**
**Implementation:**
- Analyze document context to determine applicable categories
- Same issue type can map to different categories based on document perspective
- Enhanced confidence scoring based on context relevance

## **PHASE 3: AUTHORITY vs CONTRACTOR OBLIGATIONS ENHANCEMENT**

### **3.1 Clear Category Definitions (From User Guidance)**
**Authority's Obligations:**
- What the Authority must provide/do (land, clearances, payments, approvals)
- When document shows Authority MUST provide something
- NOT Authority Default (expropriation, payment failures)

**Contractor's Obligations:**
- What the Contractor must deliver/do (designs, safety, construction, reporting)  
- When document shows Contractor MUST deliver something

### **3.2 Enhanced LLM Prompt with Obligation Distinctions**
```
AUTHORITY vs CONTRACTOR OBLIGATIONS GUIDANCE:

Authority's Obligations = When Authority has duty to provide/enable
- Examples: Land handover, clearances, utility shifting, mobilization
- Key indicators: "Authority shall provide", "Authority must arrange"

Contractor's Obligations = When Contractor has duty to deliver/perform  
- Examples: Design submission, safety measures, construction, reporting
- Key indicators: "Contractor shall submit", "Contractor must ensure"

Context-based Decision Rules:
- If document shows Authority MUST provide → Authority's Obligations
- If document shows Contractor MUST deliver → Contractor's Obligations
- Same issue type can belong to different categories based on document context
```

### **3.3 Update Synthetic Data with Obligation Context**
**Enhanced prompts:**
- Include Authority vs Contractor distinction in generation
- Create samples showing same issue from different obligation perspectives
- Add user's clarification about Authority Default vs Authority Obligation

## **PHASE 4: COMPREHENSIVE VALIDATION SYSTEM**

### **4.1 Evidence Source Verification**
**Mandatory validation:**
- All evidence must exist in current document
- Text matching verification with confidence scoring
- Separate tracking of current document vs reference evidence

### **4.2 Dual Evidence Reporting Structure**
```json
{
  "issue_type": "Design & Drawings for COS works",
  "confidence": 0.75,
  "current_document_evidence": "you are advised to submit the location along with length of Metal Beam Crash Barrier",
  "reference_evidence": "We are herewith re-submitting drawings for Cattle Underpass",
  "evidence_source": "current_document",
  "verification_status": "verified"
}
```

### **4.3 Many-to-Many Validation Testing**
- Test complex issue types mapping to multiple categories
- Validate synthetic data includes all category relationships  
- Ground truth comparison for overlapping classifications

## **PHASE 5: IMPLEMENTATION ROADMAP**

### **IMMEDIATE PRIORITY (Week 1):**
1. **Fix evidence attribution** in `hybrid_rag.py` lines 302-320
2. **Remove synthetic data limitation** in `synthetic_data_generator.py` line 100
3. **Update LLM validation prompts** with evidence requirements

### **HIGH PRIORITY (Week 2):**
4. **Implement current document evidence extraction**
5. **Add Authority/Contractor obligation distinctions** to prompts
6. **Test evidence attribution fixes** on LOT-21

### **MEDIUM PRIORITY (Week 3-4):**
7. **Regenerate synthetic data** with many-to-many support
8. **Enhance issue-category mapping** with context awareness
9. **Comprehensive validation testing**

### **LONG-TERM (Month 2):**
10. **Performance optimization** and monitoring
11. **Automated evidence verification**
12. **Advanced context-aware classification**

## **EXPECTED OUTCOMES**

### **Evidence Quality:**
- **100% verifiable evidence** from actual documents
- **Zero hallucinated evidence** from external sources
- **Clear source attribution** for all classifications

### **Classification Accuracy:**
- **Recall improvement**: 61-68% → 80-85% target
- **Authority's Obligations**: Accurate detection with proper context
- **Many-to-many relationships**: Preserved and enhanced

### **System Reliability:**
- **Trustworthy classifications** with traceable evidence  
- **Context-aware mapping** for complex issue types
- **Comprehensive training data** reflecting real-world complexity

## **SUCCESS METRICS**
- **Evidence Accuracy**: 100% traceable to source document
- **Recall Rate**: Achieve 80-85% target on LOT-21
- **Authority Obligations**: >90% accurate distinction
- **Many-to-Many Coverage**: All training data relationships preserved

This plan addresses the fundamental evidence attribution flaw while preserving and enhancing the complex many-to-many mapping capability essential for accurate contract correspondence classification.

---
*Generated on: September 12, 2025*
*Target Completion: Week 1-2 for critical fixes, Month 1-2 for full implementation*