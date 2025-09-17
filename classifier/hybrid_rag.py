"""
Hybrid RAG+LLM Classifier Module
Combines semantic search with LLM validation for classification
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple
import logging
import pandas as pd
from collections import defaultdict
from openai import OpenAI
import anthropic
from pathlib import Path

# Import Google Generative AI for fallback support
try:
    import google.generativeai as genai
except ImportError:
    genai = None

from .issue_mapper import IssueCategoryMapper
from .validation import ValidationEngine
from .data_sufficiency import DataSufficiencyAnalyzer
from .preprocessing import TextPreprocessor
from .pdf_extractor import PDFExtractor
from .embeddings import EmbeddingsManager
from .document_quality import DocumentQualityChecker

logger = logging.getLogger(__name__)


class HybridRAGClassifier:
    """
    Classifies contract correspondence using RAG (Retrieval Augmented Generation) + LLM.
    Uses semantic search to find similar examples, then validates with LLM.
    """
    
    def __init__(self,
                 config: Dict,
                 issue_mapper: IssueCategoryMapper,
                 validator: ValidationEngine,
                 data_analyzer: DataSufficiencyAnalyzer):
        """
        Initialize Hybrid RAG Classifier.
        
        Args:
            config: Configuration dictionary for the classifier
            issue_mapper: Issue to category mapper
            validator: Validation engine for preventing hallucinations
            data_analyzer: Data sufficiency analyzer
        """
        self.config = config
        self.mapper = issue_mapper
        self.validator = validator
        self.data_analyzer = data_analyzer
        
        # Initialize components
        self.preprocessor = TextPreprocessor()
        self.pdf_extractor = PDFExtractor()
        self.quality_checker = DocumentQualityChecker(config.get('quality_settings', {}))
        
        # Configuration parameters
        self.llm_model = config.get('llm_model', 'gpt-3.5-turbo')
        
        # Initialize embeddings manager
        embedding_model = config.get('embedding_model', 'all-mpnet-base-v2')
        self.embeddings_manager = EmbeddingsManager(model_name=embedding_model)
        
        # Initialize LLM client for validation
        self.llm_client = self._init_llm_client()
        self.top_k = config.get('top_k', 12)
        self.window_size = config.get('window_size', 3)
        self.overlap = config.get('overlap', 1)
        self.similarity_threshold = config.get('similarity_threshold', 0.20)
        self.confidence_decay = config.get('confidence_decay', 0.03)
        self.min_llm_confidence = config.get('min_llm_confidence', 0.3)
        
        # Initialize fallback LLM clients for hierarchical retries
        self.fallback_llm_clients = self._init_fallback_llm_clients()
        self.current_llm_provider = self._get_provider_name_from_model(self.llm_model)
        
        # Build or load index
        self.index_built = False
        self._initialize_index()
        
        logger.info(f"HybridRAGClassifier initialized with {embedding_model}")
    
    def _init_llm_client(self):
        """
        Initialize the LLM client for validation.
        
        Returns:
            LLM client instance
        """
        # Try to get API key from config first, then environment
        api_key = self.config.get('api_key')
        
        if 'gpt' in self.llm_model.lower():
            if not api_key:
                api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key required for GPT models")
            # Don't set organization unless it's provided
            org_id = os.getenv('OPENAI_ORG_ID')
            if org_id:
                return OpenAI(api_key=api_key, organization=org_id)
            else:
                return OpenAI(api_key=api_key)
        
        elif 'claude' in self.llm_model.lower():
            if not api_key:
                api_key = os.getenv('CLAUDE_API_KEY')
            if not api_key:
                raise ValueError("Anthropic API key required for Claude models")
            return anthropic.Anthropic(api_key=api_key)
        
        else:
            # Default to OpenAI
            try:
                return OpenAI(api_key=api_key) if api_key else None
            except TypeError as e:
                if "proxies" in str(e):
                    # Handle legacy OpenAI client version compatibility
                    logger.warning(f"OpenAI client initialization issue: {e}. Trying compatibility mode.")
                    try:
                        # Try without any optional parameters
                        import openai
                        openai.api_key = api_key
                        return openai
                    except Exception as fallback_error:
                        logger.error(f"Failed to initialize OpenAI client: {fallback_error}")
                        return None
                else:
                    raise
    
    def _init_fallback_llm_clients(self):
        """Initialize fallback LLM clients for hierarchical retries."""
        fallback_clients = {}
        
        # Try to initialize Anthropic Claude
        anthropic_key = os.getenv('CLAUDE_API_KEY')
        if anthropic_key:
            try:
                fallback_clients['anthropic'] = {
                    'client': anthropic.Anthropic(api_key=anthropic_key),
                    'model': 'claude-sonnet-4-20250514',
                    'name': 'Anthropic Claude'
                }
                logger.info("Initialized Anthropic Claude fallback client")
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic fallback: {e}")
        
        # Try to initialize OpenAI GPT with enhanced error handling
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            try:
                # Try modern OpenAI client initialization first
                try:
                    fallback_clients['openai'] = {
                        'client': OpenAI(api_key=openai_key),
                        'model': 'gpt-4-turbo',
                        'name': 'OpenAI GPT-4'
                    }
                    logger.info("Initialized OpenAI GPT-4 fallback client (modern)")
                except Exception as modern_error:
                    logger.warning(f"Modern OpenAI client failed: {modern_error}")
                    
                    # Try legacy OpenAI client initialization
                    try:
                        import openai as legacy_openai
                        legacy_openai.api_key = openai_key
                        fallback_clients['openai'] = {
                            'client': legacy_openai,
                            'model': 'gpt-4-turbo',
                            'name': 'OpenAI GPT-4'
                        }
                        logger.info("Initialized OpenAI GPT-4 fallback client (legacy)")
                    except Exception as legacy_error:
                        logger.warning(f"Legacy OpenAI client also failed: {legacy_error}")
                        logger.warning("OpenAI fallback client will be unavailable for hierarchical retries")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI fallback: {e}")
        
        # Try to initialize Google Gemini
        gemini_key = os.getenv('GOOGLE_API_KEY')
        if gemini_key and genai:
            try:
                genai.configure(api_key=gemini_key)
                fallback_clients['gemini'] = {
                    'client': genai.GenerativeModel('gemini-1.5-flash'),
                    'model': 'gemini-1.5-flash',
                    'name': 'Google Gemini'
                }
                logger.info("Initialized Google Gemini fallback client")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini fallback: {e}")
        elif gemini_key and not genai:
            logger.warning("Google API key found but google-generativeai package not installed")
        
        logger.info(f"Initialized {len(fallback_clients)} fallback LLM clients: {list(fallback_clients.keys())}")
        
        # If no fallback clients are available, warn but don't fail
        if not fallback_clients:
            logger.warning("No fallback LLM clients available - hierarchical retries disabled")
        
        return fallback_clients
    
    def _initialize_index(self):
        """Initialize the vector index from training data."""
        # Check if pre-built index exists
        index_path = Path(self.config.get('index_path', './data/embeddings/rag_index'))
        
        if index_path.with_suffix('.faiss').exists():
            logger.info(f"Loading pre-built index from {index_path}")
            self.embeddings_manager.load_index(str(index_path))
            self.index_built = True
        else:
            logger.info("No pre-built index found. Call build_index() to create one.")
    
    def build_index(self, training_data_path: str, save_path: str = None):
        """
        Build vector index from training data.
        
        Args:
            training_data_path: Path to training data Excel file
            save_path: Optional path to save the index
        """
        logger.info(f"Building index from {training_data_path}")
        
        try:
            # Load training data
            df = pd.read_excel(training_data_path)
            
            # Prepare texts and metadata for indexing
            texts = []
            metadata = []
            
            for _, row in df.iterrows():
                # Combine subject and body for better context
                text = f"{row.get('subject', '')} {row.get('body', '')}"
                
                # Clean and preprocess text
                text = self.preprocessor.clean_text(text)
                
                if text.strip():
                    texts.append(text)
                    metadata.append({
                        'issue_type': row.get('issue_type', ''),
                        'category': row.get('category', ''),
                        'reference_sentence': row.get('reference_sentence', ''),
                        'source_file': row.get('source_file', '')
                    })
            
            # Build index
            self.embeddings_manager.build_index(texts, metadata, save_path)
            self.index_built = True
            
            logger.info(f"Index built with {len(texts)} documents")
            
        except Exception as e:
            logger.error(f"Error building index: {e}")
            raise
    
    def classify(self, document: str, is_file_path: bool = False) -> Dict:
        """
        Classify a document using hybrid RAG+LLM approach.
        
        Args:
            document: Document text or path to document file
            is_file_path: Whether document is a file path
            
        Returns:
            Classification results with validation and sufficiency warnings
        """
        if not self.index_built:
            return {
                'status': 'error',
                'message': 'Index not built. Call build_index() first.'
            }
        
        start_time = time.time()
        
        # Extract text if file path provided
        if is_file_path:
            document_text, extraction_method = self._extract_document_text(document)
        else:
            document_text = document
            extraction_method = "direct_text"
        
        if not document_text:
            return {
                'status': 'error',
                'message': 'Failed to extract text from document',
                'extraction_method': extraction_method
            }
        
        # DEBUG: Log input document analysis
        logger.info(f"🔍 DEEP DEBUG - DOCUMENT ANALYSIS START")
        logger.info(f"📄 Raw document length: {len(document_text)} chars")
        logger.info(f"📄 Document preview: '{document_text[:200]}...'") 
        
        # CONDITIONAL QUALITY CHECK: Only apply for documents with poor extraction results
        # Apply quality filtering only if the document appears to have failed proper extraction
        if len(document_text.strip()) < 200 or "scanned by" in document_text.lower():
            should_skip, skip_reason = self.quality_checker.should_skip_classification(document_text)
            if should_skip:
                logger.warning(f"⚠️ QUALITY FILTER: Skipping low-quality document - {skip_reason}")
                return {
                    'status': 'skipped',
                    'message': f'Document quality too low: {skip_reason}',
                    'quality_check': 'failed',
                    'categories': [],
                    'identified_issues': [],
                    'extraction_method': extraction_method,
                    'processing_time': time.time() - start_time
                }
        else:
            logger.info(f"✅ QUALITY CHECK: Document passed quality validation (skipped for good extraction)")
        
        # Preprocess text
        processed_text = self.preprocessor.normalize_document(document_text)
        logger.info(f"🔧 Processed text length: {len(processed_text)} chars")
        logger.info(f"🔧 Processed text preview: '{processed_text[:200]}...'")
        
        # Phase 1: Semantic search to find similar issues
        logger.info(f"🔍 PHASE 1: Starting semantic search...")
        similar_issues = self._semantic_search_issues(processed_text)
        logger.info(f"🔍 PHASE 1: Found {len(similar_issues)} similar issues from semantic search")
        for i, issue in enumerate(similar_issues[:3]):
            logger.info(f"   Issue {i+1}: {issue.get('issue_type', 'Unknown')} (conf: {issue.get('confidence', 0):.3f})")
        
        # Phase 2: Map identified issues to categories
        logger.info(f"🗂️  PHASE 2: Mapping {len(similar_issues)} issues to categories...")
        categories = self.mapper.map_issues_to_categories(similar_issues)
        logger.info(f"🗂️  PHASE 2: Mapped to {len(categories)} categories")
        for i, cat in enumerate(categories[:3]):
            logger.info(f"   Category {i+1}: {cat.get('category', 'Unknown')} (conf: {cat.get('confidence', 0):.3f})")
        
        # Phase 3: LLM validation and refinement
        logger.info(f"🤖 PHASE 3: Starting LLM validation...")
        if self.llm_client:
            logger.info(f"🤖 LLM client available, proceeding with validation")
            refined_results = self._llm_validation(processed_text, similar_issues, categories)
            logger.info(f"🤖 LLM validation completed. Status: {refined_results.get('status', 'unknown')}")
            
            # Check if LLM validation failed
            if refined_results.get('llm_error', False):
                # Return error results instead of continuing with unreliable data
                return {
                    'status': 'error',
                    'error_type': 'llm_validation_failed',
                    'message': refined_results.get('llm_error_message', 'LLM validation failed'),
                    'provider': refined_results.get('llm_provider', 'unknown'),
                    'raw_semantic_results_count': refined_results.get('raw_semantic_results', 0),
                    'extraction_method': extraction_method,
                    'processing_time': time.time() - start_time,
                    'error_details': refined_results.get('error_details', {})
                }
        else:
            logger.warning(f"🤖 No LLM client available - skipping validation")
            refined_results = {
                'identified_issues': similar_issues,
                'categories': categories
            }
        
        # Phase 4: Apply validation
        logger.info(f"✅ PHASE 4: Validating {len(refined_results.get('identified_issues', []))} issues...")
        validated_issues = self._validate_issues(refined_results.get('identified_issues', []))
        logger.info(f"✅ PHASE 4: {len(validated_issues)} issues passed validation")
        
        # Phase 5: Apply source priority (LLM validation > semantic search)
        logger.info(f"🔄 PHASE 5: Applying source priority to {len(validated_issues)} issues...")
        validated_issues = self._apply_source_priority(validated_issues)
        logger.info(f"🔄 PHASE 5: After priority filtering: {len(validated_issues)} issues remain")
        
        # Phase 6: Apply confidence-based filtering if enabled
        if self.config.get('enable_confidence_filtering', False):
            validated_issues = self._apply_confidence_filtering(validated_issues)
        
        # CRITICAL FIX: Always re-map categories from validated issues only
        # This ensures categories are only returned if they have valid underlying issues
        # Phase 6: CRITICAL FIX - Always re-map categories from validated issues only
        logger.info(f"🔄 PHASE 6: Re-mapping categories from {len(validated_issues)} validated issues...")
        if validated_issues:
            validated_categories = self.mapper.map_issues_to_categories(validated_issues)
            logger.info(f"🔄 PHASE 6: Generated {len(validated_categories)} categories from validated issues")
        else:
            validated_categories = []
            logger.info(f"🔄 PHASE 6: No validated issues - returning 0 categories")
        
        # Phase 7: Apply data sufficiency adjustments
        result = {
            'identified_issues': validated_issues,
            'categories': validated_categories,
            'classification_path': 'semantic_search → issue_aggregation → llm_validation → category_mapping',
            'extraction_method': extraction_method,
            'search_results_used': len(similar_issues),
            'processing_time': time.time() - start_time,
            'chunk_debug_data': getattr(self, 'chunk_debug_data', [])  # Include chunk-level debugging
        }
        
        # Apply confidence adjustments
        result = self.data_analyzer.apply_confidence_adjustments(result)
        
        return result
    
    def _extract_document_text(self, file_path: str) -> Tuple[str, str]:
        """
        Extract text from document file.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Tuple of (extracted_text, extraction_method)
        """
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.pdf':
            return self.pdf_extractor.extract_text(file_path)
        elif file_path.suffix.lower() in ['.txt', '.text']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read(), 'text_file'
        else:
            logger.error(f"Unsupported file type: {file_path.suffix}")
            return "", "unsupported"
    
    def _semantic_search_issues(self, text: str) -> List[Dict]:
        """
        Use semantic search to find similar issues from training data.
        
        Args:
            text: Document text
            
        Returns:
            List of identified issues with confidence scores
        """
        # ENHANCED: Create sliding windows with smart sizing for small documents
        text_length = len(text)
        chunk_size_chars = self.window_size * 100  # Approximate chars per window
        
        logger.info(f"🔍 SEMANTIC SEARCH DEBUG: Input text analysis")
        logger.info(f"   📄 Text length: {text_length} chars")
        logger.info(f"   📄 Chunk size threshold: {chunk_size_chars} chars")
        logger.info(f"   📄 Text preview: '{text[:100]}...'") 
        
        # For small documents (< chunk size), use entire document as single chunk
        if text_length <= chunk_size_chars:
            windows = [(text, 0, text_length)]
            logger.info(f"📄 Small document ({text_length} chars) - using single chunk")
            logger.info(f"   🔎 Single chunk content: '{text[:200]}...'")
        else:
            windows = self.preprocessor.create_sliding_windows(
                text, 
                window_size=self.window_size, 
                overlap=self.overlap
            )
            # If no windows created, fall back to whole text
            if not windows:
                windows = [(text, 0, text_length)]
                logger.warning(f"   ⚠️ No windows created, falling back to whole text")
            
            logger.info(f"📄 Large document ({text_length} chars) - created {len(windows)} chunks")
            for i, (window_text, start, end) in enumerate(windows[:3]):
                logger.info(f"   🔎 Chunk {i+1}: '{window_text[:100]}...' (pos: {start}-{end})")
        
        # Initialize chunk-level debugging data
        self.chunk_debug_data = []
        
        # Search for each window and aggregate results
        logger.info(f"🔍 Starting semantic search on {len(windows)} chunks...")
        all_matches = []
        issue_scores = defaultdict(lambda: {'confidence': 0, 'evidence': [], 'count': 0})
        
        for chunk_idx, (window_text, start_idx, end_idx) in enumerate(windows, 1):
            chunk_length = len(window_text)
            chunk_issues_found = []
            chunk_search_results = []
            
            logger.info(f"📊 Processing Chunk {chunk_idx}: {chunk_length} chars ({start_idx}-{end_idx})")
            
            # Search for similar documents
            search_results = self.embeddings_manager.search(
                window_text, 
                k=self.top_k,
                threshold=self.similarity_threshold
            )
            
            chunk_search_results = search_results.copy()  # Store for debugging
            
            # Aggregate issue types from search results
            chunk_issue_types = set()
            
            for result in search_results:
                metadata = result.get('metadata', {})
                issue_type = metadata.get('issue_type', '')
                
                if issue_type:
                    chunk_issue_types.add(issue_type)
                    
                    # Update confidence (keep highest)
                    if result['similarity'] > issue_scores[issue_type]['confidence']:
                        issue_scores[issue_type]['confidence'] = result['similarity']
                    
                    # FIXED: Extract evidence from current document, not vector DB training data
                    current_doc_evidence = self._extract_evidence_from_current_doc(window_text, issue_type)
                    if current_doc_evidence and current_doc_evidence not in issue_scores[issue_type]['evidence']:
                        issue_scores[issue_type]['evidence'].append(current_doc_evidence)
                    
                    # Keep reference for comparison but don't use as primary evidence
                    reference = metadata.get('reference_sentence', '')
                    if reference:
                        issue_scores[issue_type].setdefault('reference_evidence', []).append(reference)
                    
                    # Count occurrences
                    issue_scores[issue_type]['count'] += 1
            
            # Capture chunk-level debugging data
            chunk_debug_entry = {
                'chunk_id': chunk_idx,
                'chunk_text': window_text[:200] + '...' if len(window_text) > 200 else window_text,
                'chunk_length': chunk_length,
                'start_pos': start_idx,
                'end_pos': end_idx,
                'search_results_count': len(search_results),
                'unique_issues_found': len(chunk_issue_types),
                'issues_list': '; '.join(sorted(chunk_issue_types)),
                'top_similarities': [r.get('similarity', 0) for r in search_results[:3]],
                'avg_similarity': sum(r.get('similarity', 0) for r in search_results) / max(1, len(search_results))
            }
            self.chunk_debug_data.append(chunk_debug_entry)
            
            logger.info(f"  ↳ Found {len(chunk_issue_types)} unique issues: {', '.join(list(chunk_issue_types)[:3])}{'...' if len(chunk_issue_types) > 3 else ''}")
        
        # Convert to list format
        identified_issues = []
        for issue_type, scores in issue_scores.items():
            # Boost confidence based on multiple occurrences
            confidence = scores['confidence']
            if scores['count'] > 3:
                confidence = min(1.0, confidence * 1.2)
            
            identified_issues.append({
                'issue_type': issue_type,
                'confidence': confidence,
                'evidence': '; '.join(scores['evidence'][:3]) if scores['evidence'] else '',  # Top 3 evidence from current document
                'reference_evidence': '; '.join(scores.get('reference_evidence', [])[:2]),  # Keep reference for debugging
                'search_count': scores['count'],
                'source': 'semantic_search'
            })
        
        # Sort by confidence
        identified_issues.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Apply confidence decay for ranking and quality filtering
        identified_issues = self._apply_quality_filtering(identified_issues)
        
        # Keep top issues (to avoid noise)
        max_issues = self.config.get('max_issues', 5)
        identified_issues = identified_issues[:max_issues]
        
        # LOG VECTOR SEARCH RESULTS FOR ANALYSIS
        logger.info("+"*80)
        logger.info("VECTOR SEARCH RESULTS FOR LLM CONTEXT")
        logger.info("+"*80)
        logger.info(f"Document processed in {len(windows)} chunks")
        logger.info(f"Search Parameters: top_k={self.top_k}, similarity_threshold={self.similarity_threshold}")
        logger.info(f"Max issues kept: {max_issues}")
        logger.info("IDENTIFIED ISSUES FROM VECTOR SEARCH:")
        for i, issue in enumerate(identified_issues, 1):
            logger.info(f"{i}. Issue: {issue['issue_type']}")
            logger.info(f"   Confidence: {issue['confidence']:.3f}")
            logger.info(f"   Search Count: {issue['search_count']}")
            logger.info(f"   Evidence: {issue['evidence'][:100]}...")
            logger.info("")
        logger.info("+"*80)
        
        logger.info(f"Identified {len(identified_issues)} issues via semantic search")
        return identified_issues
    
    def _llm_validation(self, text: str, issues: List[Dict], categories: List[Dict]) -> Dict:
        """
        Use LLM to validate and refine RAG results.
        
        Args:
            text: Document text
            issues: Issues identified by semantic search
            categories: Categories mapped from issues
            
        Returns:
            Refined results
        """
        logger.info(f"🤖 LLM VALIDATION DEBUG: Starting validation")
        logger.info(f"   📄 Input text length: {len(text)} chars")
        logger.info(f"   📄 Input text preview: '{text[:200]}...'")
        logger.info(f"   🔍 Input issues count: {len(issues)}")
        logger.info(f"   🔍 Input categories count: {len(categories)}")
        
        if not self.llm_client:
            logger.warning(f"   ⚠️ No LLM client available")
            return {'identified_issues': issues, 'categories': categories}
            
        if not issues:
            logger.warning(f"   ⚠️ No issues to validate")
            return {'identified_issues': issues, 'categories': categories}
        
        # Prepare context for LLM
        issue_summary = [
            f"- {i['issue_type']} (confidence: {i['confidence']:.2f})"
            for i in issues[:7]
        ]
        logger.info(f"   📅 Issue summary: {issue_summary}")
        
        category_summary = [
            f"- {c['category']}"
            for c in categories[:7]
        ]
        logger.info(f"   🗂️ Category summary: {category_summary}")
        
        # Get validation constraints
        constraints = self.validator.create_constrained_prompt('issues')
        logger.info(f"   🔒 Validation constraints loaded: {len(constraints)} chars")
        
        # Get enhanced document context
        context_chars = self.config.get('document_context_chars', 2000)
        logger.info(f"   📄 Using document context: {context_chars} chars")
        
        prompt = f"""
Review and validate these classification results from semantic search.

Document excerpt:
{text[:context_chars]}

Identified issues (from similar documents):
{chr(10).join(issue_summary)}

Mapped categories:
{chr(10).join(category_summary)}

{constraints}

EVIDENCE REQUIREMENTS:
1. Use both direct quotes AND reasonable inference from document context
2. Consider document context and industry patterns for contract correspondence
3. Authority Engineer correspondence indicates Authority obligations
4. Contract language indicates specific party obligations
5. Provide supporting text evidence for each classification

POSITIVE ISSUE DETECTION GUIDANCE:

Authority's Obligations - Look for:
✅ "Authority Engineer issued instructions/letters"
✅ "Authority shall provide/arrange/ensure"  
✅ "Authority Engineer approval/clearance required"
✅ "Authority's responsibility to deliver"
✅ Authority delays or failures to provide

Contractor's Obligations - Look for:
✅ "Contractor shall submit/deliver/ensure"
✅ "As per contract, contractor must"
✅ Contractor reporting requirements
✅ Contractor performance obligations
✅ Contractor delays or compliance issues

VALIDATION RULES:
6. Issue must be mentioned OR reasonably inferred from document context
7. Authority Engineer involvement IS a valid Authority obligation issue
8. Accept issues with confidence >= {self.min_llm_confidence}
9. Maximum 12 validated issues per document (increased for comprehensive detection)

ISSUE TYPE DISAMBIGUATION (KEY DISTINCTIONS):

COS = Change of Scope (specific scope modifications)
- "Design & Drawings for COS works" = Drawings for scope changes/modifications ONLY
- "Submission of Design and Drawings" = General design/drawing submissions (broader)

Choose the most specific and appropriate issue type:
- General submission requests → "Submission of Design and Drawings" 
- Change of scope specific work → "Design & Drawings for COS works"
- Plan/profile submissions → "Submission of Plan & Profile"

AUTHORITY vs CONTRACTOR OBLIGATIONS GUIDANCE:

Authority's Obligations = When Authority has duty to provide/enable
- Examples: Land handover, clearances, utility shifting, mobilization, payments to contractor
- Key indicators: "Authority shall provide", "Authority must arrange", "Authority to ensure"
- Document shows Authority MUST provide → Authority's Obligations

Contractor's Obligations = When Contractor has duty to deliver/perform  
- Examples: Design submission, safety measures, construction, reporting, quality compliance
- Key indicators: "Contractor shall submit", "Contractor must ensure", "Contractor to deliver"
- Document shows Contractor MUST deliver → Contractor's Obligations

Note: Same issue type can belong to different categories based on document context

CRITICAL RESPONSE FORMAT:
- Respond ONLY with valid JSON
- No explanations, no markdown, no analysis outside JSON
- Your response must start with {{ and end with }}
- No ```json``` wrappers or additional text

{{
    "validated_issues": [
        {{"issue_type": "...", "confidence": 0.9, "is_accurate": true}}
    ],
    "missing_issues": [
        {{"issue_type": "...", "confidence": {self.min_llm_confidence}, "reason": "..."}}
    ]
}}
"""
        
        try:
            response = self._call_llm(prompt)
            validation = self._parse_llm_response(response)
            
            # Update issues based on validation
            validated_issues = []
            
            # Process validated issues
            for v_issue in validation.get('validated_issues', []):
                if v_issue.get('is_accurate', True):
                    # Find and update original issue
                    found_match = False
                    for orig_issue in issues:
                        if orig_issue['issue_type'] == v_issue['issue_type']:
                            orig_issue['confidence'] *= v_issue.get('confidence', 1.0)
                            orig_issue['llm_validated'] = True
                            # PRESERVE original reference evidence from vector search
                            # Don't replace with LLM validation text
                            validated_issues.append(orig_issue)
                            found_match = True
                            break
                    
                    # If LLM identified a new issue not in semantic search results, add it
                    if not found_match and v_issue.get('confidence', 0.0) >= self.min_llm_confidence:
                        logger.info(f"LLM identified new issue not in semantic search: '{v_issue['issue_type']}' (confidence: {v_issue.get('confidence', 0.0):.3f})")
                        # Extract actual evidence from document instead of LLM analysis description
                        actual_evidence = self._extract_evidence_from_current_doc(text, v_issue['issue_type'])
                        validated_issues.append({
                            'issue_type': v_issue['issue_type'],
                            'confidence': v_issue.get('confidence', 0.0),
                            'evidence': actual_evidence or f"Document supports {v_issue['issue_type']} but no specific sentence found",
                            'reference_evidence': 'LLM validation only - no vector search match',
                            'source': 'llm_validation',
                            'llm_validated': True,
                            'validation_status': 'valid'
                        })
            
            # Process missing issues as potential better suggestions
            # When LLM suggests a more appropriate issue type, consider it if confidence is reasonable
            for missing in validation.get('missing_issues', []):
                missing_confidence = missing.get('confidence', 0.0)
                missing_issue_type = missing.get('issue_type', '')
                missing_reason = missing.get('reason', 'No reason given')
                
                logger.info(f"LLM suggested alternative issue: {missing_issue_type} (confidence: {missing_confidence}) - {missing_reason}")
                
                # If the missing issue has reasonable confidence and provides a better alternative
                if missing_confidence >= self.min_llm_confidence:
                    # Check if this is a better alternative than existing validated issues
                    should_add_as_alternative = True
                    
                    # Look for similar existing issues to potentially replace
                    for i, existing_issue in enumerate(validated_issues):
                        existing_confidence = existing_issue.get('confidence', 0.0)
                        
                        # If the missing issue has significantly higher confidence, consider replacement
                        if missing_confidence > existing_confidence * 1.5:
                            logger.info(f"Replacing lower confidence issue '{existing_issue['issue_type']}' ({existing_confidence:.3f}) with LLM suggestion '{missing_issue_type}' ({missing_confidence:.3f})")
                            
                            # Replace the existing issue with the better suggestion
                            # But preserve reference evidence if available
                            original_ref_evidence = existing_issue.get('reference_evidence', 'No reference available')
                            validated_issues[i] = {
                                'issue_type': missing_issue_type,
                                'confidence': missing_confidence,
                                'evidence': existing_issue.get('evidence', ''),  # Keep original evidence from document
                                'reference_evidence': original_ref_evidence,  # Keep original reference sentence
                                'source': 'llm_validation',
                                'replaced_issue': existing_issue['issue_type'],
                                'llm_replacement_reason': missing_reason
                            }
                            should_add_as_alternative = False
                            break
                    
                    # If no replacement occurred and confidence is high enough, add as additional issue
                    if should_add_as_alternative and missing_confidence >= 0.5:
                        # Extract actual evidence from document instead of LLM suggestion description
                        actual_evidence = self._extract_evidence_from_current_doc(text, missing_issue_type)
                        validated_issues.append({
                            'issue_type': missing_issue_type,
                            'confidence': missing_confidence,
                            'evidence': actual_evidence or f"Document supports {missing_issue_type} but no specific sentence found",
                            'reference_evidence': 'LLM validation only - no vector search match',
                            'source': 'llm_validation',
                            'llm_addition_reason': missing_reason
                        })
                        logger.info(f"Added LLM suggested issue '{missing_issue_type}' (confidence: {missing_confidence:.3f})")
            
            # Remap categories if issues changed
            if validated_issues:
                categories = self.mapper.map_issues_to_categories(validated_issues)
            
            return {
                'identified_issues': validated_issues,
                'categories': categories
            }
            
        except Exception as e:
            # Use the current provider name (which may have changed due to fallbacks)
            provider_name = getattr(self, 'current_llm_provider', 'Unknown')
            
            error_msg = f"LLM validation failed ({provider_name}): {str(e)}"
            logger.error(error_msg)
            logger.warning("Falling back to semantic search results without LLM validation")
            
            # FIXED: Fall back to semantic search results instead of returning empty
            # Mark original issues as not LLM validated and return them
            for issue in issues:
                issue['llm_validated'] = False
                issue['validation_status'] = 'failed'
            
            return {
                'identified_issues': issues,  # Use semantic search results as fallback
                'categories': categories,     # Use original category mapping
                'llm_error': True,
                'llm_error_message': error_msg,
                'llm_provider': provider_name,
                'raw_semantic_results': len(issues),  # Track what semantic search found
                'fallback_used': True,      # Flag to indicate fallback was used
                'error_details': {
                    'provider': provider_name,
                    'model': self.llm_model,
                    'error': str(e),
                    'timestamp': time.time()
                }
            }
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM API with hierarchical fallback retries.
        
        Args:
            prompt: The prompt to send
            
        Returns:
            LLM response text
        """
        # LOG THE EXACT PROMPT FOR ANALYSIS
        logger.info("="*80)
        logger.info("HYBRID RAG LLM PROMPT ANALYSIS")
        logger.info("="*80)
        logger.info(f"Model: {self.llm_model}")
        logger.info(f"Temperature: 0.1")
        logger.info(f"Max Tokens: 1000")
        logger.info("PROMPT CONTENT:")
        logger.info("-"*60)
        logger.info(prompt)
        logger.info("-"*60)
        
        # Try primary LLM client first
        primary_provider = self._get_provider_name_from_model(self.llm_model)
        try:
            response_content = self._call_single_llm(self.llm_client, self.llm_model, prompt)
            logger.info(f"✅ Primary LLM ({primary_provider}) succeeded")
            return response_content
            
        except Exception as primary_error:
            logger.warning(f"❌ Primary LLM ({primary_provider}) failed: {primary_error}")
            
            # Try fallback clients in order, but skip the same provider as primary
            fallback_order = ['anthropic', 'openai', 'gemini']
            
            # Filter out the current primary provider from fallback order
            primary_provider_key = self._get_provider_key_from_model(self.llm_model)
            fallback_order = [key for key in fallback_order if key != primary_provider_key]
            
            for fallback_key in fallback_order:
                if fallback_key in self.fallback_llm_clients:
                    fallback_info = self.fallback_llm_clients[fallback_key]
                    try:
                        logger.info(f"🔄 Trying fallback LLM: {fallback_info['name']}")
                        response_content = self._call_single_llm(
                            fallback_info['client'], 
                            fallback_info['model'], 
                            prompt
                        )
                        logger.info(f"✅ Fallback LLM ({fallback_info['name']}) succeeded")
                        
                        # Update the current provider for error reporting
                        self.current_llm_provider = fallback_info['name']
                        return response_content
                        
                    except Exception as fallback_error:
                        logger.warning(f"❌ Fallback LLM ({fallback_info['name']}) failed: {fallback_error}")
                        continue
            
            # All LLMs failed - build list of actually tried providers
            tried_providers = [primary_provider]
            for fallback_key in fallback_order:
                if fallback_key in self.fallback_llm_clients:
                    tried_providers.append(self.fallback_llm_clients[fallback_key]['name'])
            raise Exception(f"All available LLM providers failed: {', '.join(tried_providers)}. Last error: {primary_error}")
    
    def _call_single_llm(self, client, model: str, prompt: str) -> str:
        """Call a single LLM client with proper error handling."""
        if hasattr(client, 'chat'):  # OpenAI new client
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a contract classification expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=1000,
                response_format={"type": "json_object"},
                seed=42
            )
            response_content = response.choices[0].message.content
            
        elif hasattr(client, 'Completion') or hasattr(client, 'ChatCompletion'):  # Legacy OpenAI client
            response = client.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a contract classification expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=1000,
                seed=42
            )
            response_content = response.choices[0].message.content
            
        elif hasattr(client, 'messages'):  # Anthropic
            response = client.messages.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.0
            )
            response_content = response.content[0].text
            
        elif hasattr(client, 'generate_content'):  # Gemini
            if genai:
                generation_config = genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=1000
                )
                response = client.generate_content(prompt, generation_config=generation_config)
                response_content = response.text
            else:
                raise ValueError("Google Generative AI package not available")
            
        else:
            raise ValueError(f"Unknown LLM client type: {type(client)}")
        
        # LOG THE LLM RESPONSE FOR ANALYSIS
        logger.info("LLM RESPONSE:")
        logger.info("-"*60)
        logger.info(response_content)
        logger.info("-"*60)
        logger.info("END HYBRID RAG LLM ANALYSIS")
        logger.info("="*80)
        
        return response_content
    
    def _get_provider_name_from_model(self, model: str) -> str:
        """Get provider name from model string."""
        if 'gpt' in model.lower():
            return "OpenAI GPT"
        elif 'claude' in model.lower():
            return "Anthropic Claude"
        elif 'gemini' in model.lower():
            return "Google Gemini"
        else:
            return "Unknown"
    
    def _get_provider_key_from_model(self, model: str) -> str:
        """Get provider key for fallback client lookup from model string."""
        if 'gpt' in model.lower():
            return "openai"
        elif 'claude' in model.lower():
            return "anthropic"
        elif 'gemini' in model.lower():
            return "gemini"
        else:
            return "unknown"
    
    def _parse_llm_response(self, response: str) -> Dict:
        """
        Parse JSON response from LLM.
        
        Args:
            response: LLM response text
            
        Returns:
            Parsed JSON dictionary
        """
        try:
            # Clean response if needed
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]
            
            return json.loads(response)
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return {}
    
    def _validate_issues(self, issues: List[Dict]) -> List[Dict]:
        """
        Validate identified issues against allowlist.
        
        Args:
            issues: List of identified issues
            
        Returns:
            List of validated issues
        """
        validated = []
        
        for issue in issues:
            issue_type = issue.get('issue_type', '')
            validated_type, is_valid, confidence = self.validator.validate_issue_type(
                issue_type, auto_correct=True
            )
            
            if validated_type:
                issue_copy = issue.copy()
                issue_copy['issue_type'] = validated_type
                issue_copy['confidence'] = issue.get('confidence', 1.0) * confidence
                issue_copy['validation_status'] = 'valid' if is_valid else 'corrected'
                validated.append(issue_copy)
        
        return validated
    
    def _validate_categories(self, categories: List[Dict]) -> List[Dict]:
        """
        Validate categories against allowlist.
        
        Args:
            categories: List of categories
            
        Returns:
            List of validated categories
        """
        validated = []
        
        for cat in categories:
            category_name = cat.get('category', '')
            validated_cat, is_valid, confidence = self.validator.validate_category(
                category_name, auto_correct=True
            )
            
            if validated_cat:
                cat_copy = cat.copy()
                cat_copy['category'] = validated_cat
                cat_copy['confidence'] = cat.get('confidence', 1.0) * confidence
                cat_copy['validation_status'] = 'valid' if is_valid else 'corrected'
                validated.append(cat_copy)
        
        return validated
    
    def _extract_evidence_from_current_doc(self, document_text: str, issue_type: str) -> str:
        """
        Extract evidence from the current document for a given issue type.
        
        Args:
            document_text: The current document text being analyzed
            issue_type: The issue type to find evidence for
            
        Returns:
            Direct quote from current document that supports the issue type, or empty string
        """
        if not document_text or not issue_type:
            return ""
        
        # Convert to lowercase for matching
        text_lower = document_text.lower()
        issue_lower = issue_type.lower()
        
        # Define issue-specific keywords and patterns
        issue_patterns = {
            'design & drawings': ['design', 'drawing', 'plan', 'sketch', 'blueprint', 'submit', 'resubmit'],
            'payment': ['payment', 'bill', 'invoice', 'amount', 'sum', 'advance', 'retention'],
            'extension of time': ['extension', 'delay', 'time', 'schedule', 'completion', 'deadline'],
            'change of scope': ['scope', 'change', 'variation', 'additional', 'extra', 'modify'],
            'authority': ['authority', 'engineer', 'approval', 'clearance', 'permission'],
            'contractor': ['contractor', 'submit', 'deliver', 'complete', 'execute'],
            'safety': ['safety', 'accident', 'injury', 'precaution', 'protection'],
            'quality': ['quality', 'defect', 'standard', 'specification', 'test'],
            'material': ['material', 'supply', 'procurement', 'delivery'],
            'dispute': ['dispute', 'disagreement', 'claim', 'arbitration']
        }
        
        # Find relevant keywords for this issue type
        relevant_keywords = []
        for pattern_key, keywords in issue_patterns.items():
            if any(keyword in issue_lower for keyword in pattern_key.split()):
                relevant_keywords.extend(keywords)
        
        # If no specific patterns, use words from issue type itself
        if not relevant_keywords:
            relevant_keywords = [word.lower() for word in issue_type.split() if len(word) > 2]
        
        # Find sentences containing relevant keywords
        sentences = [s.strip() for s in document_text.split('.') if s.strip()]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            # Check if sentence contains multiple relevant keywords
            matches = sum(1 for keyword in relevant_keywords if keyword in sentence_lower)
            
            if matches >= 2:  # Require at least 2 keyword matches for confidence
                # Return the sentence, trimmed to reasonable length
                evidence = sentence.strip()
                if len(evidence) > 200:
                    evidence = evidence[:197] + "..."
                return evidence
        
        # Fallback: find any sentence with at least one strong keyword match
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for keyword in relevant_keywords[:3]:  # Check top 3 most relevant keywords
                if keyword in sentence_lower and len(sentence.strip()) > 20:
                    evidence = sentence.strip()
                    if len(evidence) > 200:
                        evidence = evidence[:197] + "..."
                    return evidence
        
        return ""  # No evidence found in current document
    
    def _apply_quality_filtering(self, issues: List[Dict]) -> List[Dict]:
        """
        Apply quality filtering and confidence decay to search results.
        
        Args:
            issues: List of identified issues
            
        Returns:
            Filtered and adjusted issues
        """
        if not issues:
            return issues
        
        # Generic evidence patterns that indicate poor matches
        generic_patterns = [
            'as per agreement', 'authority engineer', 'contractor shall',
            'schedule', 'clause', 'contract provision', 'the authority',
            'as per contract', 'work contract', 'agreement condition'
        ]
        
        filtered_issues = []
        
        for i, issue in enumerate(issues):
            # Apply confidence decay based on ranking
            decay_factor = 1.0 - (i * self.confidence_decay)
            adjusted_confidence = issue['confidence'] * decay_factor
            
            # Filter out issues with too generic evidence
            evidence_lower = issue.get('evidence', '').lower()
            generic_score = sum(1 for pattern in generic_patterns 
                              if pattern in evidence_lower)
            
            # More lenient penalty for generic evidence to improve recall
            if generic_score > 3:  # Increased threshold
                adjusted_confidence *= 0.8  # Less penalty
            
            # More lenient filtering for "Authority Engineer" 
            if (issue['issue_type'] == 'Authority Engineer' and 
                issue['search_count'] > 30 and  # Higher threshold
                generic_score > 2):  # Higher threshold
                adjusted_confidence *= 0.7  # Less penalty
            
            # Extremely low threshold for keeping issues (maximize recall)
            if adjusted_confidence >= (self.similarity_threshold * 0.3):  # Was 0.6, now 0.3
                issue_copy = issue.copy()
                issue_copy['confidence'] = adjusted_confidence
                issue_copy['quality_filtered'] = True
                filtered_issues.append(issue_copy)
        
        logger.info(f"Quality filtering: {len(issues)} → {len(filtered_issues)} issues")
        return filtered_issues
    
    def _apply_confidence_filtering(self, issues: List[Dict]) -> List[Dict]:
        """
        Apply confidence-based filtering when there's divergence between true and false positives.
        
        When confidence divergence is detected, keep only the top percentage of predictions
        to improve precision while maintaining reasonable recall.
        
        Args:
            issues: List of identified issues with confidence scores
            
        Returns:
            Filtered list of issues when divergence detected, otherwise original list
        """
        if not issues or len(issues) <= 1:
            return issues
        
        # Get configuration parameters
        divergence_threshold = self.config.get('confidence_divergence_threshold', 0.15)
        top_percentage = self.config.get('top_percentage_filter', 0.20)
        
        # Sort issues by confidence in descending order
        sorted_issues = sorted(issues, key=lambda x: x.get('confidence', 0), reverse=True)
        
        # Calculate statistics
        confidences = [issue.get('confidence', 0) for issue in sorted_issues]
        
        if len(confidences) < 2:
            return issues
        
        # Calculate potential true positive and false positive confidence levels
        # Assume top issues are likely true positives, bottom issues likely false positives
        top_20_percent = max(1, int(len(confidences) * 0.2))
        bottom_20_percent = max(1, int(len(confidences) * 0.2))
        
        avg_top_confidence = sum(confidences[:top_20_percent]) / top_20_percent
        avg_bottom_confidence = sum(confidences[-bottom_20_percent:]) / bottom_20_percent
        
        confidence_divergence = avg_top_confidence - avg_bottom_confidence
        
        logger.info(f"Confidence analysis: Top 20% avg: {avg_top_confidence:.3f}, "
                   f"Bottom 20% avg: {avg_bottom_confidence:.3f}, "
                   f"Divergence: {confidence_divergence:.3f}")
        
        # Apply filtering only if significant divergence detected
        if confidence_divergence >= divergence_threshold:
            # Calculate how many issues to keep (top percentage)
            keep_count = max(1, int(len(sorted_issues) * top_percentage))
            filtered_issues = sorted_issues[:keep_count]
            
            logger.info(f"Confidence filtering applied: {len(issues)} → {len(filtered_issues)} issues "
                       f"(divergence: {confidence_divergence:.3f} >= {divergence_threshold:.3f})")
            
            # Add filtering metadata
            for issue in filtered_issues:
                issue['confidence_filtered'] = True
                issue['divergence_detected'] = confidence_divergence
            
            return filtered_issues
        else:
            logger.info(f"No confidence filtering needed (divergence: {confidence_divergence:.3f} < {divergence_threshold:.3f})")
            return issues
    
    def _apply_source_priority(self, issues: List[Dict]) -> List[Dict]:
        """
        Apply source priority to favor LLM validation results over semantic search.
        
        When multiple issues exist, LLM validation results are given priority over 
        semantic search results due to their contextual accuracy.
        
        Args:
            issues: List of identified issues with different sources
            
        Returns:
            Issues sorted by source priority and confidence
        """
        if not issues:
            return issues
        
        # Separate issues by source
        llm_issues = [issue for issue in issues if issue.get('source') == 'llm_validation']
        semantic_issues = [issue for issue in issues if issue.get('source') == 'semantic_search']
        other_issues = [issue for issue in issues if issue.get('source') not in ['llm_validation', 'semantic_search']]
        
        # Sort each group by confidence
        llm_issues.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        semantic_issues.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        other_issues.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        # Combine with priority: LLM validation first, then semantic search, then others
        prioritized_issues = llm_issues + semantic_issues + other_issues
        
        # If we have both LLM and semantic issues, prefer LLM when confidence is comparable
        if llm_issues and semantic_issues:
            logger.info(f"Source priority applied: {len(llm_issues)} LLM issues prioritized over {len(semantic_issues)} semantic issues")
        
        return prioritized_issues
    
    def update_index(self, new_documents: List[Dict]):
        """
        Add new documents to the index.
        
        Args:
            new_documents: List of documents with text and metadata
        """
        texts = []
        metadata = []
        
        for doc in new_documents:
            text = doc.get('text', '')
            if text:
                texts.append(self.preprocessor.clean_text(text))
                metadata.append({
                    'issue_type': doc.get('issue_type', ''),
                    'category': doc.get('category', ''),
                    'reference_sentence': doc.get('reference_sentence', ''),
                    'source_file': doc.get('source_file', '')
                })
        
        if texts:
            self.embeddings_manager.add_to_index(texts, metadata)
            logger.info(f"Added {len(texts)} documents to index")
    
    def get_index_stats(self) -> Dict:
        """
        Get statistics about the current index.
        
        Returns:
            Index statistics
        """
        return self.embeddings_manager.get_index_stats()
    
    def __repr__(self):
        index_size = self.embeddings_manager.get_index_stats().get('total_vectors', 0)
        return f"HybridRAGClassifier(index_size={index_size}, llm={self.llm_model})"