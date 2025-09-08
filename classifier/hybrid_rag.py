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

from .issue_mapper import IssueCategoryMapper
from .validation import ValidationEngine
from .data_sufficiency import DataSufficiencyAnalyzer
from .preprocessing import TextPreprocessor
from .pdf_extractor import PDFExtractor
from .embeddings import EmbeddingsManager

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
        
        # Configuration parameters
        self.llm_model = config.get('llm_model', 'gpt-3.5-turbo')
        
        # Initialize embeddings manager
        embedding_model = config.get('embedding_model', 'all-mpnet-base-v2')
        self.embeddings_manager = EmbeddingsManager(model_name=embedding_model)
        
        # Initialize LLM client for validation
        self.llm_client = self._init_llm_client()
        self.top_k = config.get('top_k', 15)
        self.window_size = config.get('window_size', 3)
        self.overlap = config.get('overlap', 1)
        self.similarity_threshold = config.get('similarity_threshold', 0.3)
        
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
                api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("Anthropic API key required for Claude models")
            return anthropic.Anthropic(api_key=api_key)
        
        else:
            # Default to OpenAI
            return OpenAI(api_key=api_key) if api_key else None
    
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
        
        # Preprocess text
        processed_text = self.preprocessor.normalize_document(document_text)
        
        # Phase 1: Semantic search to find similar issues
        similar_issues = self._semantic_search_issues(processed_text)
        
        # Phase 2: Map identified issues to categories
        categories = self.mapper.map_issues_to_categories(similar_issues)
        
        # Phase 3: LLM validation and refinement
        if self.llm_client:
            refined_results = self._llm_validation(processed_text, similar_issues, categories)
        else:
            refined_results = {
                'identified_issues': similar_issues,
                'categories': categories
            }
        
        # Phase 4: Apply validation
        validated_issues = self._validate_issues(refined_results.get('identified_issues', []))
        validated_categories = self._validate_categories(refined_results.get('categories', []))
        
        # Phase 5: Apply data sufficiency adjustments
        result = {
            'identified_issues': validated_issues,
            'categories': validated_categories,
            'classification_path': 'semantic_search → issue_aggregation → llm_validation → category_mapping',
            'extraction_method': extraction_method,
            'search_results_used': len(similar_issues),
            'processing_time': time.time() - start_time
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
        # Create sliding windows for better context matching
        windows = self.preprocessor.create_sliding_windows(
            text, 
            window_size=self.window_size, 
            overlap=self.overlap
        )
        
        # If no windows, use the whole text
        if not windows:
            windows = [(text, 0, 0)]
        
        # Search for each window and aggregate results
        all_matches = []
        issue_scores = defaultdict(lambda: {'confidence': 0, 'evidence': [], 'count': 0})
        
        for window_text, start_idx, end_idx in windows:
            # Search for similar documents
            search_results = self.embeddings_manager.search(
                window_text, 
                k=self.top_k,
                threshold=self.similarity_threshold
            )
            
            # Aggregate issue types from search results
            for result in search_results:
                metadata = result.get('metadata', {})
                issue_type = metadata.get('issue_type', '')
                
                if issue_type:
                    # Update confidence (keep highest)
                    if result['similarity'] > issue_scores[issue_type]['confidence']:
                        issue_scores[issue_type]['confidence'] = result['similarity']
                    
                    # Collect evidence
                    reference = metadata.get('reference_sentence', '')
                    if reference and reference not in issue_scores[issue_type]['evidence']:
                        issue_scores[issue_type]['evidence'].append(reference)
                    
                    # Count occurrences
                    issue_scores[issue_type]['count'] += 1
        
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
                'evidence': '; '.join(scores['evidence'][:3]),  # Top 3 evidence
                'search_count': scores['count'],
                'source': 'semantic_search'
            })
        
        # Sort by confidence
        identified_issues.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Keep top issues (to avoid noise)
        max_issues = self.config.get('max_issues', 10)
        identified_issues = identified_issues[:max_issues]
        
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
        if not self.llm_client or not issues:
            return {'identified_issues': issues, 'categories': categories}
        
        # Prepare context for LLM
        issue_summary = [
            f"- {i['issue_type']} (confidence: {i['confidence']:.2f})"
            for i in issues[:5]
        ]
        
        category_summary = [
            f"- {c['category']}"
            for c in categories[:5]
        ]
        
        # Get validation constraints
        constraints = self.validator.create_constrained_prompt('issues')
        
        prompt = f"""
Review and validate these classification results from semantic search.

Document excerpt:
{text[:1500]}

Identified issues (from similar documents):
{chr(10).join(issue_summary)}

Mapped categories:
{chr(10).join(category_summary)}

{constraints}

Please:
1. Confirm if the identified issues are accurate for this document
2. Suggest any missing issues from the valid list
3. Rate confidence (0-1) for each issue

Return JSON:
{{
    "validated_issues": [
        {{"issue_type": "...", "confidence": 0.9, "is_accurate": true}}
    ],
    "missing_issues": [
        {{"issue_type": "...", "confidence": 0.8, "reason": "..."}}
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
                    for orig_issue in issues:
                        if orig_issue['issue_type'] == v_issue['issue_type']:
                            orig_issue['confidence'] *= v_issue.get('confidence', 1.0)
                            orig_issue['llm_validated'] = True
                            validated_issues.append(orig_issue)
                            break
            
            # Add missing issues identified by LLM
            for missing in validation.get('missing_issues', []):
                validated_issues.append({
                    'issue_type': missing['issue_type'],
                    'confidence': missing.get('confidence', 0.7),
                    'evidence': missing.get('reason', 'Identified by LLM validation'),
                    'source': 'llm_validation'
                })
            
            # Remap categories if issues changed
            if validated_issues:
                categories = self.mapper.map_issues_to_categories(validated_issues)
            
            return {
                'identified_issues': validated_issues,
                'categories': categories
            }
            
        except Exception as e:
            logger.warning(f"LLM validation failed: {e}")
            return {'identified_issues': issues, 'categories': categories}
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM API with the given prompt.
        
        Args:
            prompt: The prompt to send
            
        Returns:
            LLM response text
        """
        try:
            if isinstance(self.llm_client, OpenAI):
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": "You are a contract classification expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1000,
                    response_format={"type": "json_object"}
                )
                return response.choices[0].message.content
            
            elif isinstance(self.llm_client, anthropic.Anthropic):
                response = self.llm_client.messages.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.1
                )
                return response.content[0].text
            
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise
    
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