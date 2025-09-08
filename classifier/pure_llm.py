"""
Pure LLM Classifier Module
Direct classification using GPT-4 or Claude without training
"""

import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple
import logging
from openai import OpenAI
import anthropic
import google.generativeai as genai
from pathlib import Path

from .issue_mapper import IssueCategoryMapper
from .validation import ValidationEngine
from .data_sufficiency import DataSufficiencyAnalyzer
from .preprocessing import TextPreprocessor
from .pdf_extractor import PDFExtractor

logger = logging.getLogger(__name__)


class PureLLMClassifier:
    """
    Classifies contract correspondence using pure LLM approach (GPT-4/Claude).
    Implements two-phase classification: Issue identification → Category mapping.
    """
    
    def __init__(self, 
                 config: Dict,
                 issue_mapper: IssueCategoryMapper,
                 validator: ValidationEngine,
                 data_analyzer: DataSufficiencyAnalyzer):
        """
        Initialize Pure LLM Classifier.
        
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
        
        # Get model parameters first
        self.model = config.get('model', 'gpt-4-turbo')
        self.max_tokens = config.get('max_tokens', 4096)
        self.temperature = config.get('temperature', 0.1)
        self.validate_mapping = config.get('validate_mapping', True)
        
        # Initialize preprocessor and PDF extractor
        self.preprocessor = TextPreprocessor()
        self.pdf_extractor = PDFExtractor()
        
        # Initialize hierarchical LLM clients
        self.llm_clients = self._init_hierarchical_llm_clients()
        
        logger.info(f"PureLLMClassifier initialized with hierarchical LLM support")
    
    def _init_hierarchical_llm_clients(self):
        """
        Initialize hierarchical LLM clients in order: Gemini -> OpenAI -> Anthropic
        
        Returns:
            Dictionary of available LLM clients
        """
        clients = {}
        
        # 1. Gemini client (first priority)
        try:
            google_api_key = os.getenv('GOOGLE_API_KEY')
            if google_api_key:
                genai.configure(api_key=google_api_key)
                clients['gemini'] = genai.GenerativeModel('gemini-1.5-flash')
                logger.info("✅ Gemini client initialized")
            else:
                logger.warning("⚠️  Google API key not found - Gemini unavailable")
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize Gemini client: {e}")
        
        # 2. OpenAI client (second priority)
        try:
            openai_api_key = os.getenv('OPENAI_API_KEY') or self.config.get('api_key')
            if openai_api_key:
                org_id = os.getenv('OPENAI_ORG_ID')
                if org_id:
                    clients['openai'] = OpenAI(api_key=openai_api_key, organization=org_id)
                else:
                    clients['openai'] = OpenAI(api_key=openai_api_key)
                logger.info("✅ OpenAI client initialized")
            else:
                logger.warning("⚠️  OpenAI API key not found - OpenAI unavailable")
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize OpenAI client: {e}")
        
        # 3. Anthropic client (third priority)
        try:
            anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
            if anthropic_api_key:
                clients['anthropic'] = anthropic.Anthropic(api_key=anthropic_api_key)
                logger.info("✅ Anthropic client initialized")
            else:
                logger.warning("⚠️  Anthropic API key not found - Anthropic unavailable")
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize Anthropic client: {e}")
        
        if not clients:
            raise ValueError("No LLM clients could be initialized. Please check your API keys.")
        
        logger.info(f"🚀 Hierarchical LLM clients available: {list(clients.keys())}")
        return clients
    
    def _call_llm_hierarchical(self, messages: List[Dict], max_tokens: int = None) -> Tuple[str, str]:
        """
        Call LLMs in hierarchical order: Gemini -> OpenAI -> Anthropic
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens for response
            
        Returns:
            Tuple of (response_text, provider_used)
        """
        max_tokens = max_tokens or self.max_tokens
        providers_order = ['gemini', 'openai', 'anthropic']
        
        for provider in providers_order:
            if provider not in self.llm_clients:
                logger.debug(f"⏭️  {provider.upper()} not available, trying next...")
                continue
                
            try:
                logger.debug(f"🔄 Attempting {provider.upper()} call...")
                
                if provider == 'gemini':
                    response = self._call_gemini(messages, max_tokens)
                elif provider == 'openai':
                    response = self._call_openai(messages, max_tokens)
                elif provider == 'anthropic':
                    response = self._call_anthropic(messages, max_tokens)
                
                logger.info(f"✅ {provider.upper()} call successful")
                return response, provider
                
            except Exception as e:
                logger.warning(f"❌ {provider.upper()} call failed: {e}")
                if provider == providers_order[-1]:  # Last provider
                    logger.error("💥 All LLM providers failed")
                    raise e
                else:
                    logger.info(f"⏭️  Falling back to next provider...")
                    continue
        
        raise RuntimeError("All LLM providers exhausted")
    
    def _call_gemini(self, messages: List[Dict], max_tokens: int) -> str:
        """Call Gemini API"""
        # Convert messages to Gemini format
        prompt_parts = []
        for msg in messages:
            if msg['role'] == 'system':
                prompt_parts.append(f"System: {msg['content']}")
            elif msg['role'] == 'user':
                prompt_parts.append(f"Human: {msg['content']}")
        
        full_prompt = "\n\n".join(prompt_parts)
        
        model = self.llm_clients['gemini']
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=self.temperature,
            )
        )
        
        return response.text
    
    def _call_openai(self, messages: List[Dict], max_tokens: int) -> str:
        """Call OpenAI API"""
        client = self.llm_clients['openai']
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=self.temperature
        )
        
        return response.choices[0].message.content
    
    def _call_anthropic(self, messages: List[Dict], max_tokens: int) -> str:
        """Call Anthropic API"""
        client = self.llm_clients['anthropic']
        
        # Convert messages to Anthropic format
        system_msg = None
        anthropic_messages = []
        
        for msg in messages:
            if msg['role'] == 'system':
                system_msg = msg['content']
            else:
                anthropic_messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
        
        kwargs = {
            'model': 'claude-sonnet-4-20250514',
            'max_tokens': max_tokens,
            'temperature': self.temperature,
            'messages': anthropic_messages
        }
        
        if system_msg:
            kwargs['system'] = system_msg
        
        response = client.messages.create(**kwargs)
        return response.content[0].text
    
    def classify(self, document: str, is_file_path: bool = False) -> Dict:
        """
        Classify a document using pure LLM approach.
        
        Args:
            document: Document text or path to document file
            is_file_path: Whether document is a file path
            
        Returns:
            Classification results with validation and sufficiency warnings
        """
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
        
        # Phase 1: Identify issue types with constrained prompt
        identified_issues, llm_provider_used = self._identify_issues(processed_text)
        
        # Phase 2: Validate identified issues
        validated_issues = self._validate_issues(identified_issues)
        
        # Phase 3: Map issues to categories
        categories = self.mapper.map_issues_to_categories(validated_issues)
        
        # Phase 4: Validate categories
        validated_categories = self._validate_categories(categories)
        
        # Phase 5: Optional LLM validation of mapping
        if self.validate_mapping and validated_categories:
            validated_categories = self._llm_validate_mapping(
                processed_text, validated_categories
            )
        
        # Phase 6: Apply data sufficiency adjustments
        result = {
            'identified_issues': validated_issues,
            'categories': validated_categories,
            'classification_path': 'issue_identification → validation → category_mapping',
            'extraction_method': extraction_method,
            'model_used': self.model,
            'llm_provider_used': llm_provider_used,
            'processing_time': time.time() - start_time
        }
        
        # Apply confidence adjustments based on data sufficiency
        result = self.data_analyzer.apply_confidence_adjustments(result)
        
        # Add validation report
        result['validation_report'] = self._generate_validation_report(
            identified_issues, validated_issues, categories, validated_categories
        )
        
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
    
    def _identify_issues(self, text: str) -> Tuple[List[Dict], str]:
        """
        Identify issue types in the document using LLM.
        
        Args:
            text: Document text
            
        Returns:
            Tuple of (List of identified issues with confidence scores, provider_used)
        """
        # Get constrained prompt to prevent hallucinations
        constraints = self.validator.create_constrained_prompt('issues')
        
        # Chunk text if too long
        text_chunks = self.preprocessor.chunk_text(text, max_chunk_size=3000)
        
        # Use first chunk or combine key sentences from multiple chunks
        if len(text_chunks) > 1:
            # Extract key sentences from each chunk
            key_sentences = []
            for chunk in text_chunks[:3]:  # Use first 3 chunks
                sentences = self.preprocessor.extract_key_sentences(chunk, max_sentences=5)
                key_sentences.extend(sentences)
            text_for_analysis = ' '.join(key_sentences)
        else:
            text_for_analysis = text_chunks[0] if text_chunks else text[:3000]
        
        prompt = f"""
{constraints}

Task: Analyze this contract correspondence and identify ALL issue types being discussed.
For each issue, provide your confidence level and supporting evidence from the document.

Document:
{text_for_analysis}

Return your analysis in the following JSON format:
{{
    "issues": [
        {{
            "issue_type": "exact issue type from the list",
            "confidence": 0.95,
            "evidence": "quote from document supporting this classification"
        }}
    ]
}}

IMPORTANT: Only use issue types from the provided list. Be thorough and identify ALL relevant issues.
"""
        
        try:
            messages = [
                {"role": "system", "content": "You are a contract classification expert."},
                {"role": "user", "content": prompt}
            ]
            response, provider_used = self._call_llm_hierarchical(messages)
            logger.debug(f"Raw LLM response from {provider_used}: {response[:200]}...")
            parsed_response = self._parse_llm_response(response)
            logger.debug(f"Parsed response type: {type(parsed_response)}")
            
            # Extract issues from parsed response
            if isinstance(parsed_response, dict) and 'issues' in parsed_response:
                issues = parsed_response['issues']
            elif isinstance(parsed_response, list):
                issues = parsed_response
            else:
                logger.warning(f"Unexpected LLM response format: {type(parsed_response)}, content: {str(parsed_response)[:100]}")
                issues = []
            
            logger.info(f"Identified {len(issues)} issues in document using {provider_used.upper()}")
            return issues, provider_used
            
        except Exception as e:
            logger.error(f"Error identifying issues: {e}")
            return [], "none"
    
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
                
                if not is_valid:
                    issue_copy['original_issue_type'] = issue_type
                    logger.warning(f"Corrected issue type: {issue_type} → {validated_type}")
                
                validated.append(issue_copy)
            else:
                logger.warning(f"Rejected invalid issue type: {issue_type}")
        
        return validated
    
    def _validate_categories(self, categories: List[Dict]) -> List[Dict]:
        """
        Validate categories against allowlist.
        
        Args:
            categories: List of categories from mapping
            
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
                
                if not is_valid:
                    cat_copy['original_category'] = category_name
                    logger.warning(f"Corrected category: {category_name} → {validated_cat}")
                
                validated.append(cat_copy)
            else:
                logger.warning(f"Rejected invalid category: {category_name}")
        
        return validated
    
    def _llm_validate_mapping(self, text: str, categories: List[Dict]) -> List[Dict]:
        """
        Use LLM to validate and adjust category mappings.
        
        Args:
            text: Document text
            categories: Mapped categories
            
        Returns:
            Validated categories with adjusted confidence
        """
        # Create summary of current classification
        category_list = [c['category'] for c in categories[:5]]  # Top 5 categories
        
        prompt = f"""
Review this classification for accuracy.

Document excerpt:
{text[:1500]}

Assigned categories:
{json.dumps(category_list, indent=2)}

Are these categories appropriate for this document? 
Rate the accuracy from 0.0 to 1.0 for each category.

Return JSON:
{{
    "validations": [
        {{"category": "category_name", "accuracy": 0.9}}
    ]
}}
"""
        
        try:
            messages = [
                {"role": "system", "content": "You are a contract classification expert."},
                {"role": "user", "content": prompt}
            ]
            response, provider_used = self._call_llm_hierarchical(messages, max_tokens=500)
            validations = self._parse_llm_response(response)
            
            # Adjust confidence based on validation
            validation_dict = {
                v['category']: v['accuracy'] 
                for v in validations.get('validations', [])
            }
            
            for cat in categories:
                if cat['category'] in validation_dict:
                    accuracy = validation_dict[cat['category']]
                    cat['confidence'] *= accuracy
                    cat['llm_validation_score'] = accuracy
            
        except Exception as e:
            logger.warning(f"LLM validation failed: {e}")
        
        return categories
    
    
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
            
            # Find JSON block within the response (handles explanatory text before JSON)
            json_content = ""
            
            # Look for ```json block
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_content = json_match.group(1).strip()
            else:
                # Look for generic ``` block that might contain JSON
                code_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
                if code_match:
                    potential_json = code_match.group(1).strip()
                    # Check if it looks like JSON (starts with { or [)
                    if potential_json.startswith('{') or potential_json.startswith('['):
                        json_content = potential_json
                else:
                    # Try to find JSON-like content (starts with { and ends with })
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        json_content = json_match.group().strip()
                    else:
                        # Last resort: try the entire response if it looks like JSON
                        if response.strip().startswith('{') and response.strip().endswith('}'):
                            json_content = response.strip()
            
            if json_content:
                logger.debug(f"Extracted JSON content: {json_content[:100]}...")
                # Sanitize JSON content to fix common LLM formatting issues
                json_content = self._sanitize_json(json_content)
                return json.loads(json_content)
            else:
                logger.warning(f"No JSON content found in response: {response[:200]}...")
                return {}
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Response was: {response}")
            logger.debug(f"Extracted JSON was: {json_content}")
            return {}
    
    def _sanitize_json(self, json_str: str) -> str:
        """
        Sanitize JSON string to fix common LLM formatting issues.
        
        Args:
            json_str: Raw JSON string from LLM
            
        Returns:
            Cleaned JSON string
        """
        # Fix common JSON formatting issues from LLMs
        
        # Fix: "text" and "more text" -> "text and more text"
        # This handles cases where LLM puts quotes around parts of evidence
        json_str = re.sub(r'"([^"]*?)" and "([^"]*?)"', r'"\1 and \2"', json_str)
        
        # Fix: trailing commas before closing brackets
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Fix: single quotes instead of double quotes (less common but possible)
        json_str = re.sub(r"'([^']*?)':", r'"\1":', json_str)
        
        return json_str
    
    def _generate_validation_report(self, 
                                   original_issues: List[Dict],
                                   validated_issues: List[Dict],
                                   original_categories: List[Dict],
                                   validated_categories: List[Dict]) -> Dict:
        """
        Generate validation report.
        
        Args:
            original_issues: Original identified issues
            validated_issues: Validated issues
            original_categories: Original mapped categories
            validated_categories: Validated categories
            
        Returns:
            Validation report dictionary
        """
        report = {
            'hallucinations_detected': False,
            'corrections_made': [],
            'rejections': [],
            'validation_status': 'clean'
        }
        
        # Check for corrections in issues
        for validated in validated_issues:
            if validated.get('validation_status') == 'corrected':
                report['hallucinations_detected'] = True
                report['corrections_made'].append({
                    'type': 'issue',
                    'original': validated.get('original_issue_type'),
                    'corrected': validated['issue_type']
                })
        
        # Check for rejections in issues
        original_types = {i.get('issue_type') for i in original_issues}
        validated_types = {i['issue_type'] for i in validated_issues}
        rejected_types = original_types - validated_types
        
        for rejected in rejected_types:
            report['rejections'].append({
                'type': 'issue',
                'value': rejected,
                'reason': 'Not in allowlist'
            })
        
        # Update status
        if report['corrections_made']:
            report['validation_status'] = 'corrected'
        if report['rejections']:
            report['validation_status'] = 'rejected_items'
            report['hallucinations_detected'] = True
        
        return report
    
    def batch_classify(self, documents: List[str]) -> List[Dict]:
        """
        Classify multiple documents in batch.
        
        Args:
            documents: List of document texts or paths
            
        Returns:
            List of classification results
        """
        results = []
        
        for i, doc in enumerate(documents):
            logger.info(f"Processing document {i+1}/{len(documents)}")
            try:
                result = self.classify(doc)
                result['document_index'] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing document {i}: {e}")
                results.append({
                    'document_index': i,
                    'status': 'error',
                    'error': str(e)
                })
        
        return results
    
    def __repr__(self):
        return f"PureLLMClassifier(model={self.model})"