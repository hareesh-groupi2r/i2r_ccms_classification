"""
LLM Service
Modular service for structured data extraction and content classification using LLMs
"""

import json
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import google.generativeai as genai

from .interfaces import ILLMService, ProcessingResult, ProcessingStatus
from .configuration_service import get_config_service


class LLMService(ILLMService):
    """Service for LLM-based data extraction and classification"""
    
    def __init__(self, config_service=None):
        self.config_service = config_service or get_config_service()
        self.config = self.config_service.get_service_config("llm")
        
        # LLM configuration
        self.api_key = self.config.get("api_key")
        self.model_name = self.config.get("model_name", "gemini-2.0-flash")
        self.temperature = self.config.get("temperature", 0.1)
        self.max_tokens = self.config.get("max_tokens", 4000)
        self.timeout = self.config.get("timeout", 60)
        self.retry_count = self.config.get("retry_count", 2)
        
        # Initialize LLM client
        self.llm_available = False
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
                self.llm_available = True
            except Exception as e:
                print(f"Warning: Could not initialize LLM client: {e}")
        else:
            print("Warning: GOOGLE_API_KEY not configured. LLM service unavailable.")
    
    def extract_structured_data(self, text_content: str, extraction_schema: Dict[str, str], **kwargs) -> ProcessingResult:
        """
        Extract structured data from text using LLM
        
        Args:
            text_content: Text to analyze
            extraction_schema: Dict of field_name -> description
            **kwargs: LLM options
                - output_format: 'json' or 'text'
                - additional_instructions: Extra instructions for the LLM
                - max_retries: Override retry count
        
        Returns:
            ProcessingResult with structured data dict
        """
        try:
            if not self.llm_available:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    error_message="LLM service not available. Check API key configuration."
                )
            
            if not text_content or not text_content.strip():
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    error_message="No text content provided for extraction"
                )
            
            if not extraction_schema:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    error_message="No extraction schema provided"
                )
            
            # Preprocess text to reduce tokens
            processed_text = self._preprocess_text_for_efficiency(text_content)
            
            # Build extraction prompt
            prompt = self._build_extraction_prompt(processed_text, extraction_schema, **kwargs)
            
            # Execute LLM request with retries
            max_retries = kwargs.get("max_retries", self.retry_count)
            llm_result = self._execute_llm_request(prompt, max_retries)
            
            if llm_result.status == ProcessingStatus.ERROR:
                return llm_result
            
            # Parse and validate response
            parsed_result = self._parse_extraction_response(llm_result.data, extraction_schema)
            
            if parsed_result.status == ProcessingStatus.SUCCESS:
                parsed_result.metadata = parsed_result.metadata or {}
                parsed_result.metadata.update({
                    "extraction_schema": extraction_schema,
                    "text_length": len(text_content),
                    "model_used": self.model_name,
                    "fields_extracted": len(parsed_result.data) if isinstance(parsed_result.data, dict) else 0
                })
            
            return parsed_result
            
        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message=f"Error in structured data extraction: {str(e)}"
            )
    
    def classify_content(self, text_content: str, classification_options: List[str], **kwargs) -> ProcessingResult:
        """
        Classify content into predefined categories
        
        Args:
            text_content: Text to classify
            classification_options: List of possible categories
            **kwargs: Classification options
                - confidence_threshold: Minimum confidence for classification
                - additional_context: Extra context for classification
                - return_all_scores: Return confidence scores for all options
        
        Returns:
            ProcessingResult with classification result
        """
        try:
            if not self.llm_available:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    error_message="LLM service not available. Check API key configuration."
                )
            
            if not text_content or not text_content.strip():
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    error_message="No text content provided for classification"
                )
            
            if not classification_options:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    error_message="No classification options provided"
                )
            
            # Build classification prompt
            prompt = self._build_classification_prompt(text_content, classification_options, **kwargs)
            
            # Execute LLM request
            max_retries = kwargs.get("max_retries", self.retry_count)
            llm_result = self._execute_llm_request(prompt, max_retries)
            
            if llm_result.status == ProcessingStatus.ERROR:
                return llm_result
            
            # Parse classification response
            classification_result = self._parse_classification_response(
                llm_result.data, 
                classification_options, 
                **kwargs
            )
            
            if classification_result.status == ProcessingStatus.SUCCESS:
                classification_result.metadata = classification_result.metadata or {}
                classification_result.metadata.update({
                    "classification_options": classification_options,
                    "text_length": len(text_content),
                    "model_used": self.model_name
                })
            
            return classification_result
            
        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message=f"Error in content classification: {str(e)}"
            )
    
    def _build_extraction_prompt(self, text_content: str, extraction_schema: Dict[str, str], **kwargs) -> str:
        """Build prompt for structured data extraction"""
        output_format = kwargs.get("output_format", "json")
        additional_instructions = kwargs.get("additional_instructions", "")
        
        prompt = "From the provided text, extract the following information:\n\n"
        
        for field_name, description in extraction_schema.items():
            prompt += f"- {field_name}: {description}\n"
        
        prompt += f"\nFormat the output as a {output_format.upper()} object with keys matching the field names exactly.\n"
        
        if additional_instructions:
            prompt += f"\nAdditional instructions: {additional_instructions}\n"
        
        prompt += "\nIf a field cannot be found in the text, use null or 'Not specified'.\n"
        prompt += "Be precise and extract information exactly as it appears in the document.\n\n"
        prompt += f"Text to analyze:\n\n{text_content}"
        
        return prompt
    
    def _build_classification_prompt(self, text_content: str, classification_options: List[str], **kwargs) -> str:
        """Build prompt for content classification"""
        additional_context = kwargs.get("additional_context", "")
        return_all_scores = kwargs.get("return_all_scores", False)
        
        options_str = ", ".join(classification_options)
        
        prompt = f"Classify the following text into one of these categories: {options_str}\n\n"
        
        if additional_context:
            prompt += f"Additional context: {additional_context}\n\n"
        
        if return_all_scores:
            prompt += "Provide confidence scores (0-1) for each category in JSON format.\n"
            prompt += f"Example: {{'category': 'selected_category', 'confidence': 0.85, 'all_scores': {{'option1': 0.85, 'option2': 0.10, 'option3': 0.05}}}}\n\n"
        else:
            prompt += "Return the most appropriate category and a confidence score (0-1).\n"
            prompt += "Format: {'category': 'selected_category', 'confidence': 0.85}\n\n"
        
        prompt += f"Text to classify:\n\n{text_content}"
        
        return prompt
    
    def _execute_llm_request(self, prompt: str, max_retries: int) -> ProcessingResult:
        """Execute LLM request with retries"""
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                # Configure generation parameters
                generation_config = genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens
                )
                
                # Generate response
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                
                if not response.text:
                    raise Exception("Empty response from LLM")
                
                return ProcessingResult(
                    status=ProcessingStatus.SUCCESS,
                    data=response.text,
                    metadata={
                        "attempt": attempt + 1,
                        "prompt_length": len(prompt),
                        "response_length": len(response.text)
                    }
                )
                
            except Exception as e:
                last_error = str(e)
                error_str = str(e).lower()
                
                # Check for specific error types that shouldn't be retried
                if "quota" in error_str or "429" in error_str or "rate limit" in error_str:
                    # Quota/rate limit error - don't retry, return immediately
                    quota_error_msg = "API quota limit exceeded. Please check your Google AI API usage limits and billing details."
                    if "free tier" in error_str:
                        quota_error_msg += " You are using the free tier which has limited daily requests."
                    
                    return ProcessingResult(
                        status=ProcessingStatus.ERROR,
                        error_message=quota_error_msg,
                        metadata={
                            "error_type": "quota_exceeded",
                            "original_error": last_error,
                            "attempt": attempt + 1
                        }
                    )
                
                # For retryable errors
                if attempt < max_retries:
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    print(f"LLM request failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    break
        
        # Generic error for non-quota failures
        error_message = f"LLM request failed after {max_retries + 1} attempts. Last error: {last_error}"
        error_type = "request_failed"
        
        # Enhance error message based on error type
        if "network" in last_error.lower() or "connection" in last_error.lower():
            error_message = "Network connection error. Please check your internet connection and try again."
            error_type = "network_error"
        elif "timeout" in last_error.lower():
            error_message = "Request timeout. The AI service took too long to respond. Please try again."
            error_type = "timeout_error"
        elif "authentication" in last_error.lower() or "api key" in last_error.lower():
            error_message = "API authentication error. Please check your Google AI API key configuration."
            error_type = "auth_error"
        
        return ProcessingResult(
            status=ProcessingStatus.ERROR,
            error_message=error_message,
            metadata={
                "error_type": error_type,
                "original_error": last_error,
                "attempts": max_retries + 1
            }
        )
    
    def _parse_extraction_response(self, response_text: str, extraction_schema: Dict[str, str]) -> ProcessingResult:
        """Parse and validate extraction response"""
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response_text
            
            # Parse JSON
            try:
                parsed_data = json.loads(json_str)
            except json.JSONDecodeError:
                # Try to clean up common JSON issues
                cleaned_json = self._clean_json_response(json_str)
                parsed_data = json.loads(cleaned_json)
            
            # Validate that it's a dictionary
            if not isinstance(parsed_data, dict):
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    error_message="LLM response is not a valid JSON object"
                )
            
            # Validate extracted fields
            validated_data = {}
            missing_fields = []
            
            for field_name in extraction_schema.keys():
                if field_name in parsed_data:
                    value = parsed_data[field_name]
                    # Convert 'null', 'None', 'Not specified' to None
                    if value in ['null', 'None', 'Not specified', '', 'N/A']:
                        value = None
                    validated_data[field_name] = value
                else:
                    validated_data[field_name] = None
                    missing_fields.append(field_name)
            
            # Compute planned_end_date from start_date + project_duration if both are available
            self._compute_planned_end_date(validated_data)
            
            # Calculate confidence based on fields found
            fields_found = len([v for v in validated_data.values() if v is not None])
            total_fields = len(extraction_schema)
            confidence = fields_found / total_fields if total_fields > 0 else 0.0
            
            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data=validated_data,
                confidence=confidence,
                metadata={
                    "fields_found": fields_found,
                    "total_fields": total_fields,
                    "missing_fields": missing_fields,
                    "raw_response": response_text[:500]  # First 500 chars for debugging
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message=f"Failed to parse extraction response: {str(e)}",
                metadata={"raw_response": response_text[:500]}
            )
    
    def _parse_classification_response(self, response_text: str, classification_options: List[str], **kwargs) -> ProcessingResult:
        """Parse classification response"""
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # Try to extract category directly from text
                for option in classification_options:
                    if option.lower() in response_text.lower():
                        return ProcessingResult(
                            status=ProcessingStatus.SUCCESS,
                            data={"category": option, "confidence": 0.7},
                            confidence=0.7,
                            metadata={"raw_response": response_text[:200]}
                        )
                
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    error_message="Could not extract classification from response",
                    metadata={"raw_response": response_text[:200]}
                )
            
            # Parse JSON response
            try:
                parsed_data = json.loads(json_str)
            except json.JSONDecodeError:
                cleaned_json = self._clean_json_response(json_str)
                parsed_data = json.loads(cleaned_json)
            
            # Extract category and confidence
            if isinstance(parsed_data, dict):
                category = parsed_data.get("category")
                confidence = parsed_data.get("confidence", 0.5)
                all_scores = parsed_data.get("all_scores", {})
                
                # Validate category is in options
                if category not in classification_options:
                    # Try to find closest match
                    for option in classification_options:
                        if option.lower() in category.lower() or category.lower() in option.lower():
                            category = option
                            break
                    else:
                        return ProcessingResult(
                            status=ProcessingStatus.ERROR,
                            error_message=f"Classified category '{category}' not in provided options",
                            metadata={"raw_response": response_text[:200]}
                        )
                
                result_data = {
                    "category": category,
                    "confidence": float(confidence)
                }
                
                if all_scores:
                    result_data["all_scores"] = all_scores
                
                return ProcessingResult(
                    status=ProcessingStatus.SUCCESS,
                    data=result_data,
                    confidence=float(confidence),
                    metadata={"raw_response": response_text[:200]}
                )
            
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message="Invalid classification response format",
                metadata={"raw_response": response_text[:200]}
            )
            
        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message=f"Failed to parse classification response: {str(e)}",
                metadata={"raw_response": response_text[:200]}
            )
    
    def _clean_json_response(self, json_str: str) -> str:
        """Clean common JSON formatting issues from LLM responses"""
        import re
        
        # Remove markdown code blocks
        json_str = re.sub(r'```json\s*', '', json_str)
        json_str = re.sub(r'```\s*$', '', json_str)
        
        # Remove extra whitespace
        json_str = json_str.strip()
        
        # Fix common JSON issues
        json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Advanced JSON fixing - handle unescaped quotes and special characters
        try:
            # If it's already valid JSON, return as-is
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError as e:
            # Try to fix common issues
            original_json = json_str
            
            # Method 1: Fix unescaped quotes in string values
            # Pattern: "field": "value with "quotes" inside"
            # Should become: "field": "value with \"quotes\" inside"
            def fix_unescaped_quotes(match):
                field_name = match.group(1)
                field_value = match.group(2)
                # Escape internal quotes, but not the boundary quotes
                escaped_value = field_value.replace('"', '\\"')
                return f'"{field_name}": "{escaped_value}"'
            
            # Apply quote fixing for string fields
            json_str = re.sub(r'"([^"]+)":\s*"([^"]*"[^"]*)"(?=\s*[,}])', fix_unescaped_quotes, json_str)
            
            # Method 2: Fix newlines and special characters in string values
            # Replace literal newlines in JSON string values with \n
            def fix_newlines_in_strings(text):
                lines = text.split('\n')
                result_lines = []
                in_string = False
                escape_next = False
                
                for line in lines:
                    if not in_string:
                        result_lines.append(line)
                    else:
                        # We're inside a string that spans multiple lines
                        # Replace the actual newline with \n and continue on same line
                        if result_lines:
                            result_lines[-1] += '\\n' + line.strip()
                        else:
                            result_lines.append(line)
                    
                    # Track if we're inside a string value
                    for char in line:
                        if escape_next:
                            escape_next = False
                            continue
                        if char == '\\':
                            escape_next = True
                        elif char == '"' and not escape_next:
                            in_string = not in_string
                
                return '\n'.join(result_lines)
            
            # Apply newline fixing
            json_str = fix_newlines_in_strings(json_str)
            
            # Method 3: Remove control characters that break JSON
            json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
            
            # Method 4: Fix malformed arrays and objects
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)  # Remove trailing commas
            
            # Method 5: Try to extract valid JSON from response
            # Sometimes LLM adds extra text before/after JSON
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', json_str, re.DOTALL)
            if json_match:
                potential_json = json_match.group(0)
                try:
                    json.loads(potential_json)
                    return potential_json
                except json.JSONDecodeError:
                    pass
            
            # Method 6: Last resort - try to build valid JSON from key-value pairs
            try:
                # Extract key-value pairs with regex
                pairs = re.findall(r'"([^"]+)":\s*"([^"]*)"', original_json)
                if pairs:
                    clean_pairs = {}
                    for key, value in pairs:
                        # Clean the value
                        clean_value = value.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                        clean_value = re.sub(r'([^\\])"', r'\1\\"', clean_value)  # Escape unescaped quotes
                        clean_pairs[key] = clean_value
                    
                    return json.dumps(clean_pairs, ensure_ascii=False)
            except Exception:
                pass
            
            # If all fixes failed, return the best attempt
            return json_str
            
        except Exception as e:
            print(f"Warning: Advanced JSON cleaning failed: {e}")
            return json_str
    
    def get_service_status(self) -> ProcessingResult:
        """Get status of LLM service"""
        status_data = {
            "available": self.llm_available,
            "model_name": self.model_name,
            "api_key_configured": bool(self.api_key),
            "configuration": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "timeout": self.timeout,
                "retry_count": self.retry_count
            }
        }
        
        if self.llm_available:
            # Test LLM with simple request
            try:
                test_result = self._execute_llm_request("Test: Return 'OK'", 1)
                status_data["test_successful"] = test_result.status == ProcessingStatus.SUCCESS
                status_data["test_response"] = test_result.data[:50] if test_result.data else None
            except Exception as e:
                status_data["test_successful"] = False
                status_data["test_error"] = str(e)
        
        return ProcessingResult(
            status=ProcessingStatus.SUCCESS,
            data=status_data
        )
    
    def update_model_config(self, new_config: Dict[str, Any]) -> ProcessingResult:
        """Update LLM model configuration"""
        try:
            if "model_name" in new_config:
                self.model_name = new_config["model_name"]
                if self.llm_available:
                    self.model = genai.GenerativeModel(self.model_name)
            
            if "temperature" in new_config:
                self.temperature = float(new_config["temperature"])
            
            if "max_tokens" in new_config:
                self.max_tokens = int(new_config["max_tokens"])
            
            if "timeout" in new_config:
                self.timeout = int(new_config["timeout"])
            
            if "retry_count" in new_config:
                self.retry_count = int(new_config["retry_count"])
            
            # Update configuration service
            updated_config = {
                "model_name": self.model_name,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "timeout": self.timeout,
                "retry_count": self.retry_count
            }
            
            self.config_service.update_service_config("llm", updated_config)
            
            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data="LLM configuration updated successfully"
            )
            
        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message=f"Error updating LLM configuration: {str(e)}"
            )
    
    def _compute_planned_end_date(self, validated_data: Dict[str, Any]):
        """Compute planned_end_date from start_date + project_duration"""
        try:
            start_date_str = validated_data.get('start_date')
            duration_str = validated_data.get('project_duration')
            
            if not start_date_str or not duration_str:
                return  # Cannot compute without both values
            
            # Parse start date
            try:
                # Try common date formats
                start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            except ValueError:
                try:
                    start_date = datetime.strptime(start_date_str, '%d/%m/%Y')
                except ValueError:
                    try:
                        start_date = datetime.strptime(start_date_str, '%d-%m-%Y')
                    except ValueError:
                        return  # Could not parse start date
            
            # Parse duration and calculate end date
            duration_lower = duration_str.lower().strip()
            
            if 'month' in duration_lower:
                # Extract number of months
                import re
                months_match = re.search(r'(\d+)', duration_lower)
                if months_match:
                    months = int(months_match.group(1))
                    # Add months to start date
                    if start_date.month + months <= 12:
                        end_date = start_date.replace(month=start_date.month + months)
                    else:
                        years_to_add = (start_date.month + months - 1) // 12
                        new_month = ((start_date.month + months - 1) % 12) + 1
                        end_date = start_date.replace(year=start_date.year + years_to_add, month=new_month)
                    
                    validated_data['planned_end_date'] = end_date.strftime('%Y-%m-%d')
                    
            elif 'year' in duration_lower:
                # Extract number of years
                import re
                years_match = re.search(r'(\d+)', duration_lower)
                if years_match:
                    years = int(years_match.group(1))
                    end_date = start_date.replace(year=start_date.year + years)
                    validated_data['planned_end_date'] = end_date.strftime('%Y-%m-%d')
                    
            elif 'day' in duration_lower:
                # Extract number of days
                import re
                days_match = re.search(r'(\d+)', duration_lower)
                if days_match:
                    days = int(days_match.group(1))
                    end_date = start_date + timedelta(days=days)
                    validated_data['planned_end_date'] = end_date.strftime('%Y-%m-%d')
                    
        except Exception as e:
            # If computation fails, log but don't crash
            print(f"Could not compute planned_end_date: {e}")
            pass
    
    def _preprocess_text_for_efficiency(self, text_content: str) -> str:
        """
        Preprocess text to reduce token usage while preserving important information
        
        Strategies:
        - Remove excessive whitespace and line breaks
        - Remove repeated patterns and boilerplate text
        - Compress common legal phrases
        - Keep only relevant sections for contract extraction
        """
        try:
            import re
            
            # Remove excessive whitespace (but preserve structure)
            text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text_content)  # Max 2 consecutive line breaks
            text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single space
            text = re.sub(r'\n[ \t]+', '\n', text)  # Remove leading whitespace on lines
            
            # Remove common legal boilerplate that doesn't contain contract data
            boilerplate_patterns = [
                r'This document contains proprietary and confidential information[^.]*\.?',
                r'All rights reserved[^.]*\.?',
                r'Copyright \d{4}[^.]*\.?',
                r'The information contained herein[^.]*\.?',
                r'No part of this publication[^.]*\.?',
                r'CONFIDENTIAL[^.]*\.?',
                r'Attorney[\s-]Client Privileged[^.]*\.?'
            ]
            
            for pattern in boilerplate_patterns:
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)
            
            # Remove excessive repeated content (common in PDFs)
            # Remove lines that are repeated more than 3 times
            lines = text.split('\n')
            line_counts = {}
            for line in lines:
                stripped = line.strip()
                if len(stripped) > 10:  # Only count meaningful lines
                    line_counts[stripped] = line_counts.get(stripped, 0) + 1
            
            # Filter out excessively repeated lines (keep first 2 occurrences)
            filtered_lines = []
            line_seen_count = {}
            for line in lines:
                stripped = line.strip()
                if len(stripped) <= 10 or line_counts.get(stripped, 0) <= 3:
                    filtered_lines.append(line)
                else:
                    # Keep only first 2 occurrences of repeated lines
                    seen = line_seen_count.get(stripped, 0)
                    if seen < 2:
                        filtered_lines.append(line)
                        line_seen_count[stripped] = seen + 1
            
            text = '\n'.join(filtered_lines)
            
            # Compress common legal phrases to save tokens
            text = re.sub(r'WHEREAS,?\s+', 'WHEREAS ', text, flags=re.IGNORECASE)
            text = re.sub(r'NOW,?\s+THEREFORE,?\s+', 'NOW THEREFORE ', text, flags=re.IGNORECASE)
            text = re.sub(r'IN\s+WITNESS\s+WHEREOF[^.]*\.?', 'IN WITNESS WHEREOF...', text, flags=re.IGNORECASE)
            
            # Final cleanup
            text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Final whitespace cleanup
            text = text.strip()
            
            # Log compression ratio for monitoring
            original_length = len(text_content)
            compressed_length = len(text)
            compression_ratio = (original_length - compressed_length) / original_length * 100
            
            if compression_ratio > 5:  # Only log if significant compression
                print(f"Text preprocessing: {original_length} -> {compressed_length} chars ({compression_ratio:.1f}% reduction)")
            
            return text
            
        except Exception as e:
            print(f"Warning: Text preprocessing failed: {e}")
            return text_content  # Return original if preprocessing fails