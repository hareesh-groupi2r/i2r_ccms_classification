# AI Coding Assistant Implementation Prompt
## Contract Correspondence Multi-Category Classification System

### Project Context
You are tasked with implementing a sophisticated multi-category classification system for contract correspondence in infrastructure projects. This system will help a company that assists large infrastructure firms with documentation while they focus on project execution.

### Business Requirements
- **Volume**: 20-50 documents per day (real-time), with occasional batch processing
- **Accuracy**: Minimum 85%, with <5% false negative rate (false positives acceptable)
- **Processing Time**: 2-5 seconds per document
- **Integration**: Must integrate with existing Next.js + Flask backend
- **Privacy**: Documents are anonymized, can use external APIs

### Technical Requirements

#### Core Functionality
1. **Multi-label Classification**: Documents can have multiple issues and categories
2. **Issue Identification**: Extract all issues from subject line and body
3. **Category Mapping**: Map each issue to one or more categories
4. **Confidence Scoring**: Provide confidence level for each classification
5. **Justification**: Return reference sentences that justify classifications

#### Five Modular Approaches to Implement (Configurable On/Off)

##### Approach 1: Pure LLM Classification
```python
# Direct PDF/text to classification using GPT-4 or Claude
# No training required, highest accuracy but highest cost
# Expected: 90-95% accuracy, $0.02-0.05 per document
```

##### Approach 2: Hybrid RAG+LLM with Sliding Windows
```python
# Semantic search with reference database + LLM validation
# Uses sliding window embeddings (3-sentence windows, 1-sentence overlap)
# Expected: 85-92% accuracy, $0.005-0.01 per document
```

##### Approach 3: Fine-tuned LegalBERT
```python
# Use nlpaueb/legal-bert-base-uncased
# Fine-tune on 523 labeled samples
# Expected: 83-90% accuracy, $0.001 per document after training
```

##### Approach 4: Google Document AI + Gemini
```python
# Enterprise-grade OCR and entity extraction with Document AI
# Classification with Gemini 1.5 Flash (fast and accurate)
# Expected: 92-96% accuracy, $0.003-0.005 per document
# Best for: Scanned documents, complex layouts, highest accuracy needs
```

##### Approach 5: Open Source Stack (DocTR + Mixtral + BGE)
```python
# DocTR for superior OCR (better than Tesseract)
# Mixtral 8x7B for classification (self-hosted)
# BGE embeddings for semantic search
# Expected: 85-90% accuracy, <$0.001 per document after setup
# Best for: Cost optimization, on-premise deployment
```

### Data Structure

#### Training Data
```excel
Consolidated_labeled_data.xlsx:
- Column A (issue_type): 130 unique issue types
- Column B (category): 62 unique categories (comma-separated for multi-label)
  * IMPORTANT: Each issue_type maps to one or more categories
  * Same issue_type can have different category combinations across documents
  * Categories are the final labels we assign to documents
- Column C (reference_sentence): Example sentences
- Column D (source_file): Original document
- Column E (subject): Email/letter subject
- Column F (body): Full document text

Statistics:
- 523 labeled examples
- 59% have multiple categories
- Average body length: 1,554 characters
- Average reference sentence: 205 characters

Data Relationship Example:
Row 1: issue_type="Change of Scope" → categories="Contract Amendment, Cost Impact, Schedule Impact"
Row 2: issue_type="Payment Delay" → categories="Financial Management, Contract Compliance"
Row 3: issue_type="Change of Scope" → categories="Contract Amendment, Risk Management"
(Note: Same issue type can map to different category combinations based on context)
```

#### Test Data (Lot-11 Folder)
```excel
EDMS-Lot 11.xlsx:
- Column E (Issues discussed in the letter): Issue types
- Column F (Category): Categories (comma-separated for multi-label)
- Column H (Reference sentence in the letter): Justification text

Test PDFs:
- 26 scanned PDF documents in Lot-11 folder
- Require OCR processing (Tesseract/PyTesseract)
- Mix of change of scope proposals, toll plaza construction, etc.
```

### Implementation Tasks

#### Phase 1: Foundation Setup
1. **Project Structure**:
```
ccms_classification/
├── classifier/
│   ├── __init__.py
│   ├── config_manager.py    # Modular configuration system
│   ├── preprocessing.py     # NLP preprocessing with lemmatization/stemming
│   ├── embeddings.py        # Sliding window embeddings
│   ├── pure_llm.py         # Approach 1
│   ├── hybrid_rag.py       # Approach 2
│   ├── legalbert.py        # Approach 3
│   ├── google_docai.py     # Approach 4
│   ├── open_source.py      # Approach 5
│   ├── metrics.py          # Metrics calculation
│   └── ensemble.py         # Ensemble voting
├── data/
│   ├── processed/           # Preprocessed data
│   ├── models/             # Saved models
│   └── embeddings/         # Vector databases
├── api/
│   ├── app.py              # Flask API
│   └── routes.py           # API endpoints
├── scripts/
│   ├── prepare_data.py     # Data preparation
│   ├── build_issue_mapping.py  # Build issue-to-category mappings
│   ├── train_legalbert.py  # LegalBERT training
│   └── evaluate.py         # Model evaluation
├── tests/
├── requirements.txt
└── README.md
```

2. **Dependencies Installation**:
```txt
# Core dependencies
flask==2.3.0
flask-cors==4.0.0
pandas==2.0.0
numpy==1.24.0
scikit-learn==1.3.0
python-dotenv==1.0.0
redis==4.6.0
plotly==5.17.0
openpyxl==3.1.2
pyyaml==6.0.1

# NLP and embeddings
nltk==3.8.1
spacy==3.6.0
sentence-transformers==2.2.2
transformers==4.35.0
torch==2.1.0
faiss-cpu==1.7.4

# PDF processing
pdfplumber==0.10.3
PyPDF2==3.0.1
pytesseract==0.3.10
pdf2image==1.16.3
Pillow==10.0.0

# Approach-specific (install as needed)
openai==1.0.0  # For GPT-4
anthropic==0.7.0  # For Claude
google-cloud-documentai==2.20.0  # For Google Document AI
google-generativeai==0.3.0  # For Gemini
doctr-python==0.6.0  # For DocTR OCR
vllm==0.2.0  # For Mixtral inference
```

#### Phase 2: Core Components

##### 2.1 Configuration Manager
```python
# config_manager.py
import yaml
from typing import Dict, List

class ConfigManager:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get_enabled_approaches(self) -> List[str]:
        """Return list of enabled approaches"""
        enabled = []
        for approach, settings in self.config['approaches'].items():
            if settings.get('enabled', False):
                enabled.append(approach)
        return enabled
    
    def get_approach_config(self, approach: str) -> Dict:
        """Get configuration for specific approach"""
        return self.config['approaches'].get(approach, {})
    
    def update_approach_status(self, approach: str, enabled: bool):
        """Dynamically enable/disable approach"""
        if approach in self.config['approaches']:
            self.config['approaches'][approach]['enabled'] = enabled
            self.save_config()
    
    def get_ensemble_config(self) -> Dict:
        return self.config.get('ensemble', {})
```

##### 2.2 Issue-Category Mapper
```python
from typing import Dict, List, Set, Tuple
import pandas as pd
from collections import defaultdict

class IssueCategoryMapper:
    def __init__(self, training_data_path: str):
        """
        Build issue-type to category mapping from training data
        """
        self.issue_to_categories = defaultdict(set)
        self.category_frequencies = defaultdict(int)
        self.issue_frequencies = defaultdict(int)
        self._build_mapping(training_data_path)
    
    def _build_mapping(self, data_path: str):
        """
        Analyze training data to build issue-category relationships
        """
        df = pd.read_excel(data_path)
        
        for _, row in df.iterrows():
            issue_type = row['issue_type']
            categories = [c.strip() for c in row['category'].split(',')]
            
            # Build mapping dictionary
            self.issue_to_categories[issue_type].update(categories)
            
            # Track frequencies for confidence scoring
            self.issue_frequencies[issue_type] += 1
            for category in categories:
                self.category_frequencies[category] += 1
        
        # Convert sets to lists for consistency
        self.issue_to_categories = {
            issue: list(cats) 
            for issue, cats in self.issue_to_categories.items()
        }
        
        # Calculate mapping statistics
        self.calculate_mapping_stats()
    
    def get_categories_for_issue(self, issue_type: str, 
                                 confidence_threshold: float = 0.0) -> List[Tuple[str, float]]:
        """
        Get categories for a given issue type with confidence scores
        """
        if issue_type not in self.issue_to_categories:
            return []
        
        categories = self.issue_to_categories[issue_type]
        
        # Calculate confidence based on frequency in training data
        results = []
        for category in categories:
            confidence = self._calculate_confidence(issue_type, category)
            if confidence >= confidence_threshold:
                results.append((category, confidence))
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def _calculate_confidence(self, issue_type: str, category: str) -> float:
        """
        Calculate confidence score for issue-category mapping
        """
        # Base confidence on co-occurrence frequency
        issue_freq = self.issue_frequencies.get(issue_type, 0)
        category_freq = self.category_frequencies.get(category, 0)
        
        if issue_freq == 0 or category_freq == 0:
            return 0.0
        
        # Simple confidence calculation (can be enhanced)
        base_confidence = 0.7  # Base confidence for known mappings
        freq_boost = min(0.3, issue_freq / 100)  # Boost for common patterns
        
        return min(1.0, base_confidence + freq_boost)
    
    def calculate_mapping_stats(self):
        """
        Generate statistics about issue-category mappings
        """
        self.stats = {
            'total_issue_types': len(self.issue_to_categories),
            'total_unique_categories': len(set(
                cat for cats in self.issue_to_categories.values() 
                for cat in cats
            )),
            'avg_categories_per_issue': sum(
                len(cats) for cats in self.issue_to_categories.values()
            ) / len(self.issue_to_categories) if self.issue_to_categories else 0,
            'max_categories_per_issue': max(
                len(cats) for cats in self.issue_to_categories.values()
            ) if self.issue_to_categories else 0,
        }
        
        return self.stats
    
    def map_issues_to_categories(self, identified_issues: List[Dict]) -> List[Dict]:
        """
        Map identified issues to their categories with aggregated confidence
        
        Args:
            identified_issues: List of dicts with 'issue_type' and 'confidence'
        
        Returns:
            List of dicts with 'category', 'confidence', 'source_issues'
        """
        category_scores = defaultdict(lambda: {'confidence': 0, 'sources': []})
        
        for issue in identified_issues:
            issue_type = issue['issue_type']
            issue_confidence = issue.get('confidence', 1.0)
            
            # Get categories for this issue
            categories = self.get_categories_for_issue(issue_type)
            
            for category, mapping_confidence in categories:
                # Combine confidences: issue identification * mapping confidence
                combined_confidence = issue_confidence * mapping_confidence
                
                # Aggregate if multiple issues map to same category
                if category_scores[category]['confidence'] < combined_confidence:
                    category_scores[category]['confidence'] = combined_confidence
                
                category_scores[category]['sources'].append({
                    'issue_type': issue_type,
                    'confidence': combined_confidence
                })
        
        # Convert to list format
        results = [
            {
                'category': category,
                'confidence': data['confidence'],
                'source_issues': data['sources']
            }
            for category, data in category_scores.items()
        ]
        
        return sorted(results, key=lambda x: x['confidence'], reverse=True)
```

##### 2.3 Validation Engine (Prevent LLM Hallucinations)
```python
from typing import List, Dict, Set, Tuple, Optional
import difflib
import logging

class ValidationEngine:
    def __init__(self, training_data_path: str):
        """
        Initialize validation engine with strict allowlists
        """
        self.valid_issue_types = set()
        self.valid_categories = set()
        self._load_valid_values(training_data_path)
        self.logger = logging.getLogger(__name__)
        
    def _load_valid_values(self, data_path: str):
        """
        Load the exhaustive list of valid issue types and categories
        """
        df = pd.read_excel(data_path)
        
        # Extract all unique issue types (130 total)
        self.valid_issue_types = set(df['issue_type'].unique())
        
        # Extract all unique categories (62 total)
        all_categories = set()
        for categories_str in df['category']:
            categories = [c.strip() for c in categories_str.split(',')]
            all_categories.update(categories)
        self.valid_categories = all_categories
        
        self.logger.info(f"Loaded {len(self.valid_issue_types)} valid issue types")
        self.logger.info(f"Loaded {len(self.valid_categories)} valid categories")
    
    def validate_issue_type(self, issue_type: str, 
                           auto_correct: bool = True) -> Tuple[str, bool, float]:
        """
        Validate and optionally correct an issue type
        
        Returns:
            (validated_issue, is_valid, confidence)
        """
        # Exact match
        if issue_type in self.valid_issue_types:
            return issue_type, True, 1.0
        
        # Case-insensitive match
        issue_lower = issue_type.lower()
        for valid_issue in self.valid_issue_types:
            if valid_issue.lower() == issue_lower:
                return valid_issue, True, 0.95
        
        if auto_correct:
            # Find closest match using fuzzy matching
            closest = difflib.get_close_matches(
                issue_type, 
                self.valid_issue_types, 
                n=1, 
                cutoff=0.7
            )
            
            if closest:
                similarity = difflib.SequenceMatcher(
                    None, issue_type, closest[0]
                ).ratio()
                
                self.logger.warning(
                    f"Auto-corrected hallucinated issue '{issue_type}' to '{closest[0]}' "
                    f"(similarity: {similarity:.2f})"
                )
                return closest[0], False, similarity
        
        # No valid match found
        self.logger.error(f"Rejected hallucinated issue type: '{issue_type}'")
        return None, False, 0.0
    
    def validate_category(self, category: str, 
                         auto_correct: bool = True) -> Tuple[str, bool, float]:
        """
        Validate and optionally correct a category
        
        Returns:
            (validated_category, is_valid, confidence)
        """
        # Exact match
        if category in self.valid_categories:
            return category, True, 1.0
        
        # Case-insensitive match
        category_lower = category.lower()
        for valid_cat in self.valid_categories:
            if valid_cat.lower() == category_lower:
                return valid_cat, True, 0.95
        
        if auto_correct:
            # Find closest match
            closest = difflib.get_close_matches(
                category, 
                self.valid_categories, 
                n=1, 
                cutoff=0.7
            )
            
            if closest:
                similarity = difflib.SequenceMatcher(
                    None, category, closest[0]
                ).ratio()
                
                self.logger.warning(
                    f"Auto-corrected hallucinated category '{category}' to '{closest[0]}' "
                    f"(similarity: {similarity:.2f})"
                )
                return closest[0], False, similarity
        
        # No valid match found
        self.logger.error(f"Rejected hallucinated category: '{category}'")
        return None, False, 0.0
    
    def validate_classification_output(self, 
                                      classification: Dict) -> Dict:
        """
        Validate entire classification output and filter/correct invalid values
        """
        validated = {
            'identified_issues': [],
            'categories': [],
            'validation_report': {
                'hallucinations_detected': False,
                'corrections_made': [],
                'rejections': []
            }
        }
        
        # Validate issues
        if 'identified_issues' in classification:
            for issue in classification['identified_issues']:
                validated_issue, is_valid, confidence = self.validate_issue_type(
                    issue.get('issue_type', ''),
                    auto_correct=True
                )
                
                if validated_issue:
                    issue_copy = issue.copy()
                    issue_copy['issue_type'] = validated_issue
                    issue_copy['confidence'] *= confidence
                    issue_copy['validation_status'] = 'valid' if is_valid else 'corrected'
                    validated['identified_issues'].append(issue_copy)
                    
                    if not is_valid:
                        validated['validation_report']['hallucinations_detected'] = True
                        validated['validation_report']['corrections_made'].append({
                            'type': 'issue',
                            'original': issue.get('issue_type'),
                            'corrected': validated_issue,
                            'confidence': confidence
                        })
                else:
                    validated['validation_report']['rejections'].append({
                        'type': 'issue',
                        'value': issue.get('issue_type'),
                        'reason': 'No valid match found'
                    })
        
        # Validate categories
        if 'categories' in classification:
            for category_info in classification['categories']:
                validated_cat, is_valid, confidence = self.validate_category(
                    category_info.get('category', ''),
                    auto_correct=True
                )
                
                if validated_cat:
                    cat_copy = category_info.copy()
                    cat_copy['category'] = validated_cat
                    cat_copy['confidence'] *= confidence
                    cat_copy['validation_status'] = 'valid' if is_valid else 'corrected'
                    validated['categories'].append(cat_copy)
                    
                    if not is_valid:
                        validated['validation_report']['hallucinations_detected'] = True
                        validated['validation_report']['corrections_made'].append({
                            'type': 'category',
                            'original': category_info.get('category'),
                            'corrected': validated_cat,
                            'confidence': confidence
                        })
                else:
                    validated['validation_report']['rejections'].append({
                        'type': 'category',
                        'value': category_info.get('category'),
                        'reason': 'No valid match found'
                    })
        
        return validated
    
    def create_constrained_prompt(self, prompt_type: str = 'issues') -> str:
        """
        Create prompt with explicit constraints to prevent hallucinations
        """
        if prompt_type == 'issues':
            return f"""
STRICT INSTRUCTION: You MUST ONLY use issue types from this exact list. 
DO NOT create new issue types. If uncertain, choose the closest match from this list:

VALID ISSUE TYPES (ONLY use these):
{json.dumps(list(self.valid_issue_types), indent=2)}

Any issue type not in this list will be rejected.
"""
        else:  # categories
            return f"""
STRICT INSTRUCTION: You MUST ONLY use categories from this exact list.
DO NOT create new categories. If uncertain, choose the closest match from this list:

VALID CATEGORIES (ONLY use these):
{json.dumps(list(self.valid_categories), indent=2)}

Any category not in this list will be rejected.
"""
```

##### 2.4 Data Sufficiency Analyzer
```python
class DataSufficiencyAnalyzer:
    def __init__(self, training_data_path: str):
        """
        Analyze data sufficiency for reliable classification
        """
        self.df = pd.read_excel(training_data_path)
        self.issue_counts = {}
        self.category_counts = {}
        self.sufficiency_thresholds = {
            'critical': 5,   # < 5 samples: unreliable
            'warning': 10,   # 5-10 samples: low confidence
            'good': 20,      # 10-20 samples: acceptable
            'excellent': 50  # > 50 samples: high confidence
        }
        self._analyze_distribution()
    
    def _analyze_distribution(self):
        """
        Count samples for each issue type and category
        """
        # Count issue types
        self.issue_counts = self.df['issue_type'].value_counts().to_dict()
        
        # Count categories (handling multi-label)
        category_counter = defaultdict(int)
        for categories_str in self.df['category']:
            categories = [c.strip() for c in categories_str.split(',')]
            for category in categories:
                category_counter[category] += 1
        self.category_counts = dict(category_counter)
    
    def get_sufficiency_level(self, count: int) -> str:
        """
        Determine sufficiency level based on sample count
        """
        if count < self.sufficiency_thresholds['critical']:
            return 'critical'
        elif count < self.sufficiency_thresholds['warning']:
            return 'warning'
        elif count < self.sufficiency_thresholds['good']:
            return 'good'
        elif count < self.sufficiency_thresholds['excellent']:
            return 'very_good'
        else:
            return 'excellent'
    
    def get_confidence_adjustment(self, item_type: str, item_name: str) -> float:
        """
        Adjust confidence based on data availability
        """
        if item_type == 'issue':
            count = self.issue_counts.get(item_name, 0)
        else:  # category
            count = self.category_counts.get(item_name, 0)
        
        level = self.get_sufficiency_level(count)
        
        # Confidence multipliers based on data sufficiency
        adjustments = {
            'critical': 0.5,    # Halve confidence for very low data
            'warning': 0.7,     # Reduce confidence by 30%
            'good': 0.85,       # Reduce confidence by 15%
            'very_good': 0.95,  # Reduce confidence by 5%
            'excellent': 1.0    # No adjustment
        }
        
        return adjustments.get(level, 1.0)
    
    def generate_sufficiency_report(self) -> Dict:
        """
        Generate comprehensive data sufficiency report
        """
        report = {
            'summary': {
                'total_samples': len(self.df),
                'unique_issue_types': len(self.issue_counts),
                'unique_categories': len(self.category_counts)
            },
            'critical_issues': [],
            'warning_issues': [],
            'critical_categories': [],
            'warning_categories': [],
            'recommendations': []
        }
        
        # Analyze issue types
        for issue, count in self.issue_counts.items():
            level = self.get_sufficiency_level(count)
            if level == 'critical':
                report['critical_issues'].append({
                    'issue_type': issue,
                    'sample_count': count,
                    'status': 'CRITICAL - Needs immediate data collection'
                })
            elif level == 'warning':
                report['warning_issues'].append({
                    'issue_type': issue,
                    'sample_count': count,
                    'status': 'WARNING - Consider collecting more data'
                })
        
        # Analyze categories
        for category, count in self.category_counts.items():
            level = self.get_sufficiency_level(count)
            if level == 'critical':
                report['critical_categories'].append({
                    'category': category,
                    'sample_count': count,
                    'status': 'CRITICAL - Needs immediate data collection'
                })
            elif level == 'warning':
                report['warning_categories'].append({
                    'category': category,
                    'sample_count': count,
                    'status': 'WARNING - Consider collecting more data'
                })
        
        # Generate recommendations
        if report['critical_issues'] or report['critical_categories']:
            report['recommendations'].append(
                f"URGENT: {len(report['critical_issues'])} issue types and "
                f"{len(report['critical_categories'])} categories have critically low data. "
                f"Classification for these will be unreliable."
            )
        
        if report['warning_issues'] or report['warning_categories']:
            report['recommendations'].append(
                f"ATTENTION: {len(report['warning_issues'])} issue types and "
                f"{len(report['warning_categories'])} categories have low data. "
                f"Consider prioritizing data collection for these."
            )
        
        # Sort by sample count (ascending) to prioritize worst cases
        report['critical_issues'].sort(key=lambda x: x['sample_count'])
        report['warning_issues'].sort(key=lambda x: x['sample_count'])
        report['critical_categories'].sort(key=lambda x: x['sample_count'])
        report['warning_categories'].sort(key=lambda x: x['sample_count'])
        
        return report
    
    def apply_confidence_adjustments(self, classification: Dict) -> Dict:
        """
        Adjust classification confidence based on data sufficiency
        """
        adjusted = classification.copy()
        
        # Add data sufficiency warnings
        adjusted['data_sufficiency_warnings'] = []
        
        # Adjust issue confidences
        if 'identified_issues' in adjusted:
            for issue in adjusted['identified_issues']:
                issue_type = issue['issue_type']
                count = self.issue_counts.get(issue_type, 0)
                level = self.get_sufficiency_level(count)
                adjustment = self.get_confidence_adjustment('issue', issue_type)
                
                issue['original_confidence'] = issue['confidence']
                issue['confidence'] *= adjustment
                issue['data_sufficiency'] = level
                issue['training_samples'] = count
                
                if level in ['critical', 'warning']:
                    adjusted['data_sufficiency_warnings'].append({
                        'type': 'issue',
                        'name': issue_type,
                        'level': level,
                        'sample_count': count,
                        'message': f"Low training data for '{issue_type}' ({count} samples)"
                    })
        
        # Adjust category confidences
        if 'categories' in adjusted:
            for category in adjusted['categories']:
                cat_name = category['category']
                count = self.category_counts.get(cat_name, 0)
                level = self.get_sufficiency_level(count)
                adjustment = self.get_confidence_adjustment('category', cat_name)
                
                category['original_confidence'] = category['confidence']
                category['confidence'] *= adjustment
                category['data_sufficiency'] = level
                category['training_samples'] = count
                
                if level in ['critical', 'warning']:
                    adjusted['data_sufficiency_warnings'].append({
                        'type': 'category',
                        'name': cat_name,
                        'level': level,
                        'sample_count': count,
                        'message': f"Low training data for '{cat_name}' ({count} samples)"
                    })
        
        return adjusted
```

##### 2.5 NLP Preprocessing Module
```python
class TextPreprocessor:
    def __init__(self):
        # Initialize NLTK components
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        # Load spaCy model
        self.nlp = spacy.load("en_core_web_sm")
        
    def preprocess(self, text):
        # 1. Clean text
        # 2. Tokenization
        # 3. Lemmatization
        # 4. Stemming
        # 5. Remove stopwords (preserve contract terms)
        # Return processed text
```

##### 2.4 Sliding Window Embeddings
```python
class SlidingWindowEmbedder:
    def __init__(self, window_size=3, overlap=1):
        # Create overlapping chunks
        # Generate embeddings with context
        # Aggregate chunk classifications
```

##### 2.5 Metrics Engine
```python
class MetricsEngine:
    def calculate_metrics(self, y_true, y_pred):
        # Calculate TP, FP, TN, FN
        # Precision, Recall, F1-Score
        # Per-category metrics
        # Return comprehensive metrics dict
```

#### Phase 3: Classifier Implementation

##### Pure LLM Classifier (Two-Phase Approach)
```python
class PureLLMClassifier:
    def __init__(self, config, issue_category_mapper, validation_engine, data_analyzer):
        self.config = config
        self.mapper = issue_category_mapper
        self.validator = validation_engine
        self.data_analyzer = data_analyzer
        self.llm_client = self._init_llm_client()
    
    def classify(self, document_text: str) -> Dict:
        # Phase 1: Identify issue types with constrained prompt
        identified_issues = self._identify_issues(document_text)
        
        # Phase 2: Validate identified issues
        validated_issues = self._validate_issues(identified_issues)
        
        # Phase 3: Map issues to categories
        categories = self.mapper.map_issues_to_categories(validated_issues)
        
        # Phase 4: Validate categories
        validated_categories = self._validate_categories(categories)
        
        # Phase 5: Apply data sufficiency adjustments
        result = {
            'identified_issues': validated_issues,
            'categories': validated_categories,
            'classification_path': 'issue_identification -> validation -> category_mapping'
        }
        
        # Apply confidence adjustments based on data sufficiency
        result = self.data_analyzer.apply_confidence_adjustments(result)
        
        return result
    
    def _identify_issues(self, text: str) -> List[Dict]:
        # Get constrained prompt to prevent hallucinations
        constraints = self.validator.create_constrained_prompt('issues')
        
        prompt = f"""
        {constraints}
        
        Analyze this contract correspondence and identify all issue types discussed.
        ONLY use issue types from the provided list above.
        
        Document: {text[:3000]}
        
        Return JSON: {{"issues": [{{"issue_type": "...", "confidence": 0.95, "evidence": "..."}}]}}
        """
        
        response = self.llm_client.complete(prompt)
        return self._parse_issue_response(response)
    
    def _validate_issues(self, issues: List[Dict]) -> List[Dict]:
        validated = []
        for issue in issues:
            validated_type, is_valid, confidence = self.validator.validate_issue_type(
                issue['issue_type']
            )
            if validated_type:
                issue['issue_type'] = validated_type
                issue['confidence'] *= confidence
                issue['validation_status'] = 'valid' if is_valid else 'corrected'
                validated.append(issue)
        return validated
    
    def _validate_categories(self, categories: List[Dict]) -> List[Dict]:
        validated = []
        for cat in categories:
            validated_cat, is_valid, confidence = self.validator.validate_category(
                cat['category']
            )
            if validated_cat:
                cat['category'] = validated_cat
                cat['confidence'] *= confidence
                cat['validation_status'] = 'valid' if is_valid else 'corrected'
                validated.append(cat)
        return validated
```
- Extract text from PDF using pdfplumber for regular PDFs
- Use OCR (Tesseract) for scanned PDFs when text extraction fails
- First identify issue types, then map to categories
- Validate mapping with LLM if needed
- Handle token limits with intelligent chunking

##### Hybrid RAG+LLM (Two-Phase Approach)
```python
class HybridRAGClassifier:
    def __init__(self, config, issue_category_mapper):
        self.config = config
        self.mapper = issue_category_mapper
        self.embedder = SentenceTransformer(config['embedding_model'])
        self.index = self._build_faiss_index()
    
    def classify(self, document_text: str) -> Dict:
        # Phase 1: Semantic search to find similar issues
        similar_issues = self._semantic_search_issues(document_text)
        
        # Phase 2: Map identified issues to categories
        categories = self.mapper.map_issues_to_categories(similar_issues)
        
        # Phase 3: LLM validation and refinement
        refined_results = self._llm_validation(document_text, categories)
        
        return refined_results
    
    def _semantic_search_issues(self, text: str) -> List[Dict]:
        # Create sliding windows
        windows = create_sliding_windows(text, window_size=3, overlap=1)
        
        # Search for each window and aggregate results
        all_matches = []
        for window in windows:
            matches = self._search_index(window, k=10)
            all_matches.extend(matches)
        
        # Aggregate and deduplicate issues
        return self._aggregate_issues(all_matches)
```
- Build FAISS index from 523 reference samples with issue types
- Implement semantic search with k=10-15 for issue identification
- Use sliding windows for better context
- Map found issues to categories using IssueCategoryMapper
- LLM validation of category assignments
- Weighted aggregation of results

##### LegalBERT Classifier
- Load pre-trained LegalBERT model
- Prepare training data with multi-label encoding
- Fine-tune for 5 epochs on Google Colab
- Implement inference with confidence scores
- Handle long documents with chunking

##### Google Document AI + Gemini
```python
from google.cloud import documentai_v1 as documentai
import google.generativeai as genai

class GoogleDocAIClassifier:
    def __init__(self, config):
        self.client = documentai.DocumentProcessorServiceClient()
        self.processor_name = config['processor_id']
        genai.configure(api_key=config['gemini_api_key'])
        self.model = genai.GenerativeModel(config['gemini_model'])
    
    def classify(self, pdf_path):
        # Step 1: Extract with Document AI
        entities = self.extract_with_docai(pdf_path)
        
        # Step 2: Classify with Gemini
        classification = self.classify_with_gemini(entities)
        
        return classification
```

##### Open Source Stack (DocTR + Mixtral)
```python
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from vllm import LLM, SamplingParams

class OpenSourceClassifier:
    def __init__(self, config):
        # Initialize DocTR
        self.ocr = ocr_predictor(pretrained=True)
        
        # Initialize Mixtral
        self.llm = LLM(model="mistralai/Mixtral-8x7B-Instruct-v0.1")
        
        # Initialize BGE embeddings
        self.embedder = SentenceTransformer('BAAI/bge-large-en')
    
    def classify(self, pdf_path):
        # Step 1: OCR with DocTR
        text = self.extract_with_doctr(pdf_path)
        
        # Step 2: Embed and search
        similar_refs = self.semantic_search(text)
        
        # Step 3: Classify with Mixtral
        classification = self.classify_with_mixtral(text, similar_refs)
        
        return classification
```

#### Phase 4: API Development

##### Flask API Endpoints
```python
from classifier.config_manager import ConfigManager
from classifier.orchestrator import ClassificationOrchestrator

config_manager = ConfigManager()
orchestrator = ClassificationOrchestrator(config_manager)

@app.route('/api/classify', methods=['POST'])
def classify():
    """
    Main classification endpoint with approach selection
    """
    file = request.files['document']
    approaches = request.json.get('approaches', None)  # Optional
    
    if not approaches:
        # Use enabled approaches from config
        approaches = config_manager.get_enabled_approaches()
    
    results = orchestrator.classify(file, approaches)
    return jsonify(results)

@app.route('/api/config/approaches', methods=['GET'])
def get_approaches():
    """Get current approach configuration"""
    return jsonify(config_manager.config['approaches'])

@app.route('/api/config/approaches/<approach>', methods=['PUT'])
def toggle_approach(approach):
    """Enable/disable specific approach"""
    enabled = request.json.get('enabled')
    config_manager.update_approach_status(approach, enabled)
    return jsonify({"status": "updated"})

@app.route('/api/classify/compare', methods=['POST'])
def compare_all():
    """Compare all enabled approaches"""
    file = request.files['document']
    results = orchestrator.compare_all_approaches(file)
    return jsonify(results)

@app.route('/api/metrics/history', methods=['GET'])
def metrics_history():
    """Get historical performance metrics"""
    return jsonify(orchestrator.get_metrics_history())
```

##### Request/Response Format
```json
// Request
{
    "document": "base64_encoded_pdf_or_text",
    "approach": "pure_llm|hybrid|legalbert|all",
    "ground_truth": ["category1", "category2"] // optional
}

// Response
{
    "status": "success",
    "results": {
        "identified_issues": [
            {
                "issue_type": "Change of Scope",
                "confidence": 0.92,
                "evidence": "The contractor requests modification to the original scope...",
                "location": "paragraph_3"
            },
            {
                "issue_type": "Payment Delay",
                "confidence": 0.85,
                "evidence": "Payment for milestone 3 has been delayed...",
                "location": "paragraph_5"
            }
        ],
        "categories": [
            {
                "category": "Contract Amendment",
                "confidence": 0.95,
                "source_issues": [
                    {"issue_type": "Change of Scope", "contribution": 0.92}
                ]
            },
            {
                "category": "Financial Management",
                "confidence": 0.87,
                "source_issues": [
                    {"issue_type": "Payment Delay", "contribution": 0.85}
                ]
            },
            {
                "category": "Risk Management",
                "confidence": 0.78,
                "source_issues": [
                    {"issue_type": "Change of Scope", "contribution": 0.46},
                    {"issue_type": "Payment Delay", "contribution": 0.32}
                ]
            }
        ],
        "classification_path": "issue_identification -> category_mapping",
        "confidence_scores": {
            "Contract Amendment": 0.95,
            "Financial Management": 0.87,
            "Risk Management": 0.78
        },
        "justifications": [
            {
                "category": "Contract Amendment",
                "reference": "The contractor requests modification to the original scope",
                "similarity": 0.92,
                "source_issue": "Change of Scope"
            }
        ],
        "validation_report": {
            "hallucinations_detected": false,
            "corrections_made": [],
            "rejections": [],
            "validation_status": "clean"
        },
        "data_sufficiency_warnings": [
            {
                "type": "issue",
                "name": "Rare Issue Type",
                "level": "critical",
                "sample_count": 3,
                "message": "Low training data for 'Rare Issue Type' (3 samples)"
            }
        ],
        "metrics": {
            "issues_identified": 2,
            "categories_assigned": 3,
            "avg_confidence": 0.87,
            "confidence_adjustments_applied": true,
            "true_positives": 3,
            "false_positives": 0,
            "false_negatives": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1_score": 1.0
        },
        "processing_time": 2.3,
        "approach_used": "hybrid"
    }
}
```

### Critical Implementation Details

#### 1. Sliding Window Implementation
- Use 3-sentence windows with 1-sentence overlap
- Weight classifications by frequency across chunks
- Preserve spatial information (position in document)

#### 2. Multi-Label Handling
- Use sklearn.preprocessing.MultiLabelBinarizer
- Set threshold at 0.5 for binary classification
- Allow multiple categories per document

#### 3. Confidence Calibration
```python
confidence = weighted_average(
    semantic_similarity * 0.4,
    llm_confidence * 0.4,
    historical_accuracy * 0.2
)
```

#### 4. Error Handling
- Graceful degradation if one approach fails
- Retry logic for API calls
- Comprehensive logging

#### 5. Performance Optimization
- Cache embeddings for reference database
- Batch processing for multiple documents
- Redis for API response caching

### Evaluation Criteria

#### Metrics to Track
1. **Overall Accuracy**: >85%
2. **False Negative Rate**: <5%
3. **Per-Category Precision/Recall**
4. **Processing Time**: <5 seconds
5. **Cost per Document**
6. **Agreement Between Approaches**

#### Test Scenarios
1. Single-category documents
2. Multi-category documents (2-5 categories)
3. Long documents (>5000 chars)
4. Documents with novel issues
5. Batch processing (50 documents)
6. **Lot-11 Test Suite**:
   - 26 scanned PDFs with OCR requirements
   - Ground truth in EDMS-Lot 11.xlsx
   - Focus on change of scope proposals
   - Validate OCR accuracy
   - Compare all three approaches

### Active Learning Pipeline
1. Store all predictions with confidence scores
2. Collect user corrections via API
3. Retrain when 50+ corrections accumulated
4. Update reference database weekly
5. Monitor performance degradation

### Integration with Existing System
```python
# In existing Flask backend
from classifier.api import classify_document

@app.route('/ccms/classify', methods=['POST'])
def ccms_classify():
    file = request.files['document']
    result = classify_document(file)
    return jsonify(result)
```

### Deliverables Checklist
- [ ] Three working classifiers (Pure LLM, Hybrid, LegalBERT)
- [ ] Metrics comparison system
- [ ] Flask API with all endpoints
- [ ] Sliding window embedding system
- [ ] NLP preprocessing pipeline
- [ ] Ensemble voting mechanism
- [ ] Active learning feedback loop
- [ ] Docker deployment configuration
- [ ] Comprehensive documentation
- [ ] Test suite with >80% coverage
- [ ] Performance benchmark report
- [ ] Requirements.txt file
- [ ] Setup and deployment guide

### Development Milestones

#### Week 1
- [ ] Complete project setup
- [ ] Implement NLP preprocessing
- [ ] Create sliding window embedder
- [ ] Prepare reference database

#### Week 2
- [ ] Implement Pure LLM classifier
- [ ] Implement Hybrid RAG+LLM
- [ ] Fine-tune LegalBERT
- [ ] Individual approach testing

#### Week 3
- [ ] Build metrics engine
- [ ] Create ensemble system
- [ ] Develop comparison framework
- [ ] Performance benchmarking

#### Week 4
- [ ] Flask API development
- [ ] Integration testing
- [ ] Documentation
- [ ] Deployment preparation

### Success Criteria
1. System achieves >85% accuracy on test set
2. False negative rate <5%
3. Processing time <5 seconds per document
4. Clear performance winner identified among approaches
5. Successful integration with existing Flask backend
6. Active learning pipeline operational

### Notes for Implementation
- Start with data exploration to understand category distribution
- Implement comprehensive logging from the beginning
- Use type hints for better code maintainability
- Create unit tests alongside implementation
- Document API endpoints with examples
- Consider rate limiting for production deployment
- Implement health check endpoints
- Add monitoring and alerting capabilities

### Questions to Clarify Before Starting
1. Confirm API key availability (OpenAI/Anthropic)
2. Verify Google Colab access for LegalBERT training
3. Confirm Flask backend integration approach
4. Validate data privacy requirements
5. Confirm deployment infrastructure (local/cloud)

### Configuration File Example

```yaml
# config.yaml
approaches:
  pure_llm:
    enabled: true
    model: "gpt-4-turbo"  # or "claude-3-opus"
    max_tokens: 4096
    temperature: 0.1
    api_key: "${OPENAI_API_KEY}"
    
  hybrid_rag:
    enabled: true
    embedding_model: "all-mpnet-base-v2"
    llm_model: "gpt-3.5-turbo"
    vector_db: "faiss"  # or "pinecone", "qdrant"
    top_k: 15
    window_size: 3
    overlap: 1
    
  legalbert:
    enabled: true
    model_path: "./models/legalbert-finetuned"
    max_length: 512
    batch_size: 8
    
  google_docai:
    enabled: true
    project_id: "${GCP_PROJECT_ID}"
    processor_id: "contract-parser"
    location: "us"
    gemini_model: "gemini-1.5-flash"
    gemini_api_key: "${GEMINI_API_KEY}"
    
  open_source:
    enabled: true
    ocr: "doctr"  # or "tesseract"
    llm_model: "mistralai/Mixtral-8x7B-Instruct-v0.1"
    embedding_model: "BAAI/bge-large-en"
    device: "cuda"  # or "cpu"
    
ensemble:
  enabled: false
  min_approaches: 3
  voting_strategy: "weighted"
  weights:
    pure_llm: 0.25
    hybrid_rag: 0.25
    legalbert: 0.20
    google_docai: 0.20
    open_source: 0.10

routing:
  auto_select: true
  rules:
    - condition: "page_count > 20"
      use_approach: "google_docai"
    - condition: "is_scanned == true"
      use_approach: "google_docai"
    - condition: "criticality == 'high'"
      use_approach: "ensemble"
    - condition: "budget == 'low'"
      use_approach: "open_source"

monitoring:
  track_metrics: true
  log_level: "INFO"
  save_results: true
  results_path: "./data/results/"
```

### Example Code Snippets

#### Build Issue-to-Category Mapping Script
```python
# scripts/build_issue_mapping.py
import pandas as pd
import json
from collections import defaultdict
from pathlib import Path

def analyze_issue_category_relationships(data_path: str):
    """
    Analyze the training data to understand issue-category relationships
    """
    df = pd.read_excel(data_path)
    
    # Build comprehensive mapping
    issue_to_categories = defaultdict(list)
    category_to_issues = defaultdict(list)
    co_occurrence_matrix = defaultdict(lambda: defaultdict(int))
    
    for _, row in df.iterrows():
        issue_type = row['issue_type']
        categories = [c.strip() for c in row['category'].split(',')]
        
        # Track all category combinations for each issue
        issue_to_categories[issue_type].append(categories)
        
        # Track reverse mapping
        for category in categories:
            category_to_issues[category].append(issue_type)
        
        # Build co-occurrence matrix
        for cat in categories:
            co_occurrence_matrix[issue_type][cat] += 1
    
    # Generate statistics
    stats = {
        'total_samples': len(df),
        'unique_issues': len(issue_to_categories),
        'unique_categories': len(category_to_issues),
        'issue_category_mappings': {},
        'category_issue_mappings': {},
        'mapping_statistics': {}
    }
    
    # Process issue to category mappings
    for issue, category_lists in issue_to_categories.items():
        # Flatten and get unique categories for this issue
        all_categories = set()
        for cat_list in category_lists:
            all_categories.update(cat_list)
        
        stats['issue_category_mappings'][issue] = {
            'categories': list(all_categories),
            'frequency': len(category_lists),
            'avg_categories': sum(len(cl) for cl in category_lists) / len(category_lists)
        }
    
    # Calculate mapping statistics
    category_counts = []
    for issue_data in stats['issue_category_mappings'].values():
        category_counts.append(len(issue_data['categories']))
    
    stats['mapping_statistics'] = {
        'avg_categories_per_issue': sum(category_counts) / len(category_counts),
        'max_categories_per_issue': max(category_counts),
        'min_categories_per_issue': min(category_counts),
        'issues_with_single_category': sum(1 for c in category_counts if c == 1),
        'issues_with_multiple_categories': sum(1 for c in category_counts if c > 1)
    }
    
    # Save mapping for use in classification
    mapping_file = Path('data/processed/issue_category_mapping.json')
    mapping_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(mapping_file, 'w') as f:
        json.dump({
            'issue_to_categories': {
                issue: list(cats) 
                for issue, cats in stats['issue_category_mappings'].items()
            },
            'statistics': stats['mapping_statistics']
        }, f, indent=2)
    
    print(f"Issue-Category Mapping Analysis Complete")
    print(f"Total unique issues: {stats['unique_issues']}")
    print(f"Total unique categories: {stats['unique_categories']}")
    print(f"Average categories per issue: {stats['mapping_statistics']['avg_categories_per_issue']:.2f}")
    print(f"Issues with multiple categories: {stats['mapping_statistics']['issues_with_multiple_categories']}")
    print(f"Mapping saved to: {mapping_file}")
    
    return stats

if __name__ == "__main__":
    stats = analyze_issue_category_relationships('data/Consolidated_labeled_data.xlsx')
```

#### Data Sufficiency Analysis Script
```python
# scripts/analyze_data_sufficiency.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from classifier.data_sufficiency import DataSufficiencyAnalyzer

def generate_data_sufficiency_report(data_path: str, output_dir: str = 'reports'):
    """
    Generate comprehensive data sufficiency analysis and visualizations
    """
    # Initialize analyzer
    analyzer = DataSufficiencyAnalyzer(data_path)
    
    # Generate report
    report = analyzer.generate_sufficiency_report()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Print summary
    print("=" * 80)
    print("DATA SUFFICIENCY ANALYSIS REPORT")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  Total Samples: {report['summary']['total_samples']}")
    print(f"  Unique Issue Types: {report['summary']['unique_issue_types']}")
    print(f"  Unique Categories: {report['summary']['unique_categories']}")
    
    # Critical warnings
    if report['critical_issues']:
        print(f"\n⚠️  CRITICAL ISSUES ({len(report['critical_issues'])} issue types with <5 samples):")
        print("-" * 40)
        for item in report['critical_issues'][:10]:  # Show top 10
            print(f"  • {item['issue_type']}: {item['sample_count']} samples")
        if len(report['critical_issues']) > 10:
            print(f"  ... and {len(report['critical_issues']) - 10} more")
    
    if report['critical_categories']:
        print(f"\n⚠️  CRITICAL CATEGORIES ({len(report['critical_categories'])} categories with <5 samples):")
        print("-" * 40)
        for item in report['critical_categories'][:10]:  # Show top 10
            print(f"  • {item['category']}: {item['sample_count']} samples")
        if len(report['critical_categories']) > 10:
            print(f"  ... and {len(report['critical_categories']) - 10} more")
    
    # Warning level
    if report['warning_issues']:
        print(f"\n⚠  WARNING ISSUES ({len(report['warning_issues'])} issue types with 5-10 samples):")
        print("-" * 40)
        for item in report['warning_issues'][:5]:  # Show top 5
            print(f"  • {item['issue_type']}: {item['sample_count']} samples")
    
    if report['warning_categories']:
        print(f"\n⚠  WARNING CATEGORIES ({len(report['warning_categories'])} categories with 5-10 samples):")
        print("-" * 40)
        for item in report['warning_categories'][:5]:  # Show top 5
            print(f"  • {item['category']}: {item['sample_count']} samples")
    
    # Recommendations
    print(f"\n📋 RECOMMENDATIONS:")
    print("-" * 40)
    for rec in report['recommendations']:
        print(f"  • {rec}")
    
    # Generate visualizations
    create_visualizations(analyzer, output_path)
    
    # Save detailed report to JSON
    import json
    with open(output_path / 'data_sufficiency_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate priority list for data collection
    priority_list = generate_priority_list(report)
    with open(output_path / 'data_collection_priorities.txt', 'w') as f:
        f.write("DATA COLLECTION PRIORITY LIST\n")
        f.write("=" * 50 + "\n\n")
        f.write("TOP PRIORITY (Critical - <5 samples):\n")
        f.write("-" * 30 + "\n")
        for item in priority_list['critical']:
            f.write(f"{item}\n")
        f.write("\nMEDIUM PRIORITY (Warning - 5-10 samples):\n")
        f.write("-" * 30 + "\n")
        for item in priority_list['warning']:
            f.write(f"{item}\n")
    
    print(f"\n✅ Report saved to {output_path}")
    print(f"   - Full report: data_sufficiency_report.json")
    print(f"   - Visualizations: *.png")
    print(f"   - Priority list: data_collection_priorities.txt")
    
    return report

def create_visualizations(analyzer, output_path):
    """
    Create data distribution visualizations
    """
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Issue type distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Distribution histogram
    issue_counts = list(analyzer.issue_counts.values())
    ax1.hist(issue_counts, bins=20, edgecolor='black')
    ax1.axvline(5, color='red', linestyle='--', label='Critical (<5)')
    ax1.axvline(10, color='orange', linestyle='--', label='Warning (<10)')
    ax1.axvline(20, color='green', linestyle='--', label='Good (>20)')
    ax1.set_xlabel('Number of Samples')
    ax1.set_ylabel('Number of Issue Types')
    ax1.set_title('Distribution of Training Samples per Issue Type')
    ax1.legend()
    
    # Bottom 20 issue types
    sorted_issues = sorted(analyzer.issue_counts.items(), key=lambda x: x[1])[:20]
    issues, counts = zip(*sorted_issues)
    ax2.barh(range(len(issues)), counts)
    ax2.set_yticks(range(len(issues)))
    ax2.set_yticklabels([i[:30] + '...' if len(i) > 30 else i for i in issues], fontsize=8)
    ax2.axvline(5, color='red', linestyle='--', alpha=0.5)
    ax2.axvline(10, color='orange', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Number of Samples')
    ax2.set_title('Bottom 20 Issue Types by Sample Count')
    
    plt.tight_layout()
    plt.savefig(output_path / 'issue_type_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Category distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Distribution histogram
    category_counts = list(analyzer.category_counts.values())
    ax1.hist(category_counts, bins=20, edgecolor='black')
    ax1.axvline(5, color='red', linestyle='--', label='Critical (<5)')
    ax1.axvline(10, color='orange', linestyle='--', label='Warning (<10)')
    ax1.axvline(20, color='green', linestyle='--', label='Good (>20)')
    ax1.set_xlabel('Number of Samples')
    ax1.set_ylabel('Number of Categories')
    ax1.set_title('Distribution of Training Samples per Category')
    ax1.legend()
    
    # Bottom 20 categories
    sorted_cats = sorted(analyzer.category_counts.items(), key=lambda x: x[1])[:20]
    cats, counts = zip(*sorted_cats)
    ax2.barh(range(len(cats)), counts)
    ax2.set_yticks(range(len(cats)))
    ax2.set_yticklabels([c[:30] + '...' if len(c) > 30 else c for c in cats], fontsize=8)
    ax2.axvline(5, color='red', linestyle='--', alpha=0.5)
    ax2.axvline(10, color='orange', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Number of Samples')
    ax2.set_title('Bottom 20 Categories by Sample Count')
    
    plt.tight_layout()
    plt.savefig(output_path / 'category_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_priority_list(report):
    """
    Generate prioritized list for data collection team
    """
    priority_list = {
        'critical': [],
        'warning': []
    }
    
    # Combine issues and categories, sort by count
    for item in report['critical_issues']:
        priority_list['critical'].append(
            f"Issue: {item['issue_type']} ({item['sample_count']} samples)"
        )
    
    for item in report['critical_categories']:
        priority_list['critical'].append(
            f"Category: {item['category']} ({item['sample_count']} samples)"
        )
    
    for item in report['warning_issues']:
        priority_list['warning'].append(
            f"Issue: {item['issue_type']} ({item['sample_count']} samples)"
        )
    
    for item in report['warning_categories']:
        priority_list['warning'].append(
            f"Category: {item['category']} ({item['sample_count']} samples)"
        )
    
    return priority_list

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze data sufficiency for classification')
    parser.add_argument('--data', type=str, default='data/Consolidated_labeled_data.xlsx',
                       help='Path to training data')
    parser.add_argument('--output', type=str, default='reports',
                       help='Output directory for reports')
    
    args = parser.parse_args()
    
    report = generate_data_sufficiency_report(args.data, args.output)
```

#### OCR Processing for Scanned PDFs
```python
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import pdfplumber

def extract_text_from_pdf(pdf_path):
    """
    Extract text from PDF, using OCR if necessary
    """
    # First try regular text extraction
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            # Check if we got meaningful text
            if len(text.strip()) > 100:
                return text, "text_extraction"
    except Exception as e:
        print(f"Regular extraction failed: {e}")
    
    # Fall back to OCR for scanned PDFs
    try:
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=300)
        
        text = ""
        for i, image in enumerate(images):
            # Apply OCR to each page
            page_text = pytesseract.image_to_string(image)
            text += f"Page {i+1}:\n{page_text}\n"
        
        return text, "ocr"
    except Exception as e:
        raise Exception(f"Both text extraction and OCR failed: {e}")

def preprocess_ocr_text(text):
    """
    Clean up OCR text artifacts
    """
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Fix common OCR errors
    replacements = {
        '|': 'I',  # Common OCR mistake
        '0': 'O',  # In certain contexts
        # Add more based on observed patterns
    }
    
    for old, new in replacements.items():
        # Context-aware replacement would be better
        pass
    
    return text
```

#### Sliding Window Processing
```python
def create_sliding_windows(text, window_size=3, overlap=1):
    sentences = nltk.sent_tokenize(text)
    windows = []
    stride = window_size - overlap
    
    for i in range(0, len(sentences), stride):
        window = sentences[i:i+window_size]
        windows.append(' '.join(window))
        if i + window_size >= len(sentences):
            break
    
    return windows
```

#### Multi-Label Classification Metrics
```python
from sklearn.metrics import classification_report

def calculate_multilabel_metrics(y_true, y_pred):
    report = classification_report(
        y_true, y_pred,
        target_names=category_names,
        output_dict=True,
        zero_division=0
    )
    return report
```

#### Ensemble Voting
```python
def ensemble_vote(predictions, weights=None):
    if weights is None:
        weights = [1/len(predictions)] * len(predictions)
    
    category_scores = {}
    for pred, weight in zip(predictions, weights):
        for category in pred:
            if category not in category_scores:
                category_scores[category] = 0
            category_scores[category] += weight
    
    threshold = 0.5
    final_categories = [cat for cat, score in category_scores.items() 
                       if score >= threshold]
    return final_categories
```

This implementation should result in a production-ready system that can be immediately integrated into the existing CCMS application, with the flexibility to choose the best approach based on performance metrics.