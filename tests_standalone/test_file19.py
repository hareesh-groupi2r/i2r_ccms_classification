#!/usr/bin/env python3
"""
Quick test for File 19 processing to debug the OpenAI client issue
"""
import os
import sys

# Add project root to path
sys.path.append('/Users/hareeshkb/work/Krishna/ccms_classification')

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

print("Environment variables:")
print(f"CLAUDE_API_KEY set: {bool(os.getenv('CLAUDE_API_KEY'))}")
print(f"OPENAI_API_KEY set: {bool(os.getenv('OPENAI_API_KEY'))}")

try:
    # Test Anthropic client
    import anthropic
    anthropic_client = anthropic.Anthropic(api_key=os.getenv('CLAUDE_API_KEY'))
    print("✓ Anthropic client initialized successfully")
except Exception as e:
    print(f"✗ Anthropic client error: {e}")

try:
    # Test OpenAI client
    from openai import OpenAI
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    print("✓ OpenAI client initialized successfully")
except Exception as e:
    print(f"✗ OpenAI client error: {e}")

print("\nTesting HybridRAGClassifier initialization...")
try:
    from classifier.hybrid_rag import HybridRAGClassifier
    from classifier.issue_mapper import IssueCategoryMapper
    from classifier.validation import ValidationEngine
    from classifier.data_sufficiency import DataSufficiencyAnalyzer
    import yaml
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize components (simplified)
    issue_mapper = IssueCategoryMapper('data/raw/Consolidated_labeled_data.xlsx')
    validator = ValidationEngine(issue_mapper)
    data_analyzer = DataSufficiencyAnalyzer('data/raw/Consolidated_labeled_data.xlsx')
    
    # Try to initialize hybrid classifier
    hybrid_config = config['approaches']['hybrid_rag']
    classifier = HybridRAGClassifier(hybrid_config, issue_mapper, validator, data_analyzer)
    print("✓ HybridRAGClassifier initialized successfully")
    
    print(f"Primary LLM provider: {classifier.current_llm_provider}")
    print(f"Fallback clients: {list(classifier.fallback_llm_clients.keys())}")
    
except Exception as e:
    print(f"✗ HybridRAGClassifier initialization error: {e}")
    import traceback
    traceback.print_exc()