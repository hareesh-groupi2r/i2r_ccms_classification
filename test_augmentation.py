#!/usr/bin/env python3
"""
Test Text Augmentation for Synthetic Data Generation
Demonstrates various augmentation techniques without requiring API calls
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_augmentation():
    """Test basic augmentation techniques that don't require external APIs."""
    print("=" * 60)
    print("TEXT AUGMENTATION DEMO")
    print("=" * 60)
    
    # Load some sample data
    try:
        df = pd.read_excel('data/raw/Consolidated_labeled_data.xlsx')
        sample_texts = df['subject'].dropna().head(3).tolist()
    except:
        # Fallback sample texts
        sample_texts = [
            "Request for extension of time for completion of project",
            "Payment delay notification for contract work",
            "Change of scope approval required for additional work"
        ]
    
    print("Original texts:")
    for i, text in enumerate(sample_texts, 1):
        print(f"{i}. {text}")
    
    print("\n" + "=" * 60)
    print("AUGMENTATION TECHNIQUES")
    print("=" * 60)
    
    # 1. Synonym Replacement (basic)
    print("\n1. SYNONYM REPLACEMENT (Basic):")
    print("-" * 40)
    
    synonym_mapping = {
        'request': ['application', 'petition', 'appeal'],
        'extension': ['prolongation', 'continuation', 'expansion'],
        'completion': ['finishing', 'conclusion', 'accomplishment'],
        'project': ['undertaking', 'venture', 'initiative'],
        'payment': ['remuneration', 'settlement', 'disbursement'],
        'delay': ['postponement', 'deferral', 'hold-up'],
        'notification': ['notice', 'announcement', 'communication'],
        'contract': ['agreement', 'covenant', 'arrangement'],
        'work': ['task', 'assignment', 'job'],
        'change': ['modification', 'alteration', 'amendment'],
        'scope': ['extent', 'range', 'breadth'],
        'approval': ['authorization', 'consent', 'permission'],
        'required': ['needed', 'necessary', 'essential'],
        'additional': ['extra', 'supplementary', 'further']
    }
    
    import random
    
    def simple_synonym_replacement(text, replacement_prob=0.3):
        words = text.split()
        augmented_words = []
        
        for word in words:
            word_lower = word.lower().strip('.,!?;:')
            if word_lower in synonym_mapping and random.random() < replacement_prob:
                synonym = random.choice(synonym_mapping[word_lower])
                # Preserve capitalization
                if word[0].isupper():
                    synonym = synonym.capitalize()
                augmented_words.append(synonym)
            else:
                augmented_words.append(word)
        
        return ' '.join(augmented_words)
    
    for i, text in enumerate(sample_texts, 1):
        augmented = simple_synonym_replacement(text)
        print(f"Original:  {text}")
        print(f"Augmented: {augmented}")
        print()
    
    # 2. Entity Substitution
    print("2. ENTITY SUBSTITUTION:")
    print("-" * 40)
    
    entities = {
        'projects': ['highway construction', 'bridge development', 'tunnel excavation', 'flyover installation'],
        'amounts': ['Rs. 50,00,000', 'Rs. 1,25,00,000', 'Rs. 75,50,000', 'Rs. 2,00,00,000'],
        'durations': ['3 months', '6 months', '4 weeks', '8 weeks'],
        'departments': ['Authority', 'Engineering Department', 'Project Office', 'Contract Division']
    }
    
    def entity_substitution(text):
        import re
        
        # Replace project references
        text = re.sub(r'\\bproject\\b', random.choice(entities['projects']), text, flags=re.IGNORECASE)
        
        # Replace amount patterns (if any)
        text = re.sub(r'Rs\\.\\s*\\d+(?:,\\d+)*', random.choice(entities['amounts']), text)
        
        # Replace duration patterns
        text = re.sub(r'\\d+\\s+(?:months?|weeks?|days?)', random.choice(entities['durations']), text)
        
        return text
    
    for i, text in enumerate(sample_texts, 1):
        augmented = entity_substitution(text)
        print(f"Original:  {text}")
        print(f"Augmented: {augmented}")
        print()
    
    # 3. Sentence Structure Variation
    print("3. SENTENCE STRUCTURE VARIATION:")
    print("-" * 40)
    
    def structure_variation(text):
        # Simple transformations
        variations = []
        
        # Add formal prefixes
        prefixes = [
            "This is to inform you that ",
            "Please note that ",
            "We would like to bring to your attention that ",
            "Kindly be informed that "
        ]
        
        # Add formal suffixes
        suffixes = [
            ". Please take necessary action.",
            ". We request your immediate attention.",
            ". Kindly provide your response at the earliest.",
            ". We await your further instructions."
        ]
        
        # Create variations
        variation1 = random.choice(prefixes) + text.lower()
        variation2 = text + random.choice(suffixes)
        variation3 = random.choice(prefixes) + text.lower() + random.choice(suffixes)
        
        return [variation1, variation2, variation3]
    
    for i, text in enumerate(sample_texts[:1], 1):  # Show for first sample only
        variations = structure_variation(text)
        print(f"Original: {text}")
        for j, variation in enumerate(variations, 1):
            print(f"Variation {j}: {variation}")
        print()
    
    # 4. Template-based Generation
    print("4. TEMPLATE-BASED GENERATION:")
    print("-" * 40)
    
    templates = {
        'payment': [
            "Subject: {payment_type} Request - {project_ref}\\nWe request immediate processing of {payment_type} for {work_description}. Amount: {amount}.",
            "Subject: Delayed {payment_type} - {project_ref}\\nThis is regarding the delayed {payment_type} of {amount} for {work_description}.",
        ],
        'extension': [
            "Subject: Extension Request - {project_ref}\\nWe request an extension of {duration} for {project_ref} due to {reason}.",
            "Subject: Time Extension - {project_ref}\\nExtension of {duration} is required for completion of {project_ref}.",
        ],
        'scope': [
            "Subject: Scope Change - {project_ref}\\nChange of scope is required for {project_ref} involving {change_description}.",
            "Subject: Additional Work - {project_ref}\\nAdditional work amounting to {amount} is required for {project_ref}.",
        ]
    }
    
    # Generate samples using templates
    template_vars = {
        'payment_type': ['advance payment', 'running account bill', 'final payment'],
        'project_ref': ['Project-1001', 'NH-45 Development', 'Bridge Construction Phase-2'],
        'work_description': ['civil construction work', 'structural development', 'infrastructure setup'],
        'amount': ['Rs. 15,00,000', 'Rs. 25,50,000', 'Rs. 8,75,000'],
        'duration': ['2 months', '45 days', '3 months'],
        'reason': ['unforeseen site conditions', 'weather delays', 'material shortage'],
        'change_description': ['additional structural work', 'modified design requirements', 'expanded scope']
    }
    
    def generate_from_template(template_type, count=2):
        template_list = templates.get(template_type, templates['payment'])
        generated = []
        
        for i in range(count):
            template = random.choice(template_list)
            
            # Fill template with random values
            filled = template.format(
                payment_type=random.choice(template_vars['payment_type']),
                project_ref=random.choice(template_vars['project_ref']),
                work_description=random.choice(template_vars['work_description']),
                amount=random.choice(template_vars['amount']),
                duration=random.choice(template_vars['duration']),
                reason=random.choice(template_vars['reason']),
                change_description=random.choice(template_vars['change_description'])
            )
            generated.append(filled)
        
        return generated
    
    for template_type in ['payment', 'extension', 'scope']:
        print(f"\\n{template_type.upper()} Templates:")
        samples = generate_from_template(template_type, 2)
        for j, sample in enumerate(samples, 1):
            print(f"  Sample {j}: {sample}")
    
    print("\\n" + "=" * 60)
    print("✅ AUGMENTATION DEMO COMPLETE")
    print("=" * 60)
    
    print("\\nKey Takeaways:")
    print("• Synonym replacement adds vocabulary variation")
    print("• Entity substitution creates realistic variations")
    print("• Structure variation changes sentence patterns")
    print("• Template-based generation ensures domain relevance")
    print("• Combine techniques for maximum diversity")
    
    print("\\nNext Steps:")
    print("• Install nlpaug: pip install nlpaug")
    print("• Use OpenAI API for high-quality generation")
    print("• Implement quality control metrics")
    print("• Create validation splits to prevent overfitting")


if __name__ == "__main__":
    test_basic_augmentation()