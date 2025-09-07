#!/usr/bin/env python3
"""
Synthetic Data Generator for Contract Correspondence Classification
Generates synthetic training data to address data scarcity while preventing overfitting
"""

import sys
from pathlib import Path
import pandas as pd
import random
import json
import re
from typing import List, Dict, Tuple
import openai
from openai import OpenAI
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from classifier.data_sufficiency import DataSufficiencyAnalyzer
from classifier.issue_normalizer import IssueTypeNormalizer
from classifier.category_normalizer import CategoryNormalizer


class SyntheticDataGenerator:
    """
    Generates synthetic contract correspondence data using multiple techniques:
    1. LLM-based generation with domain-specific prompts
    2. Template-based variation with paraphrasing
    3. Entity substitution and augmentation
    4. Cross-validation to prevent overfitting
    """
    
    def __init__(self, training_data_path: str, openai_api_key: str = None):
        """
        Initialize the synthetic data generator.
        
        Args:
            training_data_path: Path to existing training data
            openai_api_key: OpenAI API key for LLM generation
        """
        self.training_data_path = training_data_path
        self.df = pd.read_excel(training_data_path)
        
        # Initialize normalizers
        self.issue_normalizer = IssueTypeNormalizer()
        self.category_normalizer = CategoryNormalizer()
        
        # Initialize data analyzer
        self.analyzer = DataSufficiencyAnalyzer(training_data_path)
        
        # Initialize OpenAI client if API key provided
        self.openai_client = None
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Load existing data for analysis
        self._analyze_existing_data()
        
        # Contract domain vocabulary and entities
        self._load_domain_vocabulary()
    
    def _analyze_existing_data(self):
        """Analyze existing data to identify critical issues needing synthetic data."""
        # First normalize the data before analysis
        print("Normalizing categories in training data before analysis...")
        self._normalize_training_data()
        
        report = self.analyzer.generate_sufficiency_report()
        
        # Identify critical issues (< 5 samples)
        self.critical_issues = [(item['issue_type'], item['sample_count']) 
                               for item in report['critical_issues']]
        
        # Identify warning issues (5-10 samples)
        self.warning_issues = [(item['issue_type'], item['sample_count']) 
                              for item in report['warning_issues']]
        
        print(f"Found {len(self.critical_issues)} critical issues and {len(self.warning_issues)} warning issues")
    
    def _normalize_training_data(self):
        """Normalize categories in training data to the 8 standard categories."""
        normalized_categories = []
        
        for idx, row in self.df.iterrows():
            # Normalize issue type
            issue_type_raw = str(row.get('issue_type', ''))
            issue_type, _, _ = self.issue_normalizer.normalize_issue_type(issue_type_raw)
            self.df.at[idx, 'issue_type'] = issue_type
            
            # Normalize categories
            category_raw = str(row.get('category', ''))
            normalized_cats = self.category_normalizer.parse_and_normalize_categories(category_raw)
            
            # Join normalized categories or use the first one
            if normalized_cats:
                # For synthetic data generation, use single category (first normalized one)
                normalized_category = normalized_cats[0]
                self.df.at[idx, 'category'] = normalized_category
            else:
                # Fallback to 'Others' if normalization fails
                self.df.at[idx, 'category'] = 'Others'
        
        # Update the analyzer with normalized data
        self.analyzer.df = self.df.copy()
        
        print(f"Normalized categories to {self.df['category'].nunique()} unique categories")
    
    def _load_domain_vocabulary(self):
        """Load contract domain-specific vocabulary and entities."""
        
        # Contract terminology
        self.contract_terms = [
            "Agreement", "Contract", "EPC Agreement", "Purchase Order", "Work Order",
            "Addendum", "Amendment", "Variation Order", "Change Order", "Supplemental Agreement"
        ]
        
        # Project entities
        self.project_entities = [
            "Highway", "Bridge", "Tunnel", "Flyover", "Underpass", "Interchange",
            "Road", "Project", "Construction", "Infrastructure", "Development"
        ]
        
        # Authority entities
        self.authority_entities = [
            "Authority", "Engineer", "Project Manager", "Consultant", "Supervisor",
            "Client", "Employer", "Principal", "Owner", "Department"
        ]
        
        # Contractor entities  
        self.contractor_entities = [
            "Contractor", "Subcontractor", "Vendor", "Supplier", "Service Provider",
            "Agency", "Company", "Firm", "Organization", "Entity"
        ]
        
        # Time-related terms
        self.time_terms = [
            "extension", "delay", "postponement", "acceleration", "suspension",
            "completion", "commencement", "milestone", "deadline", "schedule"
        ]
        
        # Financial terms
        self.financial_terms = [
            "payment", "invoice", "billing", "cost", "price", "amount", "sum",
            "compensation", "reimbursement", "advance", "retention", "deduction"
        ]
        
        # Technical terms
        self.technical_terms = [
            "specification", "design", "drawing", "plan", "method", "procedure",
            "quality", "testing", "inspection", "approval", "clearance", "permit"
        ]
    
    def generate_llm_synthetic_data(self, issue_type: str, category: str, 
                                   current_samples: int, target_samples: int = 10) -> List[Dict]:
        """
        Generate synthetic data using LLM with domain-specific prompts.
        
        Args:
            issue_type: The issue type to generate data for
            category: The associated category
            current_samples: Current number of samples
            target_samples: Target number of samples to generate
            
        Returns:
            List of synthetic data samples
        """
        if not self.openai_client:
            print("OpenAI client not initialized. Skipping LLM generation.")
            return []
        
        samples_to_generate = target_samples - current_samples
        if samples_to_generate <= 0:
            return []
        
        print(f"Generating {samples_to_generate} synthetic samples for: {issue_type}")
        
        # Get existing samples for this issue type for context
        existing_samples = self.df[self.df['issue_type'] == issue_type]
        
        # Create context from existing samples
        context_examples = []
        if len(existing_samples) > 0:
            for _, sample in existing_samples.head(3).iterrows():  # Use up to 3 examples
                context_examples.append({
                    'subject': str(sample.get('subject', '')),
                    'body': str(sample.get('body', ''))[:200] + "..."  # Truncate for prompt
                })
        
        # Create domain-specific prompt
        prompt = self._create_generation_prompt(issue_type, category, context_examples)
        
        synthetic_samples = []
        
        # Generate samples in batches to avoid rate limits
        batch_size = 3
        for i in range(0, samples_to_generate, batch_size):
            current_batch_size = min(batch_size, samples_to_generate - i)
            
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert in contract management and construction project correspondence. Generate realistic contract correspondence samples."},
                        {"role": "user", "content": prompt.replace("{num_samples}", str(current_batch_size))}
                    ],
                    temperature=0.8,  # Higher temperature for variety
                    max_tokens=2000,
                    response_format={"type": "json_object"}
                )
                
                # Parse response
                generated_data = json.loads(response.choices[0].message.content)
                
                # Process generated samples
                for sample in generated_data.get('samples', []):
                    synthetic_samples.append({
                        'issue_type': issue_type,
                        'category': category,
                        'subject': sample.get('subject', ''),
                        'body': sample.get('body', ''),
                        'reference_sentence': sample.get('key_phrase', ''),
                        'source_file': f'synthetic_llm_{datetime.now().strftime("%Y%m%d")}',
                        'generation_method': 'llm_gpt4',
                        'is_synthetic': True
                    })
                
                # Rate limiting to avoid OpenAI rate limits
                # GPT-4 has 500 RPM limit, so 2-3 second delay is safe
                print(f"  Batch {i//batch_size + 1} completed. Waiting 3 seconds...")
                time.sleep(3)
                
            except Exception as e:
                print(f"Error generating batch {i//batch_size + 1}: {e}")
                continue
        
        print(f"Successfully generated {len(synthetic_samples)} samples for {issue_type}")
        return synthetic_samples
    
    def _create_generation_prompt(self, issue_type: str, category: str, 
                                 context_examples: List[Dict]) -> str:
        """Create a domain-specific prompt for LLM generation."""
        
        # Build context from existing samples
        context_text = ""
        if context_examples:
            context_text = "\\nExisting examples for reference (DO NOT copy exactly):\\n"
            for i, example in enumerate(context_examples, 1):
                context_text += f"Example {i}:\\n"
                context_text += f"Subject: {example['subject']}\\n"
                context_text += f"Body: {example['body']}\\n\\n"
        
        prompt = f"""Generate {"{num_samples}"} diverse and realistic contract correspondence samples for construction/infrastructure projects.

Issue Type: "{issue_type}"
Category: "{category}"

Requirements:
1. Create varied correspondence types (letters, emails, notices, reports)
2. Use realistic project scenarios and entities
3. Include proper contract terminology
4. Vary the writing style, formality, and length
5. Ensure each sample clearly relates to the specified issue type
6. Avoid copying existing examples exactly
7. Include realistic dates, reference numbers, and project details

{context_text}

Domain Context:
- Infrastructure/construction projects (highways, bridges, buildings)
- Contract management scenarios
- Authority-contractor communications
- Technical and commercial issues
- Regulatory and compliance matters

Return JSON format:
{{
  "samples": [
    {{
      "subject": "Email/letter subject line",
      "body": "Full correspondence body (200-500 words)",
      "key_phrase": "Key phrase that indicates this issue type"
    }}
  ]
}}

Focus on variety and realism. Each sample should be distinctly different."""
        
        return prompt
    
    def generate_template_based_data(self, issue_type: str, category: str, 
                                   current_samples: int, target_samples: int = 10) -> List[Dict]:
        """
        Generate synthetic data using template-based variation.
        
        Args:
            issue_type: The issue type to generate data for
            category: The associated category  
            current_samples: Current number of samples
            target_samples: Target number of samples to generate
            
        Returns:
            List of synthetic data samples
        """
        samples_to_generate = target_samples - current_samples
        if samples_to_generate <= 0:
            return []
        
        print(f"Generating {samples_to_generate} template-based samples for: {issue_type}")
        
        # Get existing samples for this issue type
        existing_samples = self.df[self.df['issue_type'] == issue_type]
        
        if len(existing_samples) == 0:
            print(f"No existing samples found for {issue_type}. Using generic templates.")
            return self._generate_generic_templates(issue_type, category, samples_to_generate)
        
        synthetic_samples = []
        
        # Create variations of existing samples
        for i in range(samples_to_generate):
            # Select a random existing sample as base
            base_sample = existing_samples.sample(1).iloc[0]
            
            # Create variation
            varied_sample = self._create_template_variation(base_sample, issue_type, category)
            synthetic_samples.append(varied_sample)
        
        return synthetic_samples
    
    def _create_template_variation(self, base_sample: pd.Series, issue_type: str, category: str) -> Dict:
        """Create a variation of an existing sample using templates and entity substitution."""
        
        original_subject = str(base_sample.get('subject', ''))
        original_body = str(base_sample.get('body', ''))
        
        # Entity substitution patterns
        substitutions = {
            # Project references
            r'\\bProject\\b': random.choice(self.project_entities),
            r'\\bHighway\\b': random.choice(['Highway', 'Road', 'Expressway', 'Corridor']),
            r'\\bBridge\\b': random.choice(['Bridge', 'Flyover', 'Overpass', 'Viaduct']),
            
            # Authority references  
            r'\\bAuthority\\b': random.choice(self.authority_entities),
            r'\\bEngineer\\b': random.choice(['Engineer', 'Project Manager', 'Consultant', 'Supervisor']),
            
            # Contractor references
            r'\\bContractor\\b': random.choice(self.contractor_entities),
            
            # Time references
            r'\\b\\d{1,2}\\s+days?\\b': f"{random.randint(1, 30)} days",
            r'\\b\\d{1,2}\\s+weeks?\\b': f"{random.randint(1, 12)} weeks",
            r'\\b\\d{1,2}\\s+months?\\b': f"{random.randint(1, 6)} months",
            
            # Financial amounts (preserve format but change values)
            r'\\bRs\\.\\s*\\d+(?:,\\d+)*(?:\\.\\d+)?\\b': f"Rs. {random.randint(100000, 10000000):,}",
            r'\\b\\d+(?:,\\d+)*(?:\\.\\d+)?\\s*lakhs?\\b': f"{random.randint(10, 500)} lakhs",
            r'\\b\\d+(?:,\\d+)*(?:\\.\\d+)?\\s*crores?\\b': f"{random.randint(1, 50)} crores",
        }
        
        # Apply substitutions to subject and body
        varied_subject = original_subject
        varied_body = original_body
        
        for pattern, replacement in substitutions.items():
            varied_subject = re.sub(pattern, replacement, varied_subject, flags=re.IGNORECASE)
            varied_body = re.sub(pattern, replacement, varied_body, flags=re.IGNORECASE)
        
        # Add some stylistic variations
        varied_body = self._add_stylistic_variations(varied_body)
        
        return {
            'issue_type': issue_type,
            'category': category,
            'subject': varied_subject,
            'body': varied_body,
            'reference_sentence': str(base_sample.get('reference_sentence', '')),
            'source_file': f'synthetic_template_{datetime.now().strftime("%Y%m%d")}',
            'generation_method': 'template_variation',
            'is_synthetic': True
        }
    
    def _add_stylistic_variations(self, text: str) -> str:
        """Add stylistic variations to text to increase diversity."""
        
        # Sentence starter variations
        starters = [
            "Please note that", "Kindly be informed that", "We would like to inform you that",
            "This is to notify you that", "We hereby confirm that", "It is brought to your attention that"
        ]
        
        # Formal closings
        closings = [
            "We look forward to your response.", "Please confirm receipt of this correspondence.",
            "We await your further instructions.", "Thank you for your attention to this matter.",
            "We appreciate your cooperation in this regard."
        ]
        
        # Randomly add formal elements
        if random.random() < 0.3:  # 30% chance
            text = random.choice(starters) + " " + text
        
        if random.random() < 0.3:  # 30% chance  
            text = text + " " + random.choice(closings)
        
        return text
    
    def _generate_generic_templates(self, issue_type: str, category: str, count: int) -> List[Dict]:
        """Generate samples using generic templates when no existing data is available."""
        
        templates = {
            'payment': [
                "Subject: Request for {payment_type} - {project_ref}\\nDear Sir/Madam,\\nWe request processing of {payment_type} for work completed under {project_ref}. Amount: Rs. {amount}. Please expedite.",
                "Subject: Payment Delay - {project_ref}\\nThis is regarding the delayed payment of Rs. {amount} for {work_description} under {project_ref}. Immediate attention required."
            ],
            'scope': [
                "Subject: Change of Scope Request - {project_ref}\\nWe propose changes to the scope of work under {project_ref} due to {reason}. Please review and approve.",
                "Subject: Additional Work Requirement - {project_ref}\\nAdditional work is required for {work_description} under {project_ref}. Estimated cost: Rs. {amount}."
            ],
            'time': [
                "Subject: Extension of Time Request - {project_ref}\\nWe request extension of {duration} for completion of {project_ref} due to {reason}.",
                "Subject: Project Schedule Revision - {project_ref}\\nRevised schedule for {project_ref} is submitted herewith due to {reason}."
            ],
            'default': [
                "Subject: {issue_type} - {project_ref}\\nDear Sir/Madam,\\nThis is regarding {issue_type} under {project_ref}. Please take necessary action.",
            ]
        }
        
        # Determine template category
        template_key = 'default'
        if 'payment' in issue_type.lower():
            template_key = 'payment'
        elif 'scope' in issue_type.lower():
            template_key = 'scope'
        elif 'time' in issue_type.lower() or 'extension' in issue_type.lower():
            template_key = 'time'
        
        selected_templates = templates.get(template_key, templates['default'])
        synthetic_samples = []
        
        for i in range(count):
            template = random.choice(selected_templates)
            
            # Fill template variables
            filled_template = template.format(
                payment_type=random.choice(['advance payment', 'running bill', 'final payment']),
                project_ref=f"Project-{random.randint(1000, 9999)}",
                amount=f"{random.randint(100000, 10000000):,}",
                work_description=random.choice(['construction work', 'civil work', 'infrastructure development']),
                reason=random.choice(['unforeseen circumstances', 'weather conditions', 'material shortage']),
                duration=f"{random.randint(1, 6)} months",
                issue_type=issue_type
            )
            
            # Split into subject and body
            parts = filled_template.split('\\n', 1)
            subject = parts[0].replace('Subject: ', '')
            body = parts[1] if len(parts) > 1 else filled_template
            
            synthetic_samples.append({
                'issue_type': issue_type,
                'category': category,
                'subject': subject,
                'body': body,
                'reference_sentence': issue_type,
                'source_file': f'synthetic_generic_{datetime.now().strftime("%Y%m%d")}',
                'generation_method': 'generic_template',
                'is_synthetic': True
            })
        
        return synthetic_samples
    
    def generate_synthetic_dataset(self, target_min_samples: int = 10, 
                                 use_llm: bool = True, use_templates: bool = True) -> pd.DataFrame:
        """
        Generate comprehensive synthetic dataset to address data scarcity.
        
        Args:
            target_min_samples: Minimum samples to aim for per issue type
            use_llm: Whether to use LLM generation
            use_templates: Whether to use template-based generation
            
        Returns:
            DataFrame with synthetic samples
        """
        print("=" * 80)
        print("GENERATING SYNTHETIC DATA FOR CRITICAL ISSUE TYPES")
        print("=" * 80)
        
        all_synthetic_samples = []
        
        # Process critical issues first (highest priority)
        print(f"\\n🔴 Processing {len(self.critical_issues)} critical issue types...")
        
        for issue_type, current_count in self.critical_issues:
            # Get the category for this issue type
            issue_samples = self.df[self.df['issue_type'] == issue_type]
            if len(issue_samples) == 0:
                continue
                
            # Get the most common category for this issue type (already normalized)
            categories = issue_samples['category'].tolist()
            
            if not categories:
                continue
                
            # Use the most common normalized category
            most_common_category = max(set(categories), key=categories.count)
            
            print(f"\\nProcessing: {issue_type} (Category: {most_common_category})")
            print(f"  Current samples: {current_count}, Target: {target_min_samples}")
            
            # Generate using LLM
            if use_llm:
                llm_samples = self.generate_llm_synthetic_data(
                    issue_type, most_common_category, current_count, target_min_samples
                )
                all_synthetic_samples.extend(llm_samples)
                current_count += len(llm_samples)
            
            # Generate using templates if still needed
            if use_templates and current_count < target_min_samples:
                template_samples = self.generate_template_based_data(
                    issue_type, most_common_category, current_count, target_min_samples
                )
                all_synthetic_samples.extend(template_samples)
        
        # Process warning issues (medium priority)
        print(f"\\n🟡 Processing {len(self.warning_issues)} warning issue types...")
        
        for issue_type, current_count in self.warning_issues[:10]:  # Limit to top 10
            issue_samples = self.df[self.df['issue_type'] == issue_type]
            if len(issue_samples) == 0:
                continue
                
            # Get normalized category for this issue type (already normalized)
            categories = issue_samples['category'].tolist()
            
            if not categories:
                continue
                
            # Use the most common normalized category
            most_common_category = max(set(categories), key=categories.count)
            
            # Only generate a few samples for warning issues
            target_samples = min(target_min_samples, current_count + 3)
            
            if use_templates:  # Use templates for warning issues to save API costs
                template_samples = self.generate_template_based_data(
                    issue_type, most_common_category, current_count, target_samples
                )
                all_synthetic_samples.extend(template_samples)
        
        # Create DataFrame
        synthetic_df = pd.DataFrame(all_synthetic_samples)
        
        print(f"\\n✅ Generated {len(all_synthetic_samples)} synthetic samples")
        print(f"   LLM-generated: {len([s for s in all_synthetic_samples if s['generation_method'] == 'llm_gpt4'])}")
        print(f"   Template-based: {len([s for s in all_synthetic_samples if s['generation_method'] != 'llm_gpt4'])}")
        
        return synthetic_df
    
    def create_overfitting_prevention_splits(self, synthetic_df: pd.DataFrame, 
                                           validation_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create training/validation splits to prevent overfitting with synthetic data.
        
        Args:
            synthetic_df: DataFrame with synthetic samples
            validation_ratio: Ratio of data to use for validation
            
        Returns:
            Tuple of (training_df, validation_df)
        """
        print("\\n📊 Creating overfitting prevention splits...")
        
        # Ensure we have both real and synthetic data in training and validation
        real_df = self.df.copy()
        real_df['is_synthetic'] = False
        real_df['generation_method'] = 'real_data'
        
        # Combine real and synthetic data
        combined_df = pd.concat([real_df, synthetic_df], ignore_index=True)
        
        # Stratified split by issue type to ensure representation
        train_samples = []
        val_samples = []
        
        for issue_type in combined_df['issue_type'].unique():
            issue_samples = combined_df[combined_df['issue_type'] == issue_type]
            
            # Calculate split
            val_size = max(1, int(len(issue_samples) * validation_ratio))
            
            # Randomly sample for validation
            val_indices = issue_samples.sample(n=val_size, random_state=42).index
            
            val_samples.extend(val_indices)
            train_indices = issue_samples.index.difference(val_indices)
            train_samples.extend(train_indices)
        
        train_df = combined_df.loc[train_samples].copy()
        val_df = combined_df.loc[val_samples].copy()
        
        print(f"Training set: {len(train_df)} samples ({train_df['is_synthetic'].sum()} synthetic)")
        print(f"Validation set: {len(val_df)} samples ({val_df['is_synthetic'].sum()} synthetic)")
        
        return train_df, val_df
    
    def save_synthetic_data(self, synthetic_df: pd.DataFrame, output_path: str):
        """Save synthetic data to file with metadata."""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add generation metadata
        synthetic_df['generated_date'] = datetime.now().isoformat()
        synthetic_df['generator_version'] = '1.0'
        
        # Save to Excel
        synthetic_df.to_excel(output_path, index=False)
        
        # Save generation summary
        summary = {
            'generation_date': datetime.now().isoformat(),
            'total_samples_generated': len(synthetic_df),
            'generation_methods': synthetic_df['generation_method'].value_counts().to_dict(),
            'issue_types_covered': synthetic_df['issue_type'].nunique(),
            'categories_covered': synthetic_df['category'].nunique(),
            'samples_by_issue_type': synthetic_df['issue_type'].value_counts().to_dict()
        }
        
        summary_path = output_path.with_suffix('.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\\n💾 Synthetic data saved to: {output_path}")
        print(f"📋 Generation summary saved to: {summary_path}")


def main():
    """Main function to demonstrate synthetic data generation."""
    import os
    
    print("=" * 80)
    print("SYNTHETIC DATA GENERATION DEMO")
    print("=" * 80)
    
    # Check for OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key.startswith('your-'):
        print("⚠️  OpenAI API key not found. Will use template-based generation only.")
        use_llm = False
    else:
        print("✅ OpenAI API key found. Will use LLM + template generation.")
        use_llm = True
    
    # Initialize generator
    generator = SyntheticDataGenerator(
        training_data_path='data/raw/Consolidated_labeled_data.xlsx',
        openai_api_key=api_key if use_llm else None
    )
    
    # Generate synthetic data (small sample for demo)
    synthetic_df = generator.generate_synthetic_dataset(
        target_min_samples=8,  # Small target for demo
        use_llm=use_llm,
        use_templates=True
    )
    
    if len(synthetic_df) > 0:
        # Show sample of generated data
        print("\\n📋 Sample of generated data:")
        print("-" * 60)
        
        for i, (_, sample) in enumerate(synthetic_df.head(3).iterrows(), 1):
            print(f"Sample {i}:")
            print(f"  Issue Type: {sample['issue_type']}")
            print(f"  Category: {sample['category']}")  
            print(f"  Subject: {sample['subject'][:80]}...")
            print(f"  Method: {sample['generation_method']}")
            print()
        
        # Save synthetic data
        output_path = './data/synthetic/synthetic_training_data.xlsx'
        generator.save_synthetic_data(synthetic_df, output_path)
        
        # Create overfitting prevention splits
        train_df, val_df = generator.create_overfitting_prevention_splits(synthetic_df)
        
        # Save splits
        train_df.to_excel('./data/synthetic/training_with_synthetic.xlsx', index=False)
        val_df.to_excel('./data/synthetic/validation_with_synthetic.xlsx', index=False)
        
        print("✅ Synthetic data generation complete!")
    else:
        print("❌ No synthetic data was generated.")


if __name__ == "__main__":
    main()