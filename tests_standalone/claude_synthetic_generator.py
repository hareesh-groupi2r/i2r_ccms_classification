#!/usr/bin/env python3
"""
Claude-based Synthetic Data Generator for CCMS Classification
Uses Anthropic Claude API to generate realistic contract correspondence samples
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import anthropic
except ImportError:
    print("❌ Anthropic library not installed. Install with: pip install anthropic")
    sys.exit(1)

from classifier.unified_issue_mapper import UnifiedIssueCategoryMapper
from classifier.data_sufficiency import DataSufficiencyAnalyzer


class ClaudeSyntheticGenerator:
    """
    Generate synthetic training data using Claude API
    """
    
    def __init__(self, training_path: str, mapping_path: str, api_key: str = None):
        """Initialize the Claude-based generator"""
        self.training_path = training_path
        self.mapping_path = mapping_path
        
        # Initialize Anthropic client
        self.api_key = api_key or os.getenv('CLAUDE_API_KEY') or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("Claude API key required. Set CLAUDE_API_KEY or ANTHROPIC_API_KEY environment variable")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # Load data components
        self.mapper = UnifiedIssueCategoryMapper(training_path, mapping_path)
        self.analyzer = DataSufficiencyAnalyzer(training_path)
        
        # Contract domain knowledge for realistic generation
        self._load_domain_templates()
    
    def _load_domain_templates(self):
        """Load contract correspondence templates and patterns"""
        
        # Common contract correspondence patterns
        self.correspondence_patterns = {
            'authority_to_contractor': [
                "Letter No. {ref_no} dated {date}",
                "Vide our letter under reference",
                "As per the contract provisions",
                "You are hereby instructed to",
                "Compliance is required within",
                "Non-compliance will result in"
            ],
            'contractor_to_authority': [
                "We refer to your letter",
                "In continuation to our previous correspondence",
                "We hereby submit our request for",
                "As per the contract agreement",
                "We request your kind consideration",
                "We look forward to your approval"
            ]
        }
        
        # Project entities for realistic context
        self.project_entities = {
            'locations': ['Highway', 'Bridge', 'Tunnel', 'Flyover', 'Underpass', 'Interchange'],
            'authorities': ['Authority Engineer', 'Project Director', 'Chief Engineer', 'Superintending Engineer'],
            'contractors': ['Contractor', 'Joint Venture', 'Subcontractor', 'Vendor'],
            'works': ['Construction', 'Design', 'Survey', 'Testing', 'Installation', 'Maintenance']
        }
    
    def generate_samples_for_issue(self, issue_type: str, target_samples: int = 8) -> List[Dict]:
        """
        Generate synthetic samples for a specific issue type using Claude
        """
        print(f"🤖 Generating {target_samples} samples for: {issue_type}")
        
        # Get category mapping for this issue
        categories = self.mapper.get_categories_for_issue(issue_type)
        primary_category = categories[0] if categories else 'Others'
        
        # Get existing samples for context (if any)
        existing_samples = self._get_existing_samples_for_issue(issue_type)
        
        # Create the generation prompt
        prompt = self._create_generation_prompt(issue_type, primary_category, existing_samples, target_samples)
        
        try:
            # Call Claude API
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                temperature=0.7,  # Some creativity for variety
                messages=[{
                    "role": "user", 
                    "content": prompt
                }]
            )
            
            # Parse the response
            samples = self._parse_claude_response(response.content[0].text, issue_type, primary_category)
            
            print(f"    ✅ Generated {len(samples)} samples")
            return samples
            
        except Exception as e:
            print(f"    ❌ Error generating samples: {e}")
            return []
    
    def _get_existing_samples_for_issue(self, issue_type: str, max_examples: int = 3) -> List[Dict]:
        """Get existing training samples for context"""
        if self.analyzer.df is None:
            return []
        
        # Find existing samples for this issue type
        matching_samples = self.analyzer.df[
            self.analyzer.df['issue_type'] == issue_type
        ].head(max_examples)
        
        examples = []
        for _, row in matching_samples.iterrows():
            examples.append({
                'subject': row.get('subject', ''),
                'body': row.get('body', ''),
                'category': row.get('category', '')
            })
        
        return examples
    
    def _create_generation_prompt(self, issue_type: str, category: str, existing_samples: List[Dict], target_samples: int) -> str:
        """
        Create a detailed prompt for Claude to generate realistic contract correspondence
        """
        
        # Build context from existing samples
        context_examples = ""
        if existing_samples:
            context_examples = "\n\nEXISTING EXAMPLES FOR REFERENCE:\n"
            for i, sample in enumerate(existing_samples, 1):
                context_examples += f"\nExample {i}:\n"
                context_examples += f"Subject: {sample['subject'][:200]}...\n"
                context_examples += f"Body: {sample['body'][:300]}...\n"
                context_examples += f"Category: {sample['category']}\n"
        
        prompt = f"""You are an expert in Indian contract law and infrastructure project management. Generate {target_samples} realistic contract correspondence samples for the issue type: "{issue_type}"

ISSUE TYPE: {issue_type}
PRIMARY CATEGORY: {category}

CONTEXT: This is for a highway/infrastructure construction project in India under an EPC (Engineering, Procurement, Construction) contract. The correspondence is typically between:
- Authority/Authority Engineer/Project Director (client side)
- Contractor/Joint Venture/Subcontractor (contractor side)

REQUIREMENTS:
1. Generate exactly {target_samples} different correspondence samples
2. Each sample should be realistic contract correspondence (letters, emails, notices)
3. Include proper Indian construction industry terminology
4. Use realistic project details (highway sections, chainage, amounts in INR)
5. Reference contract clauses, schedules, and standard practices
6. Vary the correspondence direction (Authority→Contractor, Contractor→Authority)
7. Include proper dates, reference numbers, and formal language

CORRESPONDENCE TYPES TO VARY:
- Formal letters with reference numbers
- Email communications
- Notice/intimation letters
- Request letters
- Status update communications

TECHNICAL TERMS TO INCLUDE:
- Contract clauses (Schedule-A, Schedule-B, Article 13, etc.)
- Highway terminology (chainage, ROW, structures, pavement)
- Indian construction terms (lakh, crore, SOR, MORTH specifications)
- Authority designations (AE, PD, SE, CGM)

{context_examples}

FORMAT YOUR RESPONSE AS JSON:
{{
  "samples": [
    {{
      "subject": "...",
      "body": "...",
      "issue_type": "{issue_type}",
      "category": "{category}",
      "correspondence_type": "authority_to_contractor" or "contractor_to_authority",
      "synthetic": true
    }},
    ...
  ]
}}

Generate {target_samples} diverse, realistic samples now:"""

        return prompt
    
    def _parse_claude_response(self, response_text: str, issue_type: str, category: str) -> List[Dict]:
        """
        Parse Claude's JSON response into structured samples
        """
        try:
            # Try to extract JSON from the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                print(f"    ⚠️  No JSON found in response")
                return []
            
            json_text = response_text[json_start:json_end]
            parsed = json.loads(json_text)
            
            samples = []
            for sample in parsed.get('samples', []):
                # Ensure required fields
                if 'subject' in sample and 'body' in sample:
                    # Clean and validate the sample
                    cleaned_sample = {
                        'subject': sample['subject'].strip(),
                        'body': sample['body'].strip(),
                        'issue_type': issue_type,
                        'category': category,
                        'correspondence_type': sample.get('correspondence_type', 'unknown'),
                        'synthetic': True,
                        'generated_date': datetime.now().isoformat(),
                        'generator': 'claude'
                    }
                    samples.append(cleaned_sample)
            
            return samples
            
        except json.JSONDecodeError as e:
            print(f"    ⚠️  JSON parsing error: {e}")
            # Try to extract samples manually if JSON parsing fails
            return self._extract_samples_manually(response_text, issue_type, category)
        except Exception as e:
            print(f"    ⚠️  Response parsing error: {e}")
            return []
    
    def _extract_samples_manually(self, text: str, issue_type: str, category: str) -> List[Dict]:
        """
        Fallback method to extract samples if JSON parsing fails
        """
        samples = []
        
        # Simple pattern matching for subject/body pairs
        import re
        
        # Look for subject: ... body: ... patterns
        pattern = r'"subject":\s*"([^"]+)".*?"body":\s*"([^"]+)"'
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        
        for subject, body in matches:
            if len(subject.strip()) > 20 and len(body.strip()) > 50:  # Basic quality check
                samples.append({
                    'subject': subject.strip(),
                    'body': body.strip(),
                    'issue_type': issue_type,
                    'category': category,
                    'correspondence_type': 'unknown',
                    'synthetic': True,
                    'generated_date': datetime.now().isoformat(),
                    'generator': 'claude_manual_extract'
                })
        
        return samples[:8]  # Limit to target samples
    
    def generate_priority_samples(self, priority_issues: List[str], samples_per_issue: int = 8) -> List[Dict]:
        """
        Generate samples for a list of priority issues
        """
        all_samples = []
        
        print(f"\n🤖 Generating {samples_per_issue} samples each for {len(priority_issues)} priority issues...")
        print(f"📊 Total samples to generate: {len(priority_issues) * samples_per_issue}")
        
        for i, issue_type in enumerate(priority_issues, 1):
            print(f"\n📝 Progress: {i}/{len(priority_issues)} - {issue_type}")
            
            try:
                samples = self.generate_samples_for_issue(issue_type, samples_per_issue)
                all_samples.extend(samples)
                
                # Rate limiting - be respectful to the API
                if i < len(priority_issues):
                    print(f"    ⏱️  Rate limiting: waiting 2 seconds...")
                    time.sleep(2)
                    
            except Exception as e:
                print(f"    ❌ Failed to generate samples for {issue_type}: {e}")
                continue
        
        print(f"\n✅ Generated {len(all_samples)} total samples")
        return all_samples
    
    def save_samples(self, samples: List[Dict], output_path: str = None) -> str:
        """
        Save generated samples to Excel file
        """
        if not samples:
            print("❌ No samples to save")
            return None
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/synthetic/claude_generated_samples_{timestamp}.xlsx"
        
        # Create DataFrame and save
        df = pd.DataFrame(samples)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_excel(output_path, index=False)
        
        print(f"\n💾 Saved {len(samples)} samples to: {output_path}")
        print(f"📊 Unique issue types: {df['issue_type'].nunique()}")
        print(f"📊 Categories covered: {df['category'].nunique()}")
        
        return output_path
    
    def combine_with_training_data(self, synthetic_samples: List[Dict], output_path: str = None) -> str:
        """
        Combine synthetic samples with existing training data
        """
        # Load original training data
        original_df = pd.read_excel(self.training_path)
        
        # Convert synthetic samples to DataFrame
        synthetic_df = pd.DataFrame(synthetic_samples)
        
        # Combine datasets
        enhanced_df = pd.concat([original_df, synthetic_df], ignore_index=True)
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/synthetic/enhanced_training_claude_{timestamp}.xlsx"
        
        # Save enhanced dataset
        enhanced_df.to_excel(output_path, index=False)
        
        print(f"\n✅ Enhanced training data saved to: {output_path}")
        print(f"📊 Original samples: {len(original_df)}")
        print(f"📊 Synthetic samples: {len(synthetic_df)}")
        print(f"📊 Total samples: {len(enhanced_df)}")
        print(f"📊 Unique issue types: {enhanced_df['issue_type'].nunique()}")
        
        return output_path


def main():
    """Main execution function"""
    print("🤖 Claude-based Synthetic Data Generator for CCMS")
    print("=" * 60)
    
    # Initialize generator
    training_path = 'data/synthetic/combined_training_data.xlsx'
    mapping_path = 'issue_category_mapping_diffs/unified_issue_category_mapping.xlsx'
    
    try:
        generator = ClaudeSyntheticGenerator(training_path, mapping_path)
        print("✅ Claude generator initialized successfully")
        return generator
    except Exception as e:
        print(f"❌ Failed to initialize generator: {e}")
        return None


if __name__ == "__main__":
    generator = main()