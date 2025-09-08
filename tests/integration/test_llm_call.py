#!/usr/bin/env python3
"""
Test LLM call directly to see the actual prompt and response
"""
import sys
sys.path.append('.')
import os
import anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_anthropic_direct():
    """Test Anthropic call directly with simple prompt"""
    
    print("🧪 Testing Direct Anthropic LLM Call")
    print("=" * 50)
    
    # Initialize Anthropic client
    client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    
    # Simple test content about change of scope
    test_content = """Subject: Change of Scope Proposal for toll plaza construction at Chennasamudram Village

Content: It is to inform that as per the present Scope of Works the number of Toll Lanes to be made at the Chennasamudram Toll Plaza are 24. However, based on the NHAI letter it is understood that the number of Toll Lanes to be re-designed duly considering Hybrid ETC system. We request approval for proceeding with this revised proposal for Change of Scope."""
    
    # Create a simple classification prompt
    prompt = f"""
Task: Analyze this contract correspondence and identify ALL issue types being discussed.
For each issue, provide your confidence level and supporting evidence from the document.

Document:
{test_content}

The possible issue types include:
- Change of scope proposals clarifications
- Handing over of land /Possession of site
- Slow Progress of Works
- Utility shifting
- EoT
- Payments
- Contractor's Obligations
- Authority's Obligations
- Dispute Resolution

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
    
    print(f"📄 Test content: {len(test_content)} chars")
    print(f"📝 Prompt length: {len(prompt)} chars")
    print(f"📜 Full prompt:")
    print("-" * 40)
    print(prompt)
    print("-" * 40)
    
    try:
        # Make direct Anthropic call
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            temperature=0.1,
            system="You are a contract classification expert.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        response_text = response.content[0].text
        
        print(f"✅ Anthropic response received")
        print(f"📊 Response length: {len(response_text)} chars")
        print(f"📜 Full response:")
        print("-" * 40)
        print(response_text)
        print("-" * 40)
        
        # Try to parse the JSON (simulate the fixed parsing logic)
        import json
        try:
            # Apply same cleanup as the fixed _parse_llm_response
            cleaned_response = response_text.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            elif cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            print(f"🧹 Cleaned response: {cleaned_response}")
            
            parsed = json.loads(cleaned_response)
            print(f"✅ JSON parsing successful")
            print(f"📚 Issues found: {len(parsed.get('issues', []))}")
            
            for i, issue in enumerate(parsed.get('issues', []), 1):
                print(f"  {i}. {issue.get('issue_type', 'unknown')}")
                print(f"     Confidence: {issue.get('confidence', 0)}")
                print(f"     Evidence: {issue.get('evidence', 'none')[:100]}...")
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            
    except Exception as e:
        print(f"❌ Anthropic call failed: {e}")

if __name__ == "__main__":
    test_anthropic_direct()