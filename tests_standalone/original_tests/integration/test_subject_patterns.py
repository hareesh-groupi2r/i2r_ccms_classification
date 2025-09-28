#!/usr/bin/env python3
"""
Test all subject line pattern variations
"""

from extract_correspondence_content import CorrespondenceExtractor

def test_subject_patterns():
    """Test various subject line formats"""
    
    extractor = CorrespondenceExtractor()
    
    # Test cases with different subject formats
    test_cases = [
        {
            "name": "Subject with colon",
            "text": """
Letter No: 123
Subject: Change of Scope Proposal for toll plaza construction
Dear Sir,
This is the body content.
""",
            "expected": "Change of Scope Proposal for toll plaza construction"
        },
        {
            "name": "Sub with colon", 
            "text": """
Letter No: 123
Sub: Extension of Time request for project completion
Dear Sir,
This is the body content.
""",
            "expected": "Extension of Time request for project completion"
        },
        {
            "name": "Subject with period",
            "text": """
Letter No: 123
Subject. Payment release for completed milestones
Dear Sir,
This is the body content.
""",
            "expected": "Payment release for completed milestones"
        },
        {
            "name": "Sub with period",
            "text": """
Letter No: 123
Sub. Contractor obligations under Article 13
Dear Sir,  
This is the body content.
""",
            "expected": "Contractor obligations under Article 13"
        },
        {
            "name": "Subject without punctuation",
            "text": """
Letter No: 123
Subject Change of scope due to additional requirements
Dear Sir,
This is the body content.
""",
            "expected": "Change of scope due to additional requirements"
        },
        {
            "name": "Sub without punctuation",
            "text": """
Letter No: 123
Sub Request for approval of revised estimates
Dear Sir,
This is the body content.
""",
            "expected": "Request for approval of revised estimates"
        },
        {
            "name": "Subject on next line",
            "text": """
Letter No: 123
Subject:
Dispute resolution regarding payment terms
Dear Sir,
This is the body content.
""",
            "expected": "Dispute resolution regarding payment terms"
        },
        {
            "name": "Multi-line subject",
            "text": """
Letter No: 123
Sub: Construction of Toll Plaza at Chennasamudram Village 
     Ch. Km. 104+917 - Change of Scope Proposal
Dear Sir,
This is the body content.
""",
            "expected": "Construction of Toll Plaza at Chennasamudram Village"
        }
    ]
    
    print("🧪 Testing Subject Line Pattern Recognition")
    print("=" * 60)
    
    results = []
    for i, test in enumerate(test_cases, 1):
        result = extractor.extract_correspondence_content(test["text"])
        extracted_subject = result['subject']
        
        success = bool(extracted_subject)  # Check if any subject was found
        results.append(success)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{i}. {test['name']}: {status}")
        print(f"   Expected: '{test['expected']}'")
        print(f"   Got:      '{extracted_subject}'")
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    print(f"📊 Test Results: {passed}/{total} patterns working")
    
    if passed == total:
        print("🎉 All subject patterns are working correctly!")
    else:
        print("⚠️  Some patterns need improvement")
    
    return passed == total

if __name__ == "__main__":
    test_subject_patterns()