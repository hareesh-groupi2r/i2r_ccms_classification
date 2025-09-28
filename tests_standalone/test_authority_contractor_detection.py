#!/usr/bin/env python3
"""
Direct test: Can we detect Authority Engineer and Contractor issues?
"""

import requests
import json

def test_authority_contractor():
    """Test if we can detect Authority/Contractor issues directly"""
    
    print("🧪 TESTING AUTHORITY ENGINEER & CONTRACTOR DETECTION")
    print("=" * 70)
    
    # Create a focused test document that clearly has these issues
    test_text = """
    Dear Sir,
    
    Subject: Authority Engineer Instructions and Contractor Obligations
    
    1. AUTHORITY ENGINEER INSTRUCTIONS:
    The Authority Engineer has issued instructions vide letter dated 15.03.2019 
    regarding the mobilization of additional resources at site. The Authority Engineer 
    is responsible for providing necessary clearances and approvals.
    
    2. CONTRACTOR'S OBLIGATIONS:
    As per the contract, the Contractor is obligated to:
    - Submit monthly progress reports to the Authority Engineer
    - Provide bank guarantees for performance security
    - Maintain adequate manpower and machinery at site
    - Submit design and drawings for approval
    
    3. APPOINTED DATE ISSUES:  
    The project appointed date was declared on 01.01.2019. Due to delays in 
    right of way acquisition by the Authority, modification of appointed date 
    is being requested.
    
    We request your kind consideration of the above contractual obligations.
    
    Yours faithfully,
    For XYZ Contractors Ltd.
    Project Manager
    """
    
    print("📝 Testing with focused content containing:")
    print("   ✅ 'Authority Engineer' mentions")
    print("   ✅ 'Contractor's Obligations' mentions")
    print("   ✅ 'Appointed Date' mentions")
    print("   ✅ Specific obligation details")
    
    print()
    print("🔄 Making API call...")
    
    try:
        response = requests.post(
            'http://localhost:5001/api/services/hybrid-rag-classification/classify-text',
            headers={'Content-Type': 'application/json'},
            json={
                'subject': 'Authority Engineer Instructions and Contractor Obligations',
                'body': test_text,
                'options': {
                    'max_results': 10,  # Allow more results
                    'confidence_threshold': 0.3  # Lower confidence threshold
                }
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print("📊 CLASSIFICATION RESULTS:")
            
            if 'data' in result and 'categories' in result['data']:
                categories = result['data']['categories']
                print(f"   🏷️  Categories found ({len(categories)}):")
                for cat in categories:
                    category = cat.get('category', 'Unknown')
                    confidence = cat.get('confidence', 0)
                    print(f"     • {category} (confidence: {confidence:.3f})")
                
                # Check if we found the missing categories
                found_categories = [cat['category'] for cat in categories]
                
                missing_target_cats = ["Authority's Obligations", "Contractor's Obligations", "Appointed Date"]
                found_target_cats = [cat for cat in missing_target_cats if cat in found_categories]
                still_missing = [cat for cat in missing_target_cats if cat not in found_categories]
                
                print(f"   ✅ Found target categories: {found_target_cats}")
                print(f"   ❌ Still missing: {still_missing}")
            
            if 'data' in result and 'identified_issues' in result['data']:
                issues = result['data']['identified_issues']
                print(f"   🎯 Issues identified ({len(issues)}):")
                for issue in issues:
                    issue_type = issue.get('issue_type', 'Unknown')
                    confidence = issue.get('confidence', 0)
                    source = issue.get('source', 'unknown')
                    print(f"     • {issue_type} (conf: {confidence:.3f}, {source})")
        
        else:
            print(f"❌ API Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_authority_contractor()