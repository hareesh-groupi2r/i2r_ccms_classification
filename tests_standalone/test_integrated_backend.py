#!/usr/bin/env python3
"""
Test script for the integrated backend with hybrid RAG classification
Tests the Flask backend API endpoints to verify integration
"""

import requests
import json
import time
from datetime import datetime

# Configuration - Read from environment variables
import os
BASE_URL = os.getenv('BACKEND_URL', "http://localhost:5001")
API_BASE = f"{BASE_URL}/api/services"

def test_health_endpoint():
    """Test the health check endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE}/health", timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Health Status: {data.get('status', 'unknown')}")
            print("   ✅ Health check passed")
            return True
        else:
            print(f"   ❌ Health check failed: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return False

def test_classification_status():
    """Test the classification service status endpoint"""
    print("🔍 Testing classification service status...")
    try:
        response = requests.get(f"{API_BASE}/hybrid-rag-classification/status", timeout=30)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                service_data = data.get('data', {})
                print(f"   Service Name: {service_data.get('service_name')}")
                print(f"   Initialized: {service_data.get('is_initialized')}")
                print(f"   Available Approaches: {service_data.get('available_approaches', [])}")
                if service_data.get('is_initialized'):
                    print(f"   Issue Types: {service_data.get('total_issue_types', 0)}")
                    print(f"   Categories: {service_data.get('total_categories', 0)}")
                    print("   ✅ Classification service status OK")
                    return True
                else:
                    print(f"   ⚠️  Service not initialized: {service_data.get('initialization_error')}")
                    return False
            else:
                print(f"   ❌ Status request failed: {data}")
                return False
        else:
            print(f"   ❌ Status request failed: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Status request error: {e}")
        return False

def test_get_categories():
    """Test getting available categories"""
    print("🔍 Testing get categories endpoint...")
    try:
        response = requests.get(f"{API_BASE}/hybrid-rag-classification/categories", timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                categories_data = data.get('data', {})
                categories = categories_data.get('categories', [])
                print(f"   Total Categories: {len(categories)}")
                print(f"   Categories: {', '.join(categories[:5])}{'...' if len(categories) > 5 else ''}")
                print("   ✅ Categories retrieved successfully")
                return categories
            else:
                print(f"   ❌ Failed to get categories: {data}")
                return None
        else:
            print(f"   ❌ Failed to get categories: {response.text}")
            return None
    except Exception as e:
        print(f"   ❌ Get categories error: {e}")
        return None

def test_text_classification():
    """Test text classification endpoint"""
    print("🔍 Testing text classification...")
    
    # Test data
    test_cases = [
        {
            "subject": "Request for Extension of Time - Project Milestone 3",
            "body": "We request a 30-day extension for project milestone 3 due to unforeseen weather delays and material delivery issues."
        },
        {
            "subject": "Payment Delay Notification - Invoice #2024-001", 
            "body": "This is to inform you that the payment for invoice #2024-001 will be delayed by 15 days due to pending approvals."
        },
        {
            "subject": "Change of Scope Proposal - Additional Work Package",
            "body": "We propose adding extra work package as per client requirements. The additional scope includes structural modifications."
        }
    ]
    
    successful_tests = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n   📝 Test Case {i}: {test_case['subject'][:50]}...")
        
        try:
            payload = {
                "subject": test_case["subject"],
                "body": test_case["body"],
                "options": {
                    "approach": "hybrid_rag",
                    "confidence_threshold": 0.5,
                    "max_results": 3,
                    "include_justification": True,
                    "include_issue_types": True
                }
            }
            
            start_time = time.time()
            response = requests.post(
                f"{API_BASE}/hybrid-rag-classification/classify-text", 
                json=payload, 
                timeout=60
            )
            processing_time = time.time() - start_time
            
            print(f"      Status: {response.status_code} (took {processing_time:.2f}s)")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    result = data.get('data', {})
                    print(f"      Approach: {result.get('approach_used')}")
                    print(f"      Confidence: {result.get('confidence_score', 0):.3f}")
                    
                    categories = result.get('categories', [])
                    if categories:
                        print(f"      Categories ({len(categories)}):")
                        for cat in categories[:2]:
                            print(f"        - {cat['category']} (confidence: {cat['confidence']:.3f})")
                    
                    issues = result.get('identified_issues', [])
                    if issues:
                        print(f"      Issues ({len(issues)}):")
                        for issue in issues[:2]:
                            print(f"        - {issue['issue_type']} (confidence: {issue['confidence']:.3f})")
                    
                    print(f"      ✅ Classification successful")
                    successful_tests += 1
                else:
                    print(f"      ❌ Classification failed: {data}")
            else:
                print(f"      ❌ Request failed: {response.text[:200]}")
                
        except Exception as e:
            print(f"      ❌ Classification error: {e}")
    
    print(f"\n   📊 Text Classification Summary: {successful_tests}/{len(test_cases)} successful")
    return successful_tests == len(test_cases)

def test_batch_classification():
    """Test batch classification endpoint"""
    print("🔍 Testing batch classification...")
    
    try:
        payload = {
            "texts": [
                {
                    "subject": "Utility Shifting Request - Phase 1",
                    "body": "Request for utility shifting to proceed with Phase 1 construction activities."
                },
                {
                    "subject": "Quality Control Issues - Materials Testing",
                    "body": "Several quality control issues identified in materials testing. Immediate attention required."
                },
                {
                    "subject": "Authority Approval Required - Design Changes", 
                    "body": "Design changes require authority approval before implementation. Awaiting response."
                }
            ],
            "options": {
                "approach": "hybrid_rag",
                "confidence_threshold": 0.4,
                "max_results": 2
            }
        }
        
        start_time = time.time()
        response = requests.post(
            f"{API_BASE}/hybrid-rag-classification/classify-batch",
            json=payload,
            timeout=120
        )
        processing_time = time.time() - start_time
        
        print(f"   Status: {response.status_code} (took {processing_time:.2f}s)")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                result = data.get('data', {})
                print(f"   Total Items: {result.get('total_items', 0)}")
                print(f"   Successful: {result.get('successful_items', 0)}")
                print(f"   Failed: {result.get('failed_items', 0)}")
                print(f"   Avg Processing Time: {result.get('average_processing_time', 0):.2f}s")
                print("   ✅ Batch classification successful")
                return True
            else:
                print(f"   ❌ Batch classification failed: {data}")
                return False
        else:
            print(f"   ❌ Batch request failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Batch classification error: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("INTEGRATED BACKEND TESTING")
    print(f"Testing Flask backend at: {BASE_URL}")
    print("=" * 60)
    
    # Test results
    results = {
        'health_check': False,
        'classification_status': False,
        'get_categories': False,
        'text_classification': False,
        'batch_classification': False
    }
    
    print(f"\n🚀 Starting tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run tests
    print(f"\n" + "-" * 40)
    results['health_check'] = test_health_endpoint()
    
    print(f"\n" + "-" * 40)
    results['classification_status'] = test_classification_status()
    
    print(f"\n" + "-" * 40)
    categories = test_get_categories()
    results['get_categories'] = categories is not None
    
    # Only run classification tests if service is initialized
    if results['classification_status']:
        print(f"\n" + "-" * 40)
        results['text_classification'] = test_text_classification()
        
        print(f"\n" + "-" * 40)
        results['batch_classification'] = test_batch_classification()
    else:
        print(f"\n⚠️  Skipping classification tests - service not initialized")
    
    # Summary
    print(f"\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\n📊 Overall Result: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Integrated backend is working correctly.")
    elif passed >= total * 0.6:
        print("⚠️  Most tests passed. Check failed tests and retry.")
    else:
        print("❌ Many tests failed. Check backend configuration and dependencies.")
    
    print(f"\n💡 Next Steps:")
    if not results['health_check']:
        print("   1. Start the Flask backend: python api/app.py")
    elif not results['classification_status']:
        print("   2. Check classification system dependencies and API keys")
    elif passed < total:
        print("   3. Debug failed test endpoints")
    else:
        print("   4. Ready for integration with Next.js frontend!")

if __name__ == "__main__":
    main()