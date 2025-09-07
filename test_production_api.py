#!/usr/bin/env python3
"""
Test client for Contract Correspondence Classification Production API
Tests all endpoints and functionality
"""

import httpx
import asyncio
import json
import time
from typing import Dict, List
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


class CCMSAPIClient:
    """Client for testing the CCMS API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def health_check(self) -> Dict:
        """Check API health status."""
        response = await self.client.get(f"{self.base_url}/health")
        return response.json()
    
    async def get_categories(self) -> Dict:
        """Get available categories."""
        response = await self.client.get(f"{self.base_url}/categories")
        return response.json()
    
    async def get_issue_types(self) -> Dict:
        """Get available issue types."""
        response = await self.client.get(f"{self.base_url}/issue-types")
        return response.json()
    
    async def get_stats(self) -> Dict:
        """Get system statistics."""
        response = await self.client.get(f"{self.base_url}/stats")
        return response.json()
    
    async def classify_text(self, text: str, approach: str = "hybrid_rag", 
                           confidence_threshold: float = 0.7) -> Dict:
        """Classify a single text."""
        payload = {
            "text": text,
            "approach": approach,
            "confidence_threshold": confidence_threshold,
            "max_results": 5
        }
        
        response = await self.client.post(f"{self.base_url}/classify", json=payload)
        return response.json()
    
    async def classify_batch(self, texts: List[str], approach: str = "hybrid_rag") -> Dict:
        """Classify multiple texts."""
        payload = {
            "texts": texts,
            "approach": approach,
            "confidence_threshold": 0.7,
            "max_results": 5
        }
        
        response = await self.client.post(f"{self.base_url}/classify/batch", json=payload)
        return response.json()


async def run_api_tests():
    """Run comprehensive API tests."""
    print("🧪 Testing Contract Correspondence Classification API")
    print("=" * 60)
    
    async with CCMSAPIClient() as client:
        # Test 1: Health Check
        print("1. 🏥 Testing health check...")
        try:
            health = await client.health_check()
            print(f"   Status: {health.get('status', 'unknown')}")
            print(f"   Version: {health.get('version', 'unknown')}")
            print(f"   Classifiers: {health.get('classifiers_loaded', {})}")
            print("   ✅ Health check passed")
        except Exception as e:
            print(f"   ❌ Health check failed: {e}")
            return
        
        print()
        
        # Test 2: Categories
        print("2. 📂 Testing categories endpoint...")
        try:
            categories = await client.get_categories()
            print(f"   Total categories: {categories.get('total_count', 0)}")
            print(f"   Categories: {categories.get('categories', [])[:5]}...")
            print("   ✅ Categories endpoint passed")
        except Exception as e:
            print(f"   ❌ Categories endpoint failed: {e}")
        
        print()
        
        # Test 3: Issue Types
        print("3. 📋 Testing issue types endpoint...")
        try:
            issue_types = await client.get_issue_types()
            print(f"   Total issue types: {issue_types.get('total_count', 0)}")
            print(f"   Sample issues: {issue_types.get('issue_types', [])[:3]}...")
            print("   ✅ Issue types endpoint passed")
        except Exception as e:
            print(f"   ❌ Issue types endpoint failed: {e}")
        
        print()
        
        # Test 4: System Stats
        print("4. 📊 Testing system stats...")
        try:
            stats = await client.get_stats()
            training_data = stats.get('training_data', {})
            print(f"   Training samples: {training_data.get('total_samples', 0)}")
            print(f"   Available classifiers: {stats.get('classifiers', {}).get('available', [])}")
            print("   ✅ System stats passed")
        except Exception as e:
            print(f"   ❌ System stats failed: {e}")
        
        print()
        
        # Test 5: Single Classification
        print("5. 🔍 Testing single classification...")
        test_texts = [
            "Request for extension of time due to weather delays in highway construction project",
            "Payment delay notification for completed work under contract section 3.2",
            "Change of scope required for additional bridge construction not in original plan",
            "Authority failed to provide right of way clearance as per schedule"
        ]
        
        for i, text in enumerate(test_texts[:2]):  # Test first 2
            try:
                print(f"   Testing text {i+1}...")
                result = await client.classify_text(text)
                
                if result.get('status') == 'success':
                    issues = result.get('identified_issues', [])
                    categories = result.get('categories', [])
                    confidence = result.get('confidence_score', 0)
                    processing_time = result.get('processing_time', 0)
                    
                    print(f"     Issues found: {len(issues)}")
                    print(f"     Categories: {len(categories)}")
                    print(f"     Confidence: {confidence:.3f}")
                    print(f"     Time: {processing_time:.3f}s")
                    
                    if issues:
                        print(f"     Top issue: {issues[0]['issue_type']} ({issues[0]['confidence']:.3f})")
                    if categories:
                        print(f"     Top category: {categories[0]['category']} ({categories[0]['confidence']:.3f})")
                else:
                    print(f"     ❌ Classification failed: {result.get('validation_report', {}).get('error', 'Unknown error')}")
                
            except Exception as e:
                print(f"     ❌ Text {i+1} classification failed: {e}")
        
        print("   ✅ Single classification tests completed")
        print()
        
        # Test 6: Batch Classification
        print("6. 📦 Testing batch classification...")
        try:
            batch_texts = test_texts[:3]  # Use first 3 texts
            print(f"   Testing batch of {len(batch_texts)} texts...")
            
            start_time = time.time()
            batch_result = await client.classify_batch(batch_texts)
            batch_time = time.time() - start_time
            
            if batch_result.get('status') == 'completed':
                total_items = batch_result.get('total_items', 0)
                completed = batch_result.get('completed_items', 0)
                failed = batch_result.get('failed_items', 0)
                processing_time = batch_result.get('total_processing_time', 0)
                
                print(f"     Total items: {total_items}")
                print(f"     Completed: {completed}")
                print(f"     Failed: {failed}")
                print(f"     Success rate: {completed/total_items*100:.1f}%")
                print(f"     Processing time: {processing_time:.3f}s")
                print(f"     Network time: {batch_time:.3f}s")
                print(f"     Avg per item: {processing_time/total_items:.3f}s")
                
                print("   ✅ Batch classification passed")
            else:
                print(f"     ❌ Batch classification failed: {batch_result}")
                
        except Exception as e:
            print(f"   ❌ Batch classification failed: {e}")
        
        print()
        
        # Test 7: Error Handling
        print("7. 🚫 Testing error handling...")
        try:
            # Test with invalid approach
            try:
                await client.classify_text("Test text", approach="invalid_approach")
                print("     ❌ Should have failed with invalid approach")
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 400:
                    print("     ✅ Correctly rejected invalid approach")
                else:
                    print(f"     ⚠️  Unexpected status code: {e.response.status_code}")
            
            # Test with empty text
            try:
                result = await client.classify_text("")
                if result.get('status') == 'error':
                    print("     ✅ Correctly handled empty text")
                else:
                    print("     ⚠️  Empty text not properly handled")
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 422:  # Validation error
                    print("     ✅ Correctly rejected empty text")
                else:
                    print(f"     ⚠️  Unexpected status for empty text: {e.response.status_code}")
            
        except Exception as e:
            print(f"   ❌ Error handling test failed: {e}")
        
        print()
    
    print("=" * 60)
    print("🎯 API Testing Complete!")
    print()
    print("📖 API Documentation available at:")
    print("   http://localhost:8000/docs")
    print("   http://localhost:8000/redoc")


async def demo_classification():
    """Demonstrate classification with real examples."""
    print()
    print("🎭 CLASSIFICATION DEMO")
    print("=" * 60)
    
    demo_texts = [
        {
            "text": "We request an extension of 45 days for completion of bridge construction due to unprecedented rainfall and flooding at the site.",
            "description": "Extension of Time Request"
        },
        {
            "text": "Payment of Rs. 25,00,000 for completed work under milestone 3 is pending approval from finance department. Please expedite processing.",
            "description": "Payment Delay Issue"
        },
        {
            "text": "Additional work for construction of service road not included in original scope is required as per local authority demands.",
            "description": "Change of Scope Request"
        },
        {
            "text": "Authority has failed to provide environmental clearance as scheduled, causing delay in project commencement.",
            "description": "Authority Obligation Issue"
        },
        {
            "text": "Contractor has not submitted performance bank guarantee as per contract clause 5.3. Please provide immediately.",
            "description": "Contractor Obligation Issue"
        }
    ]
    
    async with CCMSAPIClient() as client:
        for i, demo in enumerate(demo_texts, 1):
            print(f"{i}. {demo['description']}")
            print(f"   Text: {demo['text'][:80]}...")
            
            try:
                result = await client.classify_text(demo['text'])
                
                if result.get('status') == 'success':
                    issues = result.get('identified_issues', [])
                    categories = result.get('categories', [])
                    confidence = result.get('confidence_score', 0)
                    
                    print(f"   📊 Confidence: {confidence:.3f}")
                    
                    if issues:
                        print("   🔍 Top Issues:")
                        for issue in issues[:3]:
                            print(f"     • {issue['issue_type']} ({issue['confidence']:.3f})")
                    
                    if categories:
                        print("   📂 Categories:")
                        for cat in categories[:3]:
                            print(f"     • {cat['category']} ({cat['confidence']:.3f})")
                    
                    # Show warnings if any
                    warnings = result.get('data_sufficiency_warnings', [])
                    if warnings:
                        print("   ⚠️  Warnings:")
                        for warning in warnings[:2]:
                            print(f"     • {warning.get('message', 'Unknown warning')}")
                
                else:
                    print(f"   ❌ Classification failed")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
            
            print()
    
    print("=" * 60)
    print("Demo complete! Try the API with your own texts.")


async def main():
    """Main test function."""
    try:
        await run_api_tests()
        await demo_classification()
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())