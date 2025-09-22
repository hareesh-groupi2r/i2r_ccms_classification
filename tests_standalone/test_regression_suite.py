#!/usr/bin/env python3
"""
Comprehensive Regression Test Suite for Integrated CCMS Backend Service
Tests all services copied from the original backend plus the new hybrid RAG classification
"""

import requests
import json
import time
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import tempfile
from dataclasses import dataclass

# Configuration
BASE_URL = os.getenv('BACKEND_URL', "http://localhost:5001")
API_BASE = f"{BASE_URL}/api/services"

# Colors for output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

@dataclass
class TestResult:
    name: str
    passed: bool
    duration: float
    details: str = ""
    error: Optional[str] = None

class RegressionTestSuite:
    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = datetime.now()
        
    def print_colored(self, color: str, message: str):
        print(f"{color}{message}{Colors.RESET}")
        
    def print_header(self, message: str):
        self.print_colored(Colors.BOLD + Colors.BLUE, f"\n{'='*80}")
        self.print_colored(Colors.BOLD + Colors.BLUE, f"{message}")
        self.print_colored(Colors.BOLD + Colors.BLUE, f"{'='*80}")
        
    def print_test_start(self, test_name: str):
        self.print_colored(Colors.CYAN, f"\n🧪 Testing: {test_name}")
        
    def print_success(self, message: str):
        self.print_colored(Colors.GREEN, f"   ✅ {message}")
        
    def print_warning(self, message: str):
        self.print_colored(Colors.YELLOW, f"   ⚠️  {message}")
        
    def print_error(self, message: str):
        self.print_colored(Colors.RED, f"   ❌ {message}")
        
    def run_test(self, test_name: str, test_func) -> TestResult:
        """Run a single test and capture results"""
        self.print_test_start(test_name)
        start_time = time.time()
        
        try:
            result = test_func()
            duration = time.time() - start_time
            
            if result.get('success', False):
                self.print_success(f"Passed in {duration:.2f}s")
                test_result = TestResult(test_name, True, duration, result.get('details', ''))
            else:
                self.print_error(f"Failed: {result.get('error', 'Unknown error')}")
                test_result = TestResult(test_name, False, duration, error=result.get('error'))
                
        except Exception as e:
            duration = time.time() - start_time
            self.print_error(f"Exception: {str(e)}")
            test_result = TestResult(test_name, False, duration, error=str(e))
            
        self.results.append(test_result)
        return test_result
    
    # ================================================================================
    # HEALTH AND STATUS TESTS
    # ================================================================================
    
    def test_health_endpoint(self) -> Dict:
        """Test the health check endpoint"""
        try:
            response = requests.get(f"{API_BASE}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    'success': True,
                    'details': f"Health status: {data.get('status', 'unknown')}"
                }
            else:
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}: {response.text[:100]}"
                }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_api_documentation(self) -> Dict:
        """Test API documentation endpoint"""
        try:
            response = requests.get(f"{BASE_URL}/api", timeout=10)
            if response.status_code == 200:
                data = response.json()
                endpoints = data.get('endpoints', {})
                return {
                    'success': True,
                    'details': f"Found {len(endpoints)} endpoint groups"
                }
            else:
                return {'success': False, 'error': f"HTTP {response.status_code}"}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    # ================================================================================
    # DOCUMENT TYPE CLASSIFICATION TESTS
    # ================================================================================
    
    def test_document_type_classification_text(self) -> Dict:
        """Test document type classification from text"""
        test_cases = [
            {
                "name": "Correspondence Letter",
                "text": "Dear Sir, Subject: Project Status Update. We are writing to inform you about the current status of the project. Yours faithfully, Project Manager",
                "expected": "correspondence_letter"
            },
            {
                "name": "Meeting Minutes",
                "text": "Minutes of Meeting held on 2024-08-23. Present: Project Manager, Site Engineer, Contractor. Action items: 1. Complete foundation work",
                "expected": "meeting_minutes"
            },
            {
                "name": "Progress Report", 
                "text": "Monthly Progress Report for August 2024. Physical Progress: 45% completed. Financial Progress: Rs. 50,00,000 spent.",
                "expected": "progress_report"
            }
        ]
        
        successful_tests = 0
        total_tests = len(test_cases)
        
        for test_case in test_cases:
            try:
                response = requests.post(
                    f"{API_BASE}/document-type/classify-text",
                    json={"text": test_case["text"]},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success'):
                        successful_tests += 1
                        
            except Exception as e:
                self.print_warning(f"Test case '{test_case['name']}' failed: {e}")
        
        return {
            'success': successful_tests == total_tests,
            'details': f"Passed {successful_tests}/{total_tests} document type tests"
        }
    
    # ================================================================================
    # OCR SERVICE TESTS  
    # ================================================================================
    
    def test_ocr_methods(self) -> Dict:
        """Test OCR methods endpoint"""
        try:
            response = requests.get(f"{API_BASE}/ocr/methods", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    methods = data.get('data', [])
                    return {
                        'success': True,
                        'details': f"Available OCR methods: {methods}"
                    }
            return {'success': False, 'error': f"HTTP {response.status_code}"}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    # ================================================================================
    # LLM SERVICE TESTS
    # ================================================================================
    
    def test_llm_status(self) -> Dict:
        """Test LLM service status"""
        try:
            response = requests.get(f"{API_BASE}/llm/status", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    return {
                        'success': True,
                        'details': "LLM service is available"
                    }
            return {'success': False, 'error': f"HTTP {response.status_code}"}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_llm_structured_extraction(self) -> Dict:
        """Test LLM structured data extraction"""
        test_text = """
        Letter No: AE/2024/001
        Date: 2024-08-23
        Subject: Project Status Update
        
        Dear Sir,
        
        We are writing to inform you about the current project status.
        The work is progressing as per schedule.
        
        Yours faithfully,
        Project Manager
        """
        
        extraction_schema = {
            'letter_id': 'Letter reference number or ID',
            'date_sent': 'Date when the letter was sent',
            'sender': 'Name of the sender',
            'subject': 'Subject line of the letter'
        }
        
        try:
            response = requests.post(
                f"{API_BASE}/llm/extract-structured",
                json={
                    "text": test_text,
                    "schema": extraction_schema,
                    "output_format": "json"
                },
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    extracted_data = data.get('data', {})
                    return {
                        'success': True,
                        'details': f"Extracted {len(extracted_data)} fields"
                    }
            
            return {'success': False, 'error': f"HTTP {response.status_code}"}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    # ================================================================================
    # CATEGORY MAPPING SERVICE TESTS
    # ================================================================================
    
    def test_category_mapping_categories(self) -> Dict:
        """Test category mapping - get available categories"""
        try:
            response = requests.get(f"{API_BASE}/category-mapping/categories", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    categories = data.get('data', [])
                    return {
                        'success': True,
                        'details': f"Found {len(categories)} categories"
                    }
            return {'success': False, 'error': f"HTTP {response.status_code}"}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_category_mapping_issue(self) -> Dict:
        """Test issue to category mapping"""
        test_issues = [
            "Extension of Time",
            "Payment Delay", 
            "Material Supply",
            "Quality Issues"
        ]
        
        successful_mappings = 0
        
        for issue in test_issues:
            try:
                response = requests.post(
                    f"{API_BASE}/category-mapping/map-issue",
                    json={"issue_type": issue},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success'):
                        successful_mappings += 1
                        
            except Exception as e:
                self.print_warning(f"Mapping issue '{issue}' failed: {e}")
        
        return {
            'success': successful_mappings > 0,
            'details': f"Successfully mapped {successful_mappings}/{len(test_issues)} issues"
        }
    
    def test_category_mapping_statistics(self) -> Dict:
        """Test category mapping statistics"""
        try:
            response = requests.get(f"{API_BASE}/category-mapping/statistics", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    stats = data.get('data', {})
                    return {
                        'success': True,
                        'details': f"Stats: {stats.get('total_mappings', 0)} mappings, {stats.get('total_categories', 0)} categories"
                    }
            return {'success': False, 'error': f"HTTP {response.status_code}"}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    # ================================================================================
    # HYBRID RAG CLASSIFICATION TESTS
    # ================================================================================
    
    def test_hybrid_rag_status(self) -> Dict:
        """Test hybrid RAG classification status"""
        try:
            response = requests.get(f"{API_BASE}/hybrid-rag-classification/status", timeout=30)
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    service_data = data.get('data', {})
                    is_initialized = service_data.get('is_initialized', False)
                    total_categories = service_data.get('total_categories', 0)
                    total_issues = service_data.get('total_issue_types', 0)
                    
                    return {
                        'success': is_initialized,
                        'details': f"Initialized: {is_initialized}, Categories: {total_categories}, Issues: {total_issues}"
                    }
            return {'success': False, 'error': f"HTTP {response.status_code}"}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_hybrid_rag_categories(self) -> Dict:
        """Test hybrid RAG categories endpoint"""
        try:
            response = requests.get(f"{API_BASE}/hybrid-rag-classification/categories", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    categories_data = data.get('data', {})
                    categories = categories_data.get('categories', [])
                    return {
                        'success': len(categories) > 0,
                        'details': f"Found {len(categories)} categories: {', '.join(categories[:3])}..."
                    }
            return {'success': False, 'error': f"HTTP {response.status_code}"}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_hybrid_rag_text_classification(self) -> Dict:
        """Test hybrid RAG text classification"""
        test_cases = [
            {
                "subject": "Request for Extension of Time - Project Milestone 3",
                "body": "We request a 30-day extension for project milestone 3 due to unforeseen weather delays."
            },
            {
                "subject": "Payment Delay Notification - Invoice #2024-001", 
                "body": "Payment for invoice #2024-001 will be delayed by 15 days due to pending approvals."
            }
        ]
        
        successful_classifications = 0
        
        for i, test_case in enumerate(test_cases, 1):
            try:
                response = requests.post(
                    f"{API_BASE}/hybrid-rag-classification/classify-text",
                    json={
                        "subject": test_case["subject"],
                        "body": test_case["body"],
                        "options": {
                            "confidence_threshold": 0.3,
                            "max_results": 3
                        }
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success'):
                        result = data.get('data', {})
                        categories = result.get('categories', [])
                        if len(categories) > 0:
                            successful_classifications += 1
                            
            except Exception as e:
                self.print_warning(f"Classification test {i} failed: {e}")
        
        return {
            'success': successful_classifications > 0,
            'details': f"Successfully classified {successful_classifications}/{len(test_cases)} texts"
        }
    
    def test_hybrid_rag_batch_classification(self) -> Dict:
        """Test hybrid RAG batch classification"""
        batch_texts = [
            {
                "subject": "Utility Shifting Request - Phase 1",
                "body": "Request for utility shifting to proceed with Phase 1 construction."
            },
            {
                "subject": "Quality Control Issues - Materials Testing",
                "body": "Several quality control issues identified in materials testing."
            }
        ]
        
        try:
            response = requests.post(
                f"{API_BASE}/hybrid-rag-classification/classify-batch",
                json={
                    "texts": batch_texts,
                    "options": {
                        "confidence_threshold": 0.3,
                        "max_results": 2
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    result = data.get('data', {})
                    total_items = result.get('total_items', 0)
                    successful_items = result.get('successful_items', 0)
                    
                    return {
                        'success': successful_items == total_items and total_items > 0,
                        'details': f"Batch processed {successful_items}/{total_items} items"
                    }
            
            return {'success': False, 'error': f"HTTP {response.status_code}"}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    # ================================================================================  
    # ORCHESTRATOR TESTS
    # ================================================================================
    
    def test_orchestrator_status(self) -> Dict:
        """Test orchestrator status"""
        try:
            response = requests.get(f"{API_BASE}/orchestrator/status", timeout=30)
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    status_data = data.get('data', {})
                    services = status_data.get('services', {})
                    available_services = sum(1 for s in services.values() if s.get('available', False))
                    
                    return {
                        'success': available_services > 0,
                        'details': f"Orchestrator status OK, {available_services} services available"
                    }
            return {'success': False, 'error': f"HTTP {response.status_code}"}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    # ================================================================================
    # CONFIGURATION TESTS
    # ================================================================================
    
    def test_configuration_validation(self) -> Dict:
        """Test configuration validation"""
        try:
            response = requests.get(f"{API_BASE}/config/validate", timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Configuration validation might have warnings but still be successful
                return {
                    'success': True,
                    'details': f"Configuration validated: {data.get('status', 'unknown')}"
                }
            return {'success': False, 'error': f"HTTP {response.status_code}"}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_configuration_services(self) -> Dict:
        """Test configuration services endpoint"""
        try:
            response = requests.get(f"{API_BASE}/config/services", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    services_config = data.get('data', {})
                    return {
                        'success': len(services_config) > 0,
                        'details': f"Found {len(services_config)} service configurations"
                    }
            return {'success': False, 'error': f"HTTP {response.status_code}"}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    # ================================================================================
    # MAIN TEST RUNNER
    # ================================================================================
    
    def run_all_tests(self):
        """Run the complete regression test suite"""
        self.print_header("INTEGRATED CCMS BACKEND REGRESSION TEST SUITE")
        self.print_colored(Colors.BLUE, f"Testing backend at: {BASE_URL}")
        self.print_colored(Colors.BLUE, f"Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Test Categories
        test_groups = [
            ("Health & Status Tests", [
                ("Health Endpoint", self.test_health_endpoint),
                ("API Documentation", self.test_api_documentation),
            ]),
            
            ("Document Type Classification Tests", [
                ("Document Type from Text", self.test_document_type_classification_text),
            ]),
            
            ("OCR Service Tests", [
                ("OCR Methods", self.test_ocr_methods),
            ]),
            
            ("LLM Service Tests", [
                ("LLM Status", self.test_llm_status),
                ("LLM Structured Extraction", self.test_llm_structured_extraction),
            ]),
            
            ("Category Mapping Service Tests", [
                ("Available Categories", self.test_category_mapping_categories),
                ("Issue Mapping", self.test_category_mapping_issue), 
                ("Mapping Statistics", self.test_category_mapping_statistics),
            ]),
            
            ("Hybrid RAG Classification Tests", [
                ("Classification Status", self.test_hybrid_rag_status),
                ("Available Categories", self.test_hybrid_rag_categories),
                ("Text Classification", self.test_hybrid_rag_text_classification),
                ("Batch Classification", self.test_hybrid_rag_batch_classification),
            ]),
            
            ("Orchestrator Tests", [
                ("Orchestrator Status", self.test_orchestrator_status),
            ]),
            
            ("Configuration Tests", [
                ("Configuration Validation", self.test_configuration_validation),
                ("Services Configuration", self.test_configuration_services),
            ])
        ]
        
        # Run all test groups
        for group_name, tests in test_groups:
            self.print_colored(Colors.PURPLE, f"\n📋 {group_name}")
            self.print_colored(Colors.PURPLE, "-" * 60)
            
            for test_name, test_func in tests:
                self.run_test(test_name, test_func)
        
        # Generate final report
        self.generate_final_report()
    
    def generate_final_report(self):
        """Generate comprehensive test report"""
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        passed_tests = [r for r in self.results if r.passed]
        failed_tests = [r for r in self.results if not r.passed]
        
        self.print_header("REGRESSION TEST RESULTS SUMMARY")
        
        # Overall Statistics
        self.print_colored(Colors.BOLD, f"📊 Test Results:")
        self.print_colored(Colors.GREEN, f"   ✅ Passed: {len(passed_tests)}")
        self.print_colored(Colors.RED, f"   ❌ Failed: {len(failed_tests)}")
        self.print_colored(Colors.BLUE, f"   📈 Total: {len(self.results)}")
        self.print_colored(Colors.BLUE, f"   ⏱️  Duration: {total_duration:.2f} seconds")
        self.print_colored(Colors.BLUE, f"   📊 Success Rate: {len(passed_tests)/len(self.results)*100:.1f}%")
        
        # Failed Tests Details
        if failed_tests:
            self.print_colored(Colors.RED, f"\n❌ Failed Tests:")
            for test in failed_tests:
                self.print_colored(Colors.RED, f"   • {test.name}: {test.error or 'Unknown error'}")
        
        # Performance Summary
        self.print_colored(Colors.CYAN, f"\n⚡ Performance Summary:")
        avg_duration = sum(r.duration for r in self.results) / len(self.results)
        fastest_test = min(self.results, key=lambda r: r.duration)
        slowest_test = max(self.results, key=lambda r: r.duration)
        
        self.print_colored(Colors.CYAN, f"   📊 Average test duration: {avg_duration:.2f}s")
        self.print_colored(Colors.CYAN, f"   🚀 Fastest test: {fastest_test.name} ({fastest_test.duration:.2f}s)")
        self.print_colored(Colors.CYAN, f"   🐌 Slowest test: {slowest_test.name} ({slowest_test.duration:.2f}s)")
        
        # Service Health Summary
        if passed_tests:
            self.print_colored(Colors.GREEN, f"\n🏥 Service Health Summary:")
            service_groups = {}
            for test in passed_tests:
                service = test.name.split(' ')[0].lower()
                service_groups[service] = service_groups.get(service, 0) + 1
            
            for service, count in service_groups.items():
                self.print_colored(Colors.GREEN, f"   ✅ {service.capitalize()}: {count} tests passed")
        
        # Final Status
        if len(failed_tests) == 0:
            self.print_colored(Colors.BOLD + Colors.GREEN, f"\n🎉 ALL TESTS PASSED! Backend is fully functional.")
        elif len(failed_tests) < len(self.results) * 0.2:  # Less than 20% failures
            self.print_colored(Colors.BOLD + Colors.YELLOW, f"\n⚠️  MOSTLY FUNCTIONAL with {len(failed_tests)} minor issues.")
        else:
            self.print_colored(Colors.BOLD + Colors.RED, f"\n❌ SIGNIFICANT ISSUES DETECTED. {len(failed_tests)} tests failed.")
        
        # Next Steps
        self.print_colored(Colors.BLUE, f"\n💡 Next Steps:")
        if failed_tests:
            self.print_colored(Colors.BLUE, f"   1. Review failed tests and fix underlying issues")
            self.print_colored(Colors.BLUE, f"   2. Check service logs for detailed error information")
            self.print_colored(Colors.BLUE, f"   3. Verify API keys and configuration settings")
        else:
            self.print_colored(Colors.BLUE, f"   1. Backend is ready for production use!")
            self.print_colored(Colors.BLUE, f"   2. Continue with frontend integration testing")
            self.print_colored(Colors.BLUE, f"   3. Set up monitoring for production deployment")
        
        self.print_colored(Colors.BLUE, f"\n" + "="*80)

def main():
    """Main entry point"""
    print(f"🚀 Starting CCMS Backend Regression Test Suite...")
    print(f"📍 Backend URL: {BASE_URL}")
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if backend is reachable
    try:
        response = requests.get(BASE_URL, timeout=5)
        print(f"✅ Backend is reachable (HTTP {response.status_code})")
    except Exception as e:
        print(f"❌ Cannot reach backend at {BASE_URL}: {e}")
        print(f"💡 Make sure the backend is running:")
        print(f"   ./start_integrated_backend.sh --force")
        sys.exit(1)
    
    # Run the test suite
    suite = RegressionTestSuite()
    suite.run_all_tests()

if __name__ == "__main__":
    main()