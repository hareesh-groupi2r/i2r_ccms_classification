#!/usr/bin/env python3
"""
Test Runner for CCMS Classification System
Discovers and runs all unit tests with detailed reporting
"""

import sys
import unittest
import os
from pathlib import Path
import time
from io import StringIO
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def discover_tests(test_dir='tests', pattern='test_*.py'):
    """
    Discover all test files in the specified directory.
    
    Args:
        test_dir: Directory containing test files
        pattern: Pattern to match test files
        
    Returns:
        TestSuite containing all discovered tests
    """
    loader = unittest.TestLoader()
    
    if Path(test_dir).exists():
        # Discover tests from tests directory and subdirectories
        suite = loader.discover(test_dir, pattern=pattern)
    else:
        # Fall back to current directory
        suite = loader.discover('.', pattern=pattern)
    
    return suite

def run_tests_with_coverage(test_suite):
    """
    Run tests with coverage reporting if coverage.py is available.
    
    Args:
        test_suite: TestSuite to run
        
    Returns:
        TestResult
    """
    try:
        import coverage
        
        # Initialize coverage
        cov = coverage.Coverage()
        cov.start()
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, buffer=True)
        result = runner.run(test_suite)
        
        # Stop coverage and generate report
        cov.stop()
        cov.save()
        
        print("\n" + "="*60)
        print("COVERAGE REPORT")
        print("="*60)
        cov.report()
        
        return result
        
    except ImportError:
        # Run without coverage
        print("Coverage.py not available, running tests without coverage reporting")
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, buffer=True)
        return runner.run(test_suite)

def run_specific_test(test_name):
    """
    Run a specific test class or method.
    
    Args:
        test_name: Name of test class or method to run
    """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Try to load as module.class.method or module.class
    try:
        # Import the test module
        if '.' in test_name:
            module_name, class_or_method = test_name.rsplit('.', 1)
            if module_name.startswith('test_'):
                module_name = f'tests.{module_name}'
            else:
                module_name = f'tests.test_{module_name}'
                
            module = __import__(module_name, fromlist=[class_or_method])
            
            if hasattr(module, class_or_method):
                test_class = getattr(module, class_or_method)
                if isinstance(test_class, type) and issubclass(test_class, unittest.TestCase):
                    # It's a test class
                    suite.addTests(loader.loadTestsFromTestCase(test_class))
                else:
                    # Try to find the method in test classes
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (isinstance(attr, type) and 
                            issubclass(attr, unittest.TestCase) and 
                            hasattr(attr, class_or_method)):
                            suite.addTest(attr(class_or_method))
                            break
        else:
            # Try to find test class by name
            test_modules = ['tests.test_correspondence_extraction', 'tests.test_batch_processor', 
                          'tests.test_classifiers', 'tests.test_pdf_extraction']
            
            for module_name in test_modules:
                try:
                    module = __import__(module_name, fromlist=[test_name])
                    if hasattr(module, test_name):
                        test_class = getattr(module, test_name)
                        suite.addTests(loader.loadTestsFromTestCase(test_class))
                        break
                except ImportError:
                    continue
    
    except Exception as e:
        print(f"Error loading test '{test_name}': {e}")
        return False
    
    if suite.countTestCases() == 0:
        print(f"No tests found for '{test_name}'")
        return False
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

def list_available_tests():
    """List all available test classes and methods."""
    print("Available Tests:")
    print("="*50)
    
    test_modules = [
        'tests.test_correspondence_extraction',
        'tests.test_batch_processor', 
        'tests.test_classifiers',
        'tests.test_pdf_extraction'
    ]
    
    # Also check integration and evaluation test directories
    integration_tests = list(Path('tests/integration').glob('test_*.py')) if Path('tests/integration').exists() else []
    evaluation_tests = list(Path('tests/evaluation').glob('test_*.py')) if Path('tests/evaluation').exists() else []
    debug_scripts = list(Path('tests/debug_scripts').glob('debug_*.py')) if Path('tests/debug_scripts').exists() else []
    
    for module_name in test_modules:
        try:
            module = __import__(module_name, fromlist=[''])
            print(f"\n📁 {module_name}:")
            
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, unittest.TestCase) and 
                    attr_name.startswith('Test')):
                    print(f"  🧪 {attr_name}")
                    
                    # List test methods
                    for method_name in dir(attr):
                        if method_name.startswith('test_'):
                            print(f"    • {method_name}")
        
        except ImportError as e:
            print(f"  ❌ Failed to import {module_name}: {e}")
    
    # List integration tests
    if integration_tests:
        print(f"\n📁 Integration Tests ({len(integration_tests)} files):")
        for test_file in integration_tests:
            print(f"  🧪 {test_file.name}")
    
    # List evaluation tests
    if evaluation_tests:
        print(f"\n📁 Evaluation Tests ({len(evaluation_tests)} files):")
        for test_file in evaluation_tests:
            print(f"  📊 {test_file.name}")
    
    # List debug scripts
    if debug_scripts:
        print(f"\n📁 Debug Scripts ({len(debug_scripts)} files):")
        for script_file in debug_scripts:
            print(f"  🔍 {script_file.name}")

def run_integration_tests():
    """Run integration tests that require external dependencies."""
    print("Running Integration Tests")
    print("="*40)
    
    # Check for data availability
    if not Path("data/Lot-11").exists():
        print("⚠️  Warning: Test data (data/Lot-11) not available")
        print("   Integration tests will be skipped")
    
    # Run tests that interact with real files
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add integration test cases
    try:
        from tests.test_pdf_extraction import TestPDFExtractionIntegration
        from tests.test_correspondence_extraction import TestCorrespondenceIntegration
        
        suite.addTests(loader.loadTestsFromTestCase(TestPDFExtractionIntegration))
        suite.addTests(loader.loadTestsFromTestCase(TestCorrespondenceIntegration))
        
    except ImportError as e:
        print(f"Could not import integration tests: {e}")
        return False
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description='CCMS Classification System Test Runner')
    parser.add_argument('--test', '-t', help='Run specific test class or method')
    parser.add_argument('--list', '-l', action='store_true', help='List available tests')
    parser.add_argument('--integration', '-i', action='store_true', help='Run integration tests')
    parser.add_argument('--coverage', '-c', action='store_true', help='Run with coverage reporting')
    parser.add_argument('--fast', '-f', action='store_true', help='Run only fast unit tests')
    parser.add_argument('--pattern', '-p', default='test_*.py', help='Test file pattern')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_tests()
        return
    
    if args.test:
        success = run_specific_test(args.test)
        sys.exit(0 if success else 1)
    
    if args.integration:
        success = run_integration_tests()
        sys.exit(0 if success else 1)
    
    # Run all tests
    print("CCMS Classification System - Test Runner")
    print("="*60)
    print(f"Python: {sys.version}")
    print(f"Test Discovery Pattern: {args.pattern}")
    print("="*60)
    
    start_time = time.time()
    
    # Discover and run tests
    test_suite = discover_tests(pattern=args.pattern)
    
    if test_suite.countTestCases() == 0:
        print("❌ No tests found!")
        sys.exit(1)
    
    print(f"Found {test_suite.countTestCases()} test(s)")
    print("-"*60)
    
    if args.coverage:
        result = run_tests_with_coverage(test_suite)
    else:
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, buffer=True)
        result = runner.run(test_suite)
    
    end_time = time.time()
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print(f"Time: {end_time - start_time:.2f}s")
    
    if result.failures:
        print(f"\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  • {test}: {traceback.split('AssertionError:')[-1].strip() if 'AssertionError:' in traceback else 'See details above'}")
    
    if result.errors:
        print(f"\n💥 ERRORS:")
        for test, traceback in result.errors:
            error_msg = traceback.split('\n')[-2] if traceback.split('\n')[-2] else traceback.split('\n')[-3]
            print(f"  • {test}: {error_msg}")
    
    if result.skipped:
        print(f"\n⏭️  SKIPPED:")
        for test, reason in result.skipped:
            print(f"  • {test}: {reason}")
    
    success = result.wasSuccessful()
    if success:
        print(f"\n✅ All tests passed!")
    else:
        print(f"\n❌ Some tests failed!")
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()