#!/usr/bin/env python3
"""
Batch processing script for individual lots using integrated backend API
"""

import sys
import os
from pathlib import Path
import argparse
import logging
import json
import requests
import shutil
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
INTEGRATED_BACKEND_URL = "http://localhost:5001"

def check_backend_health():
    """Check if integrated backend is running and healthy."""
    try:
        response = requests.get(f"{INTEGRATED_BACKEND_URL}/api/services/health", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Integrated backend is running")
            return True
        else:
            logger.error(f"❌ Integrated backend health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Cannot connect to integrated backend: {e}")
        logger.error("Please start the integrated backend server first:")
        logger.error("  cd integrated_backend && python api/app.py")
        return False

def get_pdf_files(pdf_folder, limit=None):
    """Get list of PDF files, optionally limited to first N files."""
    pdf_folder = Path(pdf_folder)
    pdf_files = sorted(pdf_folder.glob("*.pdf"))
    
    if limit:
        pdf_files = pdf_files[:limit]
        logger.info(f"📄 Limited to first {limit} files out of {len(sorted(pdf_folder.glob('*.pdf')))} total")
    
    return pdf_files

def call_integrated_backend_api(pdf_folder, ground_truth_file, output_folder, enable_metrics):
    """Call the integrated backend API to process the PDF folder."""
    
    # Build API payload
    payload = {
        "pdf_folder": str(pdf_folder),
        "output_folder": str(output_folder),
        "options": {
            "approaches": ["hybrid_rag"],
            "confidence_threshold": 0.3,
            "max_pages": 2
        },
        "enable_metrics": enable_metrics
    }
    
    # Add ground truth if provided
    if ground_truth_file and Path(ground_truth_file).exists():
        payload["ground_truth_file"] = str(ground_truth_file)
        logger.info(f"📊 Using ground truth file: {ground_truth_file}")
    
    logger.info(f"🔗 Calling integrated backend API...")
    logger.info(f"📦 Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(
            f"{INTEGRATED_BACKEND_URL}/api/services/hybrid-rag-classification/process-folder",
            json=payload,
            timeout=3600  # 1 hour timeout for large batches
        )
        
        if response.status_code == 200:
            logger.info("✅ API call successful")
            return True, response.json()
        else:
            logger.error(f"❌ API call failed with status {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False, None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ API request failed: {e}")
        return False, None

def main():
    parser = argparse.ArgumentParser(description='Process a lot of PDFs for classification using integrated backend')
    parser.add_argument('--pdf-folder', required=True, help='Path to folder containing PDF files')
    parser.add_argument('--ground-truth', help='Path to ground truth Excel file (optional)')
    parser.add_argument('--output-folder', required=True, help='Output folder for results')
    parser.add_argument('--lot-name', required=True, help='Name of the lot for logging')
    parser.add_argument('--enable-llm', action='store_true', help='Enable Pure LLM approach (currently unused - integrated backend uses hybrid_rag)')
    parser.add_argument('--disable-metrics', action='store_true', help='Disable metrics calculation')
    parser.add_argument('--limit', type=int, help='Process only first N files')
    
    args = parser.parse_args()
    
    logger.info(f"🚀 Starting batch processing for {args.lot_name} via integrated backend")
    logger.info(f"📁 PDF Folder: {args.pdf_folder}")
    logger.info(f"📊 Ground Truth: {args.ground_truth or 'Auto-detect/None'}")
    logger.info(f"📁 Output Folder: {args.output_folder}")
    logger.info(f"🤖 Approach: Hybrid RAG (via integrated backend)")
    logger.info(f"📈 Metrics: {'Disabled' if args.disable_metrics else 'Enabled'}")
    if args.limit:
        logger.info(f"📄 File Limit: First {args.limit} files")
    
    try:
        # Check integrated backend health
        if not check_backend_health():
            return 1
        
        # Check if ground truth file exists and validate it
        if args.ground_truth:
            if Path(args.ground_truth).exists():
                logger.info(f"✅ Ground truth file verified: {args.ground_truth}")
            else:
                logger.warning(f"⚠️  Ground truth file not found: {args.ground_truth}")
                logger.info(f"📊 Will process without ground truth")
                args.ground_truth = None
        else:
            logger.info(f"📊 No ground truth file provided")
        
        # Get PDF files with optional limit
        pdf_files = get_pdf_files(args.pdf_folder, args.limit)
        if not pdf_files:
            logger.error(f"❌ No PDF files found in {args.pdf_folder}")
            return 1
        
        logger.info(f"📄 Found {len(pdf_files)} PDF files to process")
        
        # Create actual PDF folder for processing
        actual_pdf_folder = args.pdf_folder
        temp_folder = None
        
        if args.limit:
            # Create temporary folder with limited files
            temp_folder = Path(tempfile.mkdtemp(prefix=f"temp_limited_lot_{args.lot_name}_"))
            
            # Copy only the first N PDF files
            pdf_files = get_pdf_files(args.pdf_folder, args.limit)
            for pdf_file in pdf_files:
                shutil.copy2(pdf_file, temp_folder / pdf_file.name)
            
            # Copy ground truth file if it exists in the original folder
            if args.ground_truth and Path(args.ground_truth).exists():
                shutil.copy2(args.ground_truth, temp_folder / Path(args.ground_truth).name)
                args.ground_truth = str(temp_folder / Path(args.ground_truth).name)
            
            actual_pdf_folder = str(temp_folder)
            logger.info(f"📄 Created temporary folder with {len(pdf_files)} files: {actual_pdf_folder}")
        
        # Create output folder
        Path(args.output_folder).mkdir(parents=True, exist_ok=True)
        
        # Call integrated backend API
        logger.info(f"🔄 Processing {args.lot_name} via integrated backend...")
        success, response_data = call_integrated_backend_api(
            actual_pdf_folder,
            args.ground_truth,
            args.output_folder,
            not args.disable_metrics
        )
        
        # Clean up temporary folder if created
        if temp_folder and temp_folder.exists():
            shutil.rmtree(temp_folder)
            logger.info(f"🧹 Cleaned up temporary folder")
        
        if success and response_data:
            # Log results summary from API response
            stats = response_data.get('processing_stats', {})
            logger.info(f"✅ {args.lot_name} processing completed:")
            logger.info(f"   📄 Total files: {stats.get('total_files', 0)}")
            logger.info(f"   ✅ Processed: {stats.get('processed_files', 0)}")
            logger.info(f"   ❌ Failed: {stats.get('failed_files', 0)}")
            
            # Log metrics if available
            overall_metrics = response_data.get('overall_metrics', {})
            if overall_metrics:
                for approach, metric_data in overall_metrics.items():
                    logger.info(f"   📊 {approach.replace('_', ' ').title()} Metrics:")
                    for metric_name, value in metric_data.items():
                        if isinstance(value, (int, float)):
                            logger.info(f"      {metric_name}: {value:.3f}")
                        else:
                            logger.info(f"      {metric_name}: {value}")
            
            logger.info(f"📁 Results saved to: {args.output_folder}")
            return 0
        else:
            logger.error(f"❌ {args.lot_name} processing failed")
            return 1
        
    except Exception as e:
        logger.error(f"❌ Error processing {args.lot_name}: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())