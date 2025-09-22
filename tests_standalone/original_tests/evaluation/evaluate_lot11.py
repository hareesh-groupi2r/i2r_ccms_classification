#!/usr/bin/env python3
"""
Comprehensive evaluation of Lot-11 PDFs with ground truth comparison
"""

import pandas as pd
import numpy as np
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import defaultdict
import sys
sys.path.append('.')

from classifier.config_manager import ConfigManager
from classifier.issue_mapper import IssueCategoryMapper
from classifier.validation import ValidationEngine
from classifier.data_sufficiency import DataSufficiencyAnalyzer
from classifier.pure_llm import PureLLMClassifier
from classifier.preprocessing import TextPreprocessor
from classifier.pdf_extractor import PDFExtractor
from classifier.category_normalizer import CategoryNormalizer

def load_ground_truth(excel_path: Path) -> Dict[str, Set[str]]:
    """
    Load and normalize ground truth categories from Excel file
    
    Returns:
        Dict mapping filename to set of normalized categories
    """
    print("📊 Loading ground truth from Excel file...")
    
    df = pd.read_excel(excel_path)
    
    # Skip header row and get filename (column C) and categories (column F)
    filename_col = df.iloc[1:, 2]  # Column C, skip header
    category_col = df.iloc[1:, 5]  # Column F, skip header
    
    # Initialize category normalizer
    normalizer = CategoryNormalizer()
    
    ground_truth = defaultdict(set)
    
    for filename, categories in zip(filename_col, category_col):
        if pd.isna(filename) or pd.isna(categories):
            continue
            
        # Add .pdf extension if missing
        if not filename.endswith('.pdf'):
            filename = filename + '.pdf'
            
        # Split multiple categories and normalize each
        if isinstance(categories, str):
            raw_categories = [cat.strip() for cat in categories.split(',')]
            normalized_cats = set()
            
            for raw_cat in raw_categories:
                if raw_cat:  # Skip empty categories
                    normalized_cat = normalizer.normalize_category(raw_cat)
                    if normalized_cat:
                        normalized_cats.add(normalized_cat)
            
            if normalized_cats:  # Only add if we have valid categories
                ground_truth[filename].update(normalized_cats)
    
    print(f"✅ Loaded ground truth for {len(ground_truth)} files")
    
    # Show category distribution
    all_categories = set()
    for cats in ground_truth.values():
        all_categories.update(cats)
    
    print(f"📊 Found {len(all_categories)} unique normalized categories:")
    for cat in sorted(all_categories):
        count = sum(1 for cats in ground_truth.values() if cat in cats)
        print(f"  - {cat}: {count} files")
    
    return dict(ground_truth)

def classify_lot11_pdfs(pdf_dir: Path, classifier: PureLLMClassifier) -> Dict[str, Tuple[List[str], str]]:
    """
    Classify all PDFs in Lot-11 directory
    
    Returns:
        Dict mapping filename to (categories, llm_provider)
    """
    print(f"\n🚀 Classifying PDFs in {pdf_dir}")
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    pdf_files = [f for f in pdf_files if f.name.endswith('.pdf') and not f.name.startswith('.')]
    
    print(f"📄 Found {len(pdf_files)} PDF files to classify")
    
    results = {}
    
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n📄 [{i}/{len(pdf_files)}] Processing: {pdf_file.name}")
        
        try:
            start_time = time.time()
            
            # Extract text from PDF
            pdf_extractor = PDFExtractor()
            extracted_data = pdf_extractor.extract_text_from_pdf(str(pdf_file))
            
            if not extracted_data or not extracted_data.get('text'):
                print(f"❌ Failed to extract text from {pdf_file.name}")
                continue
            
            text = extracted_data['text']
            print(f"📝 Extracted {len(text)} characters")
            
            # Classify the document
            result, provider = classifier.classify_document(text)
            
            processing_time = time.time() - start_time
            
            # Extract categories from result
            categories = []
            if result and 'categories' in result:
                categories = [cat_info.get('category', '') for cat_info in result['categories']]
                categories = [cat for cat in categories if cat]  # Remove empty categories
            
            results[pdf_file.name] = (categories, provider)
            
            print(f"✅ Classified in {processing_time:.2f}s using {provider}")
            print(f"🏷️  Categories: {categories}")
            
        except Exception as e:
            print(f"❌ Error processing {pdf_file.name}: {e}")
            continue
    
    print(f"\n✅ Successfully classified {len(results)} PDFs")
    return results

def calculate_metrics(ground_truth: Dict[str, Set[str]], 
                     predictions: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics
    """
    print("\n📊 Calculating evaluation metrics...")
    
    # Find common files
    common_files = set(ground_truth.keys()) & set(predictions.keys())
    print(f"📋 Evaluating {len(common_files)} common files")
    
    if not common_files:
        print("❌ No common files found between ground truth and predictions")
        return {}
    
    # Get all unique categories
    all_gt_categories = set()
    all_pred_categories = set()
    
    for filename in common_files:
        all_gt_categories.update(ground_truth[filename])
        all_pred_categories.update(predictions[filename])
    
    all_categories = sorted(all_gt_categories | all_pred_categories)
    print(f"📊 Total unique categories: {len(all_categories)}")
    
    # Calculate per-file metrics (treat each file as multi-label classification)
    file_metrics = []
    detailed_results = []
    
    for filename in sorted(common_files):
        gt_cats = ground_truth[filename]
        pred_cats = set(predictions[filename])
        
        # Calculate metrics for this file
        true_positives = len(gt_cats & pred_cats)
        false_positives = len(pred_cats - gt_cats)
        false_negatives = len(gt_cats - pred_cats)
        true_negatives = len(all_categories) - len(gt_cats | pred_cats)
        
        # File-level metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (true_positives + true_negatives) / len(all_categories) if len(all_categories) > 0 else 0
        
        file_metrics.append({
            'filename': filename,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'tp': true_positives,
            'fp': false_positives,
            'fn': false_negatives,
            'tn': true_negatives
        })
        
        detailed_results.append({
            'filename': filename,
            'ground_truth': sorted(gt_cats),
            'predicted': sorted(pred_cats),
            'correct': sorted(gt_cats & pred_cats),
            'missed': sorted(gt_cats - pred_cats),
            'extra': sorted(pred_cats - gt_cats)
        })
    
    # Calculate overall metrics
    total_tp = sum(m['tp'] for m in file_metrics)
    total_fp = sum(m['fp'] for m in file_metrics)
    total_fn = sum(m['fn'] for m in file_metrics)
    total_tn = sum(m['tn'] for m in file_metrics)
    
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    overall_accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn) if (total_tp + total_fp + total_fn + total_tn) > 0 else 0
    
    # Average metrics across files
    avg_precision = np.mean([m['precision'] for m in file_metrics])
    avg_recall = np.mean([m['recall'] for m in file_metrics])
    avg_f1 = np.mean([m['f1'] for m in file_metrics])
    avg_accuracy = np.mean([m['accuracy'] for m in file_metrics])
    
    metrics = {
        'overall_metrics': {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1,
            'accuracy': overall_accuracy,
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'total_tn': total_tn
        },
        'average_metrics': {
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1': avg_f1,
            'avg_accuracy': avg_accuracy
        },
        'file_metrics': file_metrics,
        'detailed_results': detailed_results
    }
    
    return metrics

def print_evaluation_report(metrics: Dict, predictions: Dict[str, Tuple[List[str], str]]):
    """
    Print comprehensive evaluation report
    """
    print("\n" + "="*80)
    print("📊 LOT-11 EVALUATION REPORT")
    print("="*80)
    
    overall = metrics['overall_metrics']
    average = metrics['average_metrics']
    
    print(f"\n🎯 OVERALL METRICS (Micro-averaged)")
    print(f"   Accuracy:  {overall['accuracy']:.3f}")
    print(f"   Precision: {overall['precision']:.3f}")
    print(f"   Recall:    {overall['recall']:.3f}")
    print(f"   F1-Score:  {overall['f1_score']:.3f}")
    
    print(f"\n📊 AVERAGE METRICS (Macro-averaged)")
    print(f"   Accuracy:  {average['avg_accuracy']:.3f}")
    print(f"   Precision: {average['avg_precision']:.3f}")
    print(f"   Recall:    {average['avg_recall']:.3f}")
    print(f"   F1-Score:  {average['avg_f1']:.3f}")
    
    print(f"\n📋 CONFUSION MATRIX TOTALS")
    print(f"   True Positives:  {overall['total_tp']}")
    print(f"   False Positives: {overall['total_fp']}")
    print(f"   False Negatives: {overall['total_fn']}")
    print(f"   True Negatives:  {overall['total_tn']}")
    
    # LLM Provider usage
    provider_usage = defaultdict(int)
    for categories, provider in predictions.values():
        provider_usage[provider] += 1
    
    print(f"\n🤖 LLM PROVIDER USAGE")
    for provider, count in sorted(provider_usage.items()):
        percentage = count / len(predictions) * 100
        print(f"   {provider}: {count} files ({percentage:.1f}%)")
    
    # Show top performing and worst performing files
    file_metrics = sorted(metrics['file_metrics'], key=lambda x: x['f1'], reverse=True)
    
    print(f"\n🏆 TOP 5 PERFORMING FILES (by F1-Score)")
    for i, fm in enumerate(file_metrics[:5]):
        print(f"   {i+1}. {fm['filename'][:60]}...")
        print(f"      F1: {fm['f1']:.3f}, P: {fm['precision']:.3f}, R: {fm['recall']:.3f}")
    
    print(f"\n❌ BOTTOM 5 PERFORMING FILES (by F1-Score)")
    for i, fm in enumerate(file_metrics[-5:]):
        print(f"   {i+1}. {fm['filename'][:60]}...")
        print(f"      F1: {fm['f1']:.3f}, P: {fm['precision']:.3f}, R: {fm['recall']:.3f}")

def save_results(metrics: Dict, predictions: Dict[str, Tuple[List[str], str]], 
                 ground_truth: Dict[str, Set[str]], output_path: str):
    """
    Save detailed results to JSON file
    """
    print(f"\n💾 Saving results to {output_path}")
    
    # Convert predictions to serializable format
    serializable_predictions = {}
    for filename, (categories, provider) in predictions.items():
        serializable_predictions[filename] = {
            'categories': categories,
            'llm_provider': provider
        }
    
    # Convert ground truth to serializable format
    serializable_gt = {}
    for filename, categories in ground_truth.items():
        serializable_gt[filename] = sorted(list(categories))
    
    results = {
        'evaluation_summary': {
            'total_files_evaluated': len(set(ground_truth.keys()) & set(predictions.keys())),
            'total_files_classified': len(predictions),
            'total_files_in_ground_truth': len(ground_truth)
        },
        'metrics': metrics,
        'predictions': serializable_predictions,
        'ground_truth': serializable_gt,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Results saved to {output_path}")

def main():
    """
    Main evaluation pipeline
    """
    print("🚀 Starting Lot-11 Evaluation Pipeline")
    print("="*50)
    
    # Set up environment
    os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY', '')
    
    # Paths
    lot11_dir = Path("data/Lot-11")
    excel_path = lot11_dir / "EDMS-Lot 11.xlsx"
    results_path = "results/lot11_evaluation_results.json"
    
    # Create results directory
    Path("results").mkdir(exist_ok=True)
    
    try:
        # Load ground truth
        ground_truth = load_ground_truth(excel_path)
        
        # Initialize classifier
        print("\n🔧 Initializing classifier...")
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        # Load training data for mappings
        training_data_path = Path("./data/synthetic/combined_training_data.xlsx")
        issue_mapper = IssueCategoryMapper(training_data_path)
        validator = ValidationEngine(training_data_path)
        data_analyzer = DataSufficiencyAnalyzer(training_data_path)
        
        classifier = PureLLMClassifier(
            config=config['classifiers']['pure_llm'],
            issue_mapper=issue_mapper,
            validator=validator,
            data_analyzer=data_analyzer
        )
        
        print("✅ Classifier initialized")
        
        # Classify PDFs
        predictions = classify_lot11_pdfs(lot11_dir, classifier)
        
        # Convert predictions to just categories for metrics calculation
        pred_categories = {k: v[0] for k, v in predictions.items()}
        
        # Calculate metrics
        metrics = calculate_metrics(ground_truth, pred_categories)
        
        # Print report
        print_evaluation_report(metrics, predictions)
        
        # Save results
        save_results(metrics, predictions, ground_truth, results_path)
        
        print(f"\n🎉 Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in evaluation pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()