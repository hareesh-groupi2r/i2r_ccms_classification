#!/usr/bin/env python3
"""
Metrics Calculator for Classification Results
Handles precision, recall, F1-score, and other evaluation metrics
"""

from typing import Dict, List, Set, Tuple
import logging
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculate evaluation metrics for multi-label classification results.
    """
    
    def __init__(self):
        """Initialize the metrics calculator."""
        pass
    
    def calculate_metrics(self, ground_truth: List[str], predicted: List[str]) -> Dict:
        """
        Calculate metrics for a single prediction.
        
        Args:
            ground_truth: List of true categories
            predicted: List of predicted categories
            
        Returns:
            Dictionary containing various metrics
        """
        # Convert to sets for easier comparison
        gt_set = set(ground_truth) if ground_truth else set()
        pred_set = set(predicted) if predicted else set()
        
        # Calculate basic metrics
        tp = len(gt_set.intersection(pred_set))  # True Positives
        fp = len(pred_set - gt_set)              # False Positives  
        fn = len(gt_set - pred_set)              # False Negatives
        tn = 0  # True Negatives (not meaningful in multi-label)
        
        # Calculate precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Exact match accuracy (all categories must match exactly)
        exact_match = 1.0 if gt_set == pred_set else 0.0
        
        # Jaccard similarity (IoU)
        union = gt_set.union(pred_set)
        jaccard = len(gt_set.intersection(pred_set)) / len(union) if union else 1.0
        
        return {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1_score, 4),
            'exact_match': exact_match,
            'jaccard_similarity': round(jaccard, 4),
            'ground_truth_count': len(gt_set),
            'predicted_count': len(pred_set),
            'correct_predictions': tp,
            'missed_categories': list(gt_set - pred_set),
            'extra_categories': list(pred_set - gt_set),
            'correct_categories': list(gt_set.intersection(pred_set))
        }
    
    def calculate_batch_metrics(self, ground_truth_list: List[List[str]], 
                               predicted_list: List[List[str]]) -> Dict:
        """
        Calculate aggregated metrics across multiple predictions.
        
        Args:
            ground_truth_list: List of ground truth category lists
            predicted_list: List of predicted category lists
            
        Returns:
            Dictionary containing aggregated metrics
        """
        if len(ground_truth_list) != len(predicted_list):
            raise ValueError("Ground truth and predicted lists must have the same length")
        
        # Calculate individual metrics
        individual_metrics = []
        for gt, pred in zip(ground_truth_list, predicted_list):
            metrics = self.calculate_metrics(gt, pred)
            individual_metrics.append(metrics)
        
        # Aggregate metrics
        total_files = len(individual_metrics)
        
        # Micro-averaging (aggregate TP, FP, FN across all samples)
        total_tp = sum(m['tp'] for m in individual_metrics)
        total_fp = sum(m['fp'] for m in individual_metrics)
        total_fn = sum(m['fn'] for m in individual_metrics)
        
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
        
        # Macro-averaging (average of individual metrics)
        macro_precision = np.mean([m['precision'] for m in individual_metrics])
        macro_recall = np.mean([m['recall'] for m in individual_metrics])
        macro_f1 = np.mean([m['f1_score'] for m in individual_metrics])
        
        # Other aggregated metrics
        exact_match_accuracy = np.mean([m['exact_match'] for m in individual_metrics])
        average_jaccard = np.mean([m['jaccard_similarity'] for m in individual_metrics])
        
        # Category-level analysis
        category_stats = self._calculate_category_stats(ground_truth_list, predicted_list)
        
        return {
            'total_files': total_files,
            'micro_precision': round(micro_precision, 4),
            'micro_recall': round(micro_recall, 4),
            'micro_f1': round(micro_f1, 4),
            'macro_precision': round(macro_precision, 4),
            'macro_recall': round(macro_recall, 4),
            'macro_f1': round(macro_f1, 4),
            'exact_match_accuracy': round(exact_match_accuracy, 4),
            'average_jaccard_similarity': round(average_jaccard, 4),
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'perfect_predictions': sum(1 for m in individual_metrics if m['exact_match'] == 1.0),
            'category_statistics': category_stats,
            'individual_metrics': individual_metrics
        }
    
    def _calculate_category_stats(self, ground_truth_list: List[List[str]], 
                                 predicted_list: List[List[str]]) -> Dict:
        """Calculate per-category statistics."""
        # Collect all unique categories
        all_categories = set()
        for gt_list in ground_truth_list:
            all_categories.update(gt_list)
        for pred_list in predicted_list:
            all_categories.update(pred_list)
        
        category_stats = {}
        
        for category in all_categories:
            category_tp = 0
            category_fp = 0
            category_fn = 0
            category_appearances_gt = 0
            category_appearances_pred = 0
            
            for gt_list, pred_list in zip(ground_truth_list, predicted_list):
                gt_has = category in gt_list
                pred_has = category in pred_list
                
                if gt_has:
                    category_appearances_gt += 1
                if pred_has:
                    category_appearances_pred += 1
                
                if gt_has and pred_has:
                    category_tp += 1
                elif not gt_has and pred_has:
                    category_fp += 1
                elif gt_has and not pred_has:
                    category_fn += 1
            
            # Calculate precision, recall, F1 for this category
            cat_precision = category_tp / (category_tp + category_fp) if (category_tp + category_fp) > 0 else 0.0
            cat_recall = category_tp / (category_tp + category_fn) if (category_tp + category_fn) > 0 else 0.0
            cat_f1 = 2 * (cat_precision * cat_recall) / (cat_precision + cat_recall) if (cat_precision + cat_recall) > 0 else 0.0
            
            category_stats[category] = {
                'precision': round(cat_precision, 4),
                'recall': round(cat_recall, 4),
                'f1_score': round(cat_f1, 4),
                'tp': category_tp,
                'fp': category_fp,
                'fn': category_fn,
                'appearances_in_ground_truth': category_appearances_gt,
                'appearances_in_predictions': category_appearances_pred,
                'support': category_appearances_gt  # Number of true instances
            }
        
        return category_stats
    
    def compare_approaches(self, results: Dict, ground_truth: Dict) -> Dict:
        """
        Compare multiple approaches using the same ground truth.
        
        Args:
            results: Dictionary with approach names as keys and prediction lists as values
            ground_truth: Dictionary with file names as keys and category lists as values
            
        Returns:
            Dictionary containing comparison metrics
        """
        comparison = {}
        
        # Prepare ground truth and predictions for each approach
        for approach_name, approach_results in results.items():
            gt_lists = []
            pred_lists = []
            
            for file_result in approach_results:
                file_name = file_result['file_name']
                if file_name in ground_truth:
                    gt_lists.append(ground_truth[file_name])
                    pred_lists.append(file_result.get('categories', []))
            
            if gt_lists and pred_lists:
                approach_metrics = self.calculate_batch_metrics(gt_lists, pred_lists)
                comparison[approach_name] = approach_metrics
        
        # Add relative comparisons
        if len(comparison) > 1:
            comparison['_comparison'] = self._generate_approach_comparison(comparison)
        
        return comparison
    
    def _generate_approach_comparison(self, approach_metrics: Dict) -> Dict:
        """Generate relative comparison between approaches."""
        approaches = [k for k in approach_metrics.keys() if not k.startswith('_')]
        
        if len(approaches) < 2:
            return {}
        
        comparison = {}
        
        # Compare key metrics
        metrics_to_compare = ['micro_f1', 'macro_f1', 'exact_match_accuracy', 'micro_precision', 'micro_recall']
        
        for metric in metrics_to_compare:
            metric_values = {approach: approach_metrics[approach].get(metric, 0) for approach in approaches}
            best_approach = max(metric_values.keys(), key=lambda k: metric_values[k])
            
            comparison[f'best_{metric}'] = {
                'approach': best_approach,
                'value': metric_values[best_approach],
                'all_values': metric_values
            }
        
        return comparison
    
    def generate_report(self, metrics: Dict, include_individual: bool = False) -> str:
        """
        Generate a human-readable metrics report.
        
        Args:
            metrics: Metrics dictionary from calculate_batch_metrics
            include_individual: Whether to include individual file metrics
            
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("CLASSIFICATION METRICS REPORT")
        lines.append("=" * 50)
        lines.append(f"Total Files Processed: {metrics['total_files']}")
        lines.append(f"Perfect Predictions: {metrics['perfect_predictions']} ({metrics['perfect_predictions']/metrics['total_files']*100:.1f}%)")
        lines.append("")
        
        lines.append("OVERALL PERFORMANCE:")
        lines.append(f"  Micro F1-Score: {metrics['micro_f1']:.4f}")
        lines.append(f"  Macro F1-Score: {metrics['macro_f1']:.4f}")
        lines.append(f"  Exact Match Accuracy: {metrics['exact_match_accuracy']:.4f}")
        lines.append(f"  Average Jaccard Similarity: {metrics['average_jaccard_similarity']:.4f}")
        lines.append("")
        
        lines.append("PRECISION & RECALL:")
        lines.append(f"  Micro Precision: {metrics['micro_precision']:.4f}")
        lines.append(f"  Micro Recall: {metrics['micro_recall']:.4f}")
        lines.append(f"  Macro Precision: {metrics['macro_precision']:.4f}")
        lines.append(f"  Macro Recall: {metrics['macro_recall']:.4f}")
        lines.append("")
        
        lines.append("CONFUSION MATRIX TOTALS:")
        lines.append(f"  True Positives: {metrics['total_tp']}")
        lines.append(f"  False Positives: {metrics['total_fp']}")
        lines.append(f"  False Negatives: {metrics['total_fn']}")
        lines.append("")
        
        # Category-level statistics
        if 'category_statistics' in metrics and metrics['category_statistics']:
            lines.append("PER-CATEGORY PERFORMANCE:")
            cat_stats = metrics['category_statistics']
            
            # Sort by F1 score descending
            sorted_cats = sorted(cat_stats.items(), key=lambda x: x[1]['f1_score'], reverse=True)
            
            lines.append(f"{'Category':<30} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Supp':>6}")
            lines.append("-" * 56)
            
            for cat_name, cat_data in sorted_cats[:10]:  # Top 10 categories
                lines.append(f"{cat_name:<30} {cat_data['f1_score']:>6.3f} {cat_data['precision']:>6.3f} {cat_data['recall']:>6.3f} {cat_data['support']:>6}")
            
            if len(sorted_cats) > 10:
                lines.append(f"... and {len(sorted_cats) - 10} more categories")
            lines.append("")
        
        # Individual file metrics (optional)
        if include_individual and 'individual_metrics' in metrics:
            lines.append("INDIVIDUAL FILE PERFORMANCE:")
            lines.append(f"{'File':<40} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Match':>6}")
            lines.append("-" * 70)
            
            for i, file_metrics in enumerate(metrics['individual_metrics'][:20]):  # First 20 files
                file_name = f"File_{i+1}" if 'file_name' not in file_metrics else file_metrics.get('file_name', f'File_{i+1}')
                lines.append(f"{file_name:<40} {file_metrics['f1_score']:>6.3f} {file_metrics['precision']:>6.3f} {file_metrics['recall']:>6.3f} {file_metrics['exact_match']:>6.0f}")
            
            if len(metrics['individual_metrics']) > 20:
                lines.append(f"... and {len(metrics['individual_metrics']) - 20} more files")
        
        return "\n".join(lines)