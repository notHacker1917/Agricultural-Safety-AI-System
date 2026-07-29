#!/usr/bin/env python3
"""
DISPLAY VALIDATION RESULTS

Shows the key performance metrics from the agricultural safety system validation.
"""

import json
import os

def main():
    # Find the latest validation report
    log_dir = os.path.expanduser("~/safety_logs")
    reports = [f for f in os.listdir(log_dir) if f.startswith("real_world_validation_report_")]
    if not reports:
        print("No validation reports found!")
        return

    latest_report = max(reports)
    report_path = os.path.join(log_dir, latest_report)

    with open(report_path, 'r') as f:
        report = json.load(f)

    print("="*80)
    print("AGRICULTURAL SAFETY SYSTEM - FINAL VALIDATION RESULTS")
    print("="*80)

    # Dataset info
    summary = report.get('summary', {})
    print("\n📊 DATASET OVERVIEW:")
    print(f"   Images Processed: {summary.get('images_processed', 0)}")
    print(f"   Total Detections: {summary.get('total_detections', 0)}")
    print(f"   Ground Truth Annotations: {summary.get('total_ground_truth', 0)}")

    # Confusion matrix
    cm = report.get('confusion_matrix', {})
    tp = cm.get('true_positives', 0)
    fp = cm.get('false_positives', 0)
    fn = cm.get('false_negatives', 0)

    print("\n📈 CONFUSION MATRIX:")
    print(f"   True Positives: {tp}")
    print(f"   False Positives: {fp}")
    print(f"   False Negatives: {fn}")

    # Calculate actual metrics
    if tp + fp > 0:
        precision = tp / (tp + fp)
        print(f"   Precision: {precision:.1%}")
    else:
        precision = 0.0

    if tp + fn > 0:
        recall = tp / (tp + fn)
        print(f"   Recall: {recall:.1%}")
    else:
        recall = 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
        print(f"   F1-Score: {f1:.1%}")

    # Distance-stratified performance
    print("\n📍 DISTANCE PERFORMANCE:")
    dist_perf = report.get('distance_stratified_performance', {})
    if dist_perf:
        for range_name, metrics in dist_perf.items():
            print(f"   {range_name.title()}: P={metrics.get('precision', 0):.1%}, R={metrics.get('recall', 0):.1%}, F1={metrics.get('f1_score', 0):.1%}")
    else:
        print("   No distance-stratified data available")

    # KPI Assessment
    print("\n🎯 KPI TARGET ASSESSMENT:")
    kpi_data = report.get('kpi_comparison', {})
    for key, data in kpi_data.items():
        target = data.get('target', 0)
        actual = data.get('actual', 0)
        gap = data.get('gap', 0)
        met = data.get('met', False)
        status = "✅ MET" if met else "❌ GAP"
        clean_key = key.replace('_', ' ').title()
        print(f"   {clean_key}: {actual:.1%} vs {target:.1%} target ({status}, gap: {gap:.1%})")

    # Recommendations
    recommendations = report.get('recommendations', [])
    if recommendations:
        print("\n💡 RECOMMENDATIONS:")
        for rec in recommendations:
            print(f"   • {rec}")

    print("\n" + "="*80)
    print("SYSTEM STATUS: Agricultural safety AI validation completed successfully!")
    print("The system demonstrates real-world performance with room for improvement.")
    print("="*80)

if __name__ == "__main__":
    main()