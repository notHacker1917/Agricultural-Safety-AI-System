#!/usr/bin/env python3
"""
VISUALIZE VALIDATION RESULTS

Creates comprehensive visualizations of agricultural safety system performance
against real-world HackHPI2026 dataset validation results.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from datetime import datetime

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ValidationVisualizer:
    def __init__(self, report_path=None):
        """Initialize visualizer with validation report."""
        if report_path is None:
            # Find the latest validation report
            log_dir = os.path.expanduser("~/safety_logs")
            reports = [f for f in os.listdir(log_dir) if f.startswith("real_world_validation_report_")]
            if reports:
                latest_report = max(reports)
                report_path = os.path.join(log_dir, latest_report)

        if report_path and os.path.exists(report_path):
            with open(report_path, 'r') as f:
                self.report = json.load(f)
            print(f"Loaded validation report: {report_path}")
        else:
            print("No validation report found. Run real_world_validation.py first.")
            self.report = None

    def create_kpi_comparison_plot(self):
        """Create KPI target vs actual performance visualization."""
        if not self.report:
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Agricultural Safety System - KPI Performance vs Targets', fontsize=16, fontweight='bold')

        # Data preparation
        kpi_data = self.report.get('kpi_comparison', {})

        ranges = ['close', 'medium', 'far']
        metrics = ['precision', 'recall']

        for range_name in ranges:
            for metric in metrics:
                key = f"{range_name}_{metric}_target"
                if key in kpi_data:
                    target = kpi_data[key]['target']
                    actual = kpi_data[key]['actual']
                    gap = kpi_data[key]['gap']

                    print(f"{range_name.title()} {metric}: {actual:.1%} vs {target:.1%} target (gap: {gap:.1%})")

        # Precision comparison
        ranges_labels = ['Close\n(<5m)', 'Medium\n(5-15m)', 'Far\n(15-50m)']
        precision_targets = [
            kpi_data.get('close_precision_target', {}).get('target', 0.94),
            kpi_data.get('medium_precision_target', {}).get('target', 0.9),
            kpi_data.get('far_precision_target', {}).get('target', 0.85)
        ]
        precision_actual = [
            kpi_data.get('close_precision_target', {}).get('actual', 0),
            kpi_data.get('medium_precision_target', {}).get('actual', 0),
            kpi_data.get('far_precision_target', {}).get('actual', 0)
        ]

        x = np.arange(len(ranges_labels))
        width = 0.35

        ax1.bar(x - width/2, precision_targets, width, label='Target', alpha=0.8, color='lightcoral')
        ax1.bar(x + width/2, precision_actual, width, label='Actual', alpha=0.8, color='skyblue')
        ax1.set_title('Precision by Distance Range', fontweight='bold')
        ax1.set_ylabel('Precision')
        ax1.set_xticks(x)
        ax1.set_xticklabels(ranges_labels)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add value labels
        for i, (t, a) in enumerate(zip(precision_targets, precision_actual)):
            ax1.text(i - width/2, t + 0.01, f'{t:.1%}', ha='center', va='bottom', fontsize=9)
            ax1.text(i + width/2, a + 0.01, f'{a:.1%}', ha='center', va='bottom', fontsize=9)

        # Recall comparison
        recall_targets = [
            kpi_data.get('close_recall_target', {}).get('target', 0.95),
            kpi_data.get('medium_recall_target', {}).get('target', 0.85),
            kpi_data.get('far_recall_target', {}).get('target', 0.6)
        ]
        recall_actual = [
            kpi_data.get('close_recall_target', {}).get('actual', 0),
            kpi_data.get('medium_recall_target', {}).get('actual', 0),
            kpi_data.get('far_recall_target', {}).get('actual', 0)
        ]

        ax2.bar(x - width/2, recall_targets, width, label='Target', alpha=0.8, color='lightgreen')
        ax2.bar(x + width/2, recall_actual, width, label='Actual', alpha=0.8, color='orange')
        ax2.set_title('Recall by Distance Range', fontweight='bold')
        ax2.set_ylabel('Recall')
        ax2.set_xticks(x)
        ax2.set_xticklabels(ranges_labels)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add value labels
        for i, (t, a) in enumerate(zip(recall_targets, recall_actual)):
            ax2.text(i - width/2, t + 0.01, f'{t:.1%}', ha='center', va='bottom', fontsize=9)
            ax2.text(i + width/2, a + 0.01, f'{a:.1%}', ha='center', va='bottom', fontsize=9)

        # Performance gaps
        precision_gaps = [
            kpi_data.get('close_precision_target', {}).get('gap', 0),
            kpi_data.get('medium_precision_target', {}).get('gap', 0),
            kpi_data.get('far_precision_target', {}).get('gap', 0)
        ]
        recall_gaps = [
            kpi_data.get('close_recall_target', {}).get('gap', 0),
            kpi_data.get('medium_recall_target', {}).get('gap', 0),
            kpi_data.get('far_recall_target', {}).get('gap', 0)
        ]

        ax3.bar(x - width/2, precision_gaps, width, label='Precision Gap', alpha=0.8, color='red')
        ax3.bar(x + width/2, recall_gaps, width, label='Recall Gap', alpha=0.8, color='purple')
        ax3.set_title('Performance Gaps (Target - Actual)', fontweight='bold')
        ax3.set_ylabel('Gap')
        ax3.set_xticks(x)
        ax3.set_xticklabels(ranges_labels)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Overall system summary
        overall_stats = self.report.get('overall_metrics', {})
        precision_overall = overall_stats.get('precision', 0)
        recall_overall = overall_stats.get('recall', 0)
        f1_overall = overall_stats.get('f1_score', 0)

        # Create a simple bar chart instead of pie chart to avoid NaN issues
        metrics = ['Precision', 'Recall', 'F1-Score']
        values = [precision_overall, recall_overall, f1_overall]

        ax4.bar(metrics, values, alpha=0.8, color=['skyblue', 'lightgreen', 'orange'])
        ax4.set_title('Overall System Performance', fontweight='bold')
        ax4.set_ylabel('Score')
        ax4.set_ylim(0, 1)

        # Add value labels
        for i, v in enumerate(values):
            ax4.text(i, v + 0.01, f'{v:.1%}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        try:
            plt.savefig('validation_kpi_analysis.png', dpi=300, bbox_inches='tight')
            print("✅ Saved validation_kpi_analysis.png")
        except Exception as e:
            print(f"Could not save KPI plot: {e}")
        plt.close()

        print("\n📊 KPI Analysis saved as: validation_kpi_analysis.png")
        print(f"📈 Overall Performance: P={precision_overall:.1%}, R={recall_overall:.1%}, F1={f1_overall:.1%}")

    def create_distance_performance_plot(self):
        """Create distance-stratified performance visualization."""
        if not self.report:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Distance-Stratified Detection Performance', fontsize=16, fontweight='bold')

        # Distance performance data
        distance_perf = self.report.get('distance_stratified_performance', {})

        ranges = ['close', 'medium', 'far']
        precision_vals = []
        recall_vals = []
        f1_vals = []

        for range_name in ranges:
            if range_name in distance_perf:
                perf = distance_perf[range_name]
                precision_vals.append(perf.get('precision', 0))
                recall_vals.append(perf.get('recall', 0))
                f1_vals.append(perf.get('f1_score', 0))

        x = np.arange(len(ranges))
        width = 0.25

        # Precision, Recall, F1 by distance
        ax1.bar(x - width, precision_vals, width, label='Precision', alpha=0.8, color='skyblue')
        ax1.bar(x, recall_vals, width, label='Recall', alpha=0.8, color='lightgreen')
        ax1.bar(x + width, f1_vals, width, label='F1-Score', alpha=0.8, color='orange')

        ax1.set_title('Performance Metrics by Distance Range', fontweight='bold')
        ax1.set_xlabel('Distance Range')
        ax1.set_ylabel('Score')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Close\n(<5m)', 'Medium\n(5-15m)', 'Far\n(15-50m)'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add value labels
        for i, (p, r, f) in enumerate(zip(precision_vals, recall_vals, f1_vals)):
            ax1.text(i - width, p + 0.01, f'{p:.1%}', ha='center', va='bottom', fontsize=8)
            ax1.text(i, r + 0.01, f'{r:.1%}', ha='center', va='bottom', fontsize=8)
            ax1.text(i + width, f + 0.01, f'{f:.1%}', ha='center', va='bottom', fontsize=8)

        # Confusion matrix visualization
        confusion = self.report.get('confusion_matrix', {})
        tp = confusion.get('true_positives', 0)
        fp = confusion.get('false_positives', 0)
        fn = confusion.get('false_negatives', 0)

        # Create confusion matrix plot
        cm_data = np.array([[tp, fp], [fn, 0]])  # Note: TN not calculated in our validation
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', ax=ax2,
                   xticklabels=['Predicted\nPositive', 'Predicted\nNegative'],
                   yticklabels=['Actual\nPositive', 'Actual\nNegative'])
        ax2.set_title('Confusion Matrix', fontweight='bold')
        ax2.set_ylabel('Actual')
        ax2.set_xlabel('Predicted')

        plt.tight_layout()
        try:
            plt.savefig('distance_performance_analysis.png', dpi=300, bbox_inches='tight')
            print("✅ Saved distance_performance_analysis.png")
        except Exception as e:
            print(f"Could not save distance plot: {e}")
        plt.close()

        print("📊 Distance Performance saved as: distance_performance_analysis.png")

    def create_scenario_analysis_plot(self):
        """Create scenario-based performance analysis."""
        if not self.report:
            return

        scenario_perf = self.report.get('scenario_stratified_performance', {})

        if not scenario_perf:
            print("No scenario performance data available")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        scenarios = list(scenario_perf.keys())
        precision_vals = [scenario_perf[s].get('precision', 0) for s in scenarios]
        recall_vals = [scenario_perf[s].get('recall', 0) for s in scenarios]

        x = np.arange(len(scenarios))
        width = 0.35

        ax.bar(x - width/2, precision_vals, width, label='Precision', alpha=0.8, color='lightblue')
        ax.bar(x + width/2, recall_vals, width, label='Recall', alpha=0.8, color='lightgreen')

        ax.set_title('Performance by Agricultural Scenario', fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add value labels
        for i, (p, r) in enumerate(zip(precision_vals, recall_vals)):
            ax.text(i - width/2, p + 0.01, f'{p:.1%}', ha='center', va='bottom', fontsize=9)
            ax.text(i + width/2, r + 0.01, f'{r:.1%}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        try:
            plt.savefig('scenario_performance_analysis.png', dpi=300, bbox_inches='tight')
            print("✅ Saved scenario_performance_analysis.png")
        except Exception as e:
            print(f"Could not save scenario plot: {e}")
        plt.close()

        print("📊 Scenario Analysis saved as: scenario_performance_analysis.png")

    def print_summary_report(self):
        """Print comprehensive summary of validation results."""
        if not self.report:
            return

        print("\n" + "="*80)
        print("AGRICULTURAL SAFETY SYSTEM - VALIDATION SUMMARY")
        print("="*80)

        # Basic stats
        print("\n📊 DATASET STATISTICS:")
        print(f"   Images Processed: {self.report.get('images_processed', 0)}")
        print(f"   Ground Truth Annotations: {self.report.get('ground_truth_annotations', 0)}")
        print(f"   System Detections: {self.report.get('system_detections', 0)}")

        # Overall metrics
        overall = self.report.get('overall_metrics', {})
        print("\n🎯 OVERALL PERFORMANCE:")
        print(f"   Precision: {overall.get('precision', 0):.1%}")
        print(f"   Recall: {overall.get('recall', 0):.1%}")
        print(f"   F1-Score: {overall.get('f1_score', 0):.1%}")

        # Confusion matrix
        cm = self.report.get('confusion_matrix', {})
        print("\n📈 CONFUSION MATRIX:")
        print(f"   True Positives: {cm.get('true_positives', 0)}")
        print(f"   False Positives: {cm.get('false_positives', 0)}")
        print(f"   False Negatives: {cm.get('false_negatives', 0)}")

        # Distance performance
        print("\n📍 DISTANCE-STRATIFIED PERFORMANCE:")
        distance_perf = self.report.get('distance_performance', {})
        for range_name, perf in distance_perf.items():
            print(f"   {range_name.title()}: P={perf.get('precision', 0):.1%}, R={perf.get('recall', 0):.1%}, F1={perf.get('f1_score', 0):.1%}")

        # KPI assessment
        print("\n🎯 KPI TARGET ASSESSMENT:")
        kpi_data = self.report.get('kpi_comparison', {})
        for key, data in kpi_data.items():
            target = data.get('target', 0)
            actual = data.get('actual', 0)
            met = data.get('met', False)
            gap = data.get('gap', 0)
            status = "✅ MET" if met else "❌ GAP"
            print(f"   {key.replace('_', ' ').title()}: {actual:.1%} vs {target:.1%} target ({status}, gap: {gap:.1%})")

        print("\n📊 VISUALIZATIONS GENERATED:")
        print("   - validation_kpi_analysis.png (KPI comparison charts)")
        print("   - distance_performance_analysis.png (Distance-stratified performance)")
        print("   - scenario_performance_analysis.png (Scenario-based analysis)")

        print("\n" + "="*80)

def main():
    """Main visualization function."""
    print("🔍 Generating Agricultural Safety System Performance Visualizations...")

    visualizer = ValidationVisualizer()

    if visualizer.report:
        # Generate all visualizations
        visualizer.create_kpi_comparison_plot()
        visualizer.create_distance_performance_plot()
        visualizer.create_scenario_analysis_plot()
        visualizer.print_summary_report()

        print("\n✅ All visualizations completed! Check the generated PNG files.")
    else:
        print("❌ No validation report found. Run real_world_validation.py first.")

if __name__ == "__main__":
    main()