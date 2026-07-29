from real_world_validation import RealWorldValidator
import logging

# Reduce logging to see results better
logging.getLogger().setLevel(logging.WARNING)

validator = RealWorldValidator()

# Step 1: Load ground truth
print("Loading ground truth...")
ground_truth = validator.load_ground_truth()
if not ground_truth:
    print("Failed to load ground truth")
    exit(1)

# Step 2: Run detection on dataset (smaller sample for speed)
print("Running detection on 20 images...")
detections = validator.run_detection_on_dataset(max_images=20)

# Step 3: Calculate metrics
print("Calculating metrics...")
metrics = validator.calculate_metrics()
if not metrics:
    print("Failed to calculate metrics")
    exit(1)

# Step 4: Generate validation report
print("Generating report...")
report = validator.generate_validation_report(metrics)

print("\n" + "="*60)
print("VALIDATION RESULTS")
print("="*60)
print(f"Images processed: {report['summary']['total_images_processed']}")
print(f"Ground truth annotations: {report['summary']['total_ground_truth_annotations']}")
print(f"System detections: {report['summary']['total_system_detections']}")
print(".1%")
print(".1%")
print(".1%")

print("\nDistance Performance:")
for band, metrics in report['distance_stratified_performance'].items():
    print(f"  {band.capitalize()}: P={metrics['precision']:.1%}, R={metrics['recall']:.1%}")

print("\nKPI Comparison:")
for target_name, comparison in report['kpi_comparison'].items():
    status = "✅ MET" if comparison['met'] else f"❌ GAP: {comparison['gap']:.1%}"
    print(f"  {target_name}: {comparison['actual']:.1%} vs {comparison['target']:.1%} {status}")

print("="*60)