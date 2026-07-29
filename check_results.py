from real_world_validation import RealWorldValidator

validator = RealWorldValidator()

# Step 1: Load ground truth
ground_truth = validator.load_ground_truth()
if not ground_truth:
    print("Failed to load ground truth")
    exit(1)

# Step 2: Run detection on dataset
detections = validator.run_detection_on_dataset(max_images=50)  # Quick test

# Step 3: Calculate metrics
metrics = validator.calculate_metrics()
if not metrics:
    print("Failed to calculate metrics")
    exit(1)

# Step 4: Generate validation report
report = validator.generate_validation_report(metrics)

print('Validation completed')
print(f'Images processed: {report["summary"]["total_images_processed"]}')
print(f'Ground truth annotations: {report["summary"]["total_ground_truth_annotations"]}')
print(f'Total detections: {report["summary"]["total_system_detections"]}')
print(f'Overall precision: {report["summary"]["overall_precision"]:.3f}')
print(f'Overall recall: {report["summary"]["overall_recall"]:.3f}')