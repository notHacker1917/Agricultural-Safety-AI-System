@echo off
del main.py
del coco_inference.py
del coco_loader.py
del dashboard.py
del detection.py
del evaluate_agri_models.py
del evaluate_agri_safety.py
del evaluate_coco.py
del evaluate_safety_system.py
del failure_case_analysis.py
del generate_sample_video.py
del homography.py
del preprocessing.py
del remove_emoji.py
del run_agri_demo.py
del safety_engine.py
del safety_engine_new.py
del segmentation_tracking.py
del setup_agri_dataset.py
del stabilization.py
del trajectory_storage.py
del video_input.py
del visualization.py
del generate_challenge_kpis.py
for %%f in (test_*.py) do del %%f
for %%f in (train_*.py) do del %%f
rd /s /q data
rd /s /q result
rd /s /q templates
echo Cleanup complete