#!/usr/bin/env python3
"""
Agricultural Safety AI - GIF Display Interface
Displays compiled GIF animations from demo runs
"""

import streamlit as st
import os
import glob
import tempfile
from pathlib import Path
from PIL import Image
import json
import pandas as pd

st.set_page_config(
    page_title="🚜 Agri Safety GIF Viewer",
    page_icon="🚜",
    layout="wide"
)

st.title("🚜 Agricultural Safety AI - GIF Display Interface")
st.markdown("View compiled GIF animations from safety system demos")

def find_demo_outputs():
    """Find demo output directories in temp folder"""
    temp_base = Path(tempfile.gettempdir())
    demo_dirs = []

    for item in temp_base.iterdir():
        if item.is_dir() and (item / "demo_frames").exists():
            demo_dirs.append(item)

    # Sort by modification time, most recent first
    demo_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return demo_dirs

def compile_frames_to_gif(frame_files, output_path, max_frames=100, duration=100):
    """Compile frame images into a GIF"""
    if not frame_files:
        return None

    selected_files = frame_files[:max_frames]
    images = []

    for frame_file in selected_files:
        try:
            img = Image.open(frame_file).convert("RGB")
            images.append(img)
        except Exception as e:
            st.warning(f"Could not load {frame_file}: {e}")
            continue

    if not images:
        return None

    try:
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0,
            optimize=True,
        )
        return output_path
    except Exception as e:
        st.error(f"Could not create GIF: {e}")
        return None

def load_demo_stats(stats_path):
    """Load demo statistics from JSON file"""
    try:
        with open(stats_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Could not load stats: {e}")
        return None

# Main interface
demo_dirs = find_demo_outputs()

if not demo_dirs:
    st.error("❌ No demo outputs found!")
    st.info("Run a demo first using: `python run_demo.py --input-type video --input-path 0 --max-images 50`")
    st.stop()

# Demo selection
st.sidebar.header("🎬 Demo Selection")
selected_demo = st.sidebar.selectbox(
    "Choose demo run:",
    options=[f"Demo {i+1} - {d.name}" for i, d in enumerate(demo_dirs)],
    index=0
)

demo_idx = [f"Demo {i+1} - {d.name}" for i, d in enumerate(demo_dirs)].index(selected_demo)
demo_dir = demo_dirs[demo_idx]

st.success(f"📁 Loaded demo from: `{demo_dir.name}`")

# Check for required files
frames_dir = demo_dir / "demo_frames"
gif_path = demo_dir / "compiled_demo.gif"
stats_path = demo_dir / "demo_stats.json"

col1, col2, col3 = st.columns(3)

with col1:
    frame_count = len(list(frames_dir.glob("frame_*.jpg"))) if frames_dir.exists() else 0
    st.metric("📸 Frames Available", frame_count)

with col2:
    gif_exists = gif_path.exists()
    st.metric("🎬 GIF Status", "✅ Ready" if gif_exists else "⏳ Needs compilation")

with col3:
    stats_exists = stats_path.exists()
    st.metric("📊 Stats Available", "✅ Yes" if stats_exists else "❌ No")

# GIF Display Section
st.header("🎬 Compiled GIF Animation")

if frames_dir.exists():
    frame_files = sorted(glob.glob(str(frames_dir / "frame_*.jpg")))

    if frame_files:
        # GIF compilation controls
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            max_frames = st.slider("Max frames in GIF", 10, min(200, len(frame_files)), min(100, len(frame_files)))

        with col2:
            duration = st.slider("Frame duration (ms)", 50, 500, 100)

        with col3:
            force_recompile = st.button("🔄 Recompile GIF")

        # Compile or load existing GIF
        if not gif_path.exists() or force_recompile:
            with st.spinner("🎬 Compiling frames into GIF..."):
                compiled_gif = compile_frames_to_gif(frame_files, gif_path, max_frames, duration)
                if compiled_gif:
                    st.success("✅ GIF compiled successfully!")
                else:
                    st.error("❌ Failed to compile GIF")
                    st.stop()

        # Display the GIF
        if gif_path.exists():
            st.image(str(gif_path), caption="🚜 Agricultural Safety Demo - Compiled Animation", use_column_width=True)

            # GIF info
            gif_size = gif_path.stat().st_size / (1024 * 1024)  # MB
            st.info(f"📊 GIF Details: {max_frames} frames, {duration}ms/frame, {gif_size:.1f} MB")

        else:
            st.warning("⚠️ GIF compilation failed")

    else:
        st.warning("⚠️ No frame images found in demo_frames directory")
else:
    st.error("❌ demo_frames directory not found")

# Statistics Section
if stats_exists:
    st.header("📊 Demo Statistics")

    stats = load_demo_stats(stats_path)
    if stats:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("⏱️ Total Time", f"{stats.get('total_time_s', 0):.1f}s")

        with col2:
            st.metric("🎯 Average FPS", f"{stats.get('average_fps', 0):.1f}")

        with col3:
            st.metric("👥 Total Detections", stats.get('total_detections', 0))

        with col4:
            st.metric("📍 Total Tracks", stats.get('total_tracks', 0))

        # Risk distribution
        risk_dist = stats.get('average_risk_distribution', {})
        if risk_dist:
            st.subheader("⚠️ Risk Level Distribution")
            df_risk = pd.DataFrame.from_dict(risk_dist, orient='index', columns=['Average Count'])
            st.bar_chart(df_risk)

        # Detection trends
        detections = stats.get('per_frame_detections', [])
        if detections:
            st.subheader("📈 Detection Trends Over Time")
            df_trends = pd.DataFrame({
                'Frame': range(len(detections)),
                'Detections': detections
            })
            st.line_chart(df_trends.set_index('Frame'))

# Frame Browser
if frames_dir.exists() and frame_files:
    st.header("🖼️ Frame Browser")

    frame_idx = st.slider("Select Frame", 0, len(frame_files)-1, 0)
    selected_frame = frame_files[frame_idx]

    col1, col2 = st.columns([3, 1])

    with col1:
        st.image(selected_frame, caption=f"Frame {frame_idx:06d}", use_column_width=True)

    with col2:
        st.markdown("**Frame Info:**")
        st.write(f"📁 File: {Path(selected_frame).name}")
        st.write(f"📊 Index: {frame_idx}")
        st.write(f"📈 Progress: {frame_idx+1}/{len(frame_files)} ({(frame_idx+1)/len(frame_files)*100:.1f}%)")

# Footer
st.markdown("---")
st.markdown("*Built for HackHPI 2026 - Agricultural Safety AI Challenge*")
st.markdown("**System Status:** 🟢 Optimized YOLOv8 + DeepSORT + LLM Risk Assessment")