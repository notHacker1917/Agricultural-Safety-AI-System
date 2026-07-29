import streamlit as st
import os
import cv2
import glob
import json
import pandas as pd
from pathlib import Path
import tempfile
from PIL import Image

st.set_page_config(page_title="Agricultural Safety AI Demo", page_icon="🚜", layout="wide")

st.title("🚜 Agricultural Safety AI Demo Interface")
st.markdown("Display and analyze demo results from the autonomous harvester safety system.")


def compile_frames_to_gif(frame_files, output_path, max_frames=80, duration=120):
    """Compile a set of frame images into a GIF for quick playback."""
    if not frame_files:
        return None
    selected_files = frame_files[:max_frames]
    images = []
    for frame_file in selected_files:
        try:
            img = Image.open(frame_file).convert("RGB")
            images.append(img)
        except Exception:
            continue
    if not images:
        return None
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        optimize=True,
    )
    return output_path


def load_demo_stats(stats_path):
    try:
        with open(stats_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

# Find the most recent demo outputs in temp directory
temp_base = Path(tempfile.gettempdir())
demo_dirs = []
for item in temp_base.iterdir():
    if item.is_dir() and (item / "demo_frames").exists():
        demo_dirs.append(item)

if demo_dirs:
    # Sort by modification time, most recent first
    demo_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    demo_dir = demo_dirs[0]
    st.success(f"Found demo outputs from: {demo_dir}")
else:
    st.error("No demo outputs found. Run the demo first to generate results.")
    st.stop()
frames_dir = demo_dir / "demo_frames"
video_path = demo_dir / "demo_result.mp4"

# Check if outputs exist
if not demo_dir.exists():
    st.error("No demo outputs found. Run the demo first to generate results.")
    st.stop()

st.header("📊 Demo Results Overview")

col1, col2, col3 = st.columns(3)

with col1:
    if frames_dir.exists():
        frame_files = sorted(glob.glob(str(frames_dir / "frame_*.jpg")))
        st.metric("Frames Processed", len(frame_files))
    else:
        st.metric("Frames Processed", 0)

with col2:
    if video_path.exists():
        st.metric("Video Generated", "✅")
    else:
        st.metric("Video Generated", "❌")

with col3:
    st.metric("Detection System", "YOLO + DeepSORT")

st.header("️ Compiled Frame Playback")
compiled_gif_path = demo_dir / "compiled_demo.gif"
stats_path = demo_dir / "demo_stats.json"
if frames_dir.exists():
    frame_files = sorted(glob.glob(str(frames_dir / "frame_*.jpg")))
    if frame_files:
        if not compiled_gif_path.exists():
            st.info("Compiling frames into a GIF preview. This may take a few seconds...")
            try:
                compiled = compile_frames_to_gif(frame_files, compiled_gif_path)
                if compiled:
                    st.image(str(compiled), caption="Compiled demo frames as GIF", use_column_width=True)
                else:
                    st.warning("Failed to compile frames into GIF.")
            except Exception as e:
                st.error(f"Could not generate GIF: {e}")
        else:
            st.image(str(compiled_gif_path), caption="Compiled demo frames as GIF", use_column_width=True)
    else:
        st.warning("No frame images available to compile.")
else:
    st.warning("demo_frames directory not found, cannot compile frames.")

stats = load_demo_stats(stats_path)
if stats:
    st.subheader("📊 Visual Data")
    col1, col2, col3 = st.columns(3)
    col1.metric("Frames", stats.get('frame_count', 0))
    col2.metric("Average FPS", f"{stats.get('average_fps', 0):.2f}")
    col3.metric("Total Detections", stats.get('total_detections', 0))

    col4, col5, col6 = st.columns(3)
    col4.metric("Total Tracks", stats.get('total_tracks', 0))
    col5.metric("Run Time (s)", f"{stats.get('total_time_s', 0):.2f}")
    col6.metric("GIF Frames", len(frame_files) if frame_files else 0)

    df_trends = pd.DataFrame({
        'Detections': stats.get('per_frame_detections', []),
        'Tracks': stats.get('per_frame_tracks', []),
    })
    st.line_chart(df_trends)

    risk_dist = stats.get('average_risk_distribution', {})
    if risk_dist:
        df_risk = pd.DataFrame.from_dict(risk_dist, orient='index', columns=['Average'])
        st.bar_chart(df_risk)
else:
    st.info("No demo metrics file found; run the demo to capture visual analytics.")

st.header("🖼️ Frame Gallery")
if frames_dir.exists():
    frame_files = sorted(glob.glob(str(frames_dir / "frame_*.jpg")))
    if frame_files:
        # Create a slider to select frame
        frame_idx = st.slider("Select Frame", 0, len(frame_files)-1, 0)
        selected_frame = frame_files[frame_idx]
        
        # Display the selected frame
        st.image(selected_frame, caption=f"Frame {frame_idx:06d}", use_column_width=True)
        
        # Try to load frame-specific data if available
        frame_data_path = demo_dir / f"frame_{frame_idx:06d}_data.json"
        if frame_data_path.exists():
            try:
                with open(frame_data_path, 'r') as f:
                    frame_data = json.load(f)
                
                # Display LLM insights if available
                if frame_data.get('llm_enhanced', False):
                    st.subheader("🤖 LLM-Enhanced Analysis")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("LLM Risk Level", frame_data.get('new_risk_level', 'N/A'))
                        st.metric("LLM Confidence", f"{frame_data.get('llm_confidence', 0):.2f}")
                    
                    with col2:
                        scenarios = frame_data.get('predicted_scenarios', [])
                        if scenarios:
                            st.write("**Predicted Scenarios:**")
                            for scenario in scenarios[:3]:  # Show top 3
                                st.write(f"• {scenario}")
                    
                    reasoning = frame_data.get('llm_reasoning', '')
                    if reasoning:
                        st.write("**LLM Reasoning:**")
                        st.info(reasoning[:500] + "..." if len(reasoning) > 500 else reasoning)
                    
                    actions = frame_data.get('recommended_actions', [])
                    if actions:
                        st.write("**Recommended Actions:**")
                        for action in actions[:3]:  # Show top 3
                            st.write(f"• {action}")
                else:
                    st.info("Traditional computer vision analysis (no LLM enhancement)")
                    
            except Exception as e:
                st.warning(f"Could not load frame data: {e}")
        
        # Show frame info
        st.write(f"**Frame Path:** {selected_frame}")
        
        # Optional: Show multiple frames in a grid
        st.subheader("Frame Grid View")
        cols = st.columns(5)
        for i, frame_file in enumerate(frame_files[:20]):  # Show first 20 frames
            with cols[i % 5]:
                st.image(frame_file, caption=f"Frame {i:06d}", width=150)
    else:
        st.warning("No frame images found in demo_frames directory.")
else:
    st.warning("demo_frames directory not found.")

st.header("📈 System Capabilities")
st.markdown("""
### Core Features
- **Human Detection**: YOLOv8 with agricultural optimizations
- **Object Tracking**: DeepSORT for persistent tracking
- **Risk Assessment**: 5-tier safety zones (CRITICAL, HIGH_WARNING, WARNING, LOW_WARNING, SAFE)
- **Edge Case Handling**: Occlusion, vibration, perspective corrections
- **Movement Prediction**: Trajectory forecasting with obstruction awareness

### Technical Specifications
- **Detection Range**: Up to 150m with enhanced far-distance logic
- **Processing Speed**: ~5 FPS on CPU
- **Safety Zones**: Configurable distance-based risk assessment
- **Fallback Detection**: HOG, contour, and skin-color methods
""")

st.header("🔧 How to Use")
st.markdown("""
1. **Run Demo**: Execute `python run_demo.py --input-type video --input-path 0 --max-images 50`
2. **View Results**: Refresh this interface to see new outputs
3. **Analyze Frames**: Use the slider to inspect individual frames
4. **Inspect Visual Data**: Use the charts and GIF preview below to track detections and risk distribution

### Output Files
- `demo_outputs/demo_frames/frame_XXXXXX.jpg`: Individual processed frames
- `demo_outputs/demo_stats.json`: Analytics metadata for the demo
""")

if st.button("🔄 Refresh Interface"):
    st.rerun()

st.markdown("---")
st.markdown("*Agricultural Safety AI - Hackathon Project*")