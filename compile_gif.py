#!/usr/bin/env python3
"""
Agricultural Safety AI - GIF Compiler
Command-line tool to compile demo frames into GIF animations
"""

import os
import glob
import argparse
import tempfile
from pathlib import Path
from PIL import Image
import json
import webbrowser

def compile_frames_to_gif(frame_files, output_path, max_frames=100, duration=100, verbose=True):
    """Compile frame images into a GIF"""
    if not frame_files:
        print("❌ No frame files provided")
        return None

    selected_files = frame_files[:max_frames]
    images = []

    if verbose:
        print(f"📸 Processing {len(selected_files)} frames...")

    for i, frame_file in enumerate(selected_files):
        try:
            img = Image.open(frame_file).convert("RGB")
            images.append(img)
            if verbose and (i + 1) % 20 == 0:
                print(f"  ✓ Processed {i+1}/{len(selected_files)} frames")
        except Exception as e:
            print(f"⚠️  Could not load {frame_file}: {e}")
            continue

    if not images:
        print("❌ No valid images to compile")
        return None

    try:
        if verbose:
            print(f"🎬 Creating GIF with {len(images)} frames...")
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0,
            optimize=True,
        )
        gif_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"✅ GIF created: {output_path}")
        print(f"📊 GIF size: {gif_size:.1f} MB")
        return output_path
    except Exception as e:
        print(f"❌ Could not create GIF: {e}")
        return None

def find_latest_demo():
    """Find the most recent demo output directory"""
    temp_base = Path(tempfile.gettempdir())
    demo_dirs = []

    for item in temp_base.iterdir():
        if item.is_dir() and (item / "demo_frames").exists():
            demo_dirs.append(item)

    if not demo_dirs:
        return None

    # Sort by modification time, most recent first
    demo_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return demo_dirs[0]

def main():
    parser = argparse.ArgumentParser(description="Compile demo frames into GIF animation")
    parser.add_argument("--demo-dir", help="Path to demo output directory")
    parser.add_argument("--output", "-o", help="Output GIF file path")
    parser.add_argument("--max-frames", type=int, default=100, help="Maximum frames to include (default: 100)")
    parser.add_argument("--duration", type=int, default=100, help="Frame duration in ms (default: 100)")
    parser.add_argument("--open", action="store_true", help="Open GIF in default viewer after creation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Find demo directory
    if args.demo_dir:
        demo_dir = Path(args.demo_dir)
        if not demo_dir.exists():
            print(f"❌ Demo directory not found: {demo_dir}")
            return
    else:
        demo_dir = find_latest_demo()
        if not demo_dir:
            print("❌ No demo outputs found in temp directory")
            print("💡 Run a demo first: python run_demo.py --input-type video --input-path 0 --max-images 50")
            return

    print(f"📁 Using demo directory: {demo_dir}")

    # Check for frames
    frames_dir = demo_dir / "demo_frames"
    if not frames_dir.exists():
        print(f"❌ demo_frames directory not found in {demo_dir}")
        return

    frame_files = sorted(glob.glob(str(frames_dir / "frame_*.jpg")))
    if not frame_files:
        print(f"❌ No frame files found in {frames_dir}")
        return

    print(f"📸 Found {len(frame_files)} frame files")

    # Set output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = demo_dir / "compiled_demo.gif"

    # Compile GIF
    result = compile_frames_to_gif(frame_files, output_path, args.max_frames, args.duration, args.verbose)

    if result and args.open:
        print("🌐 Opening GIF in default viewer...")
        webbrowser.open(str(output_path))

    # Show stats if available
    stats_path = demo_dir / "demo_stats.json"
    if stats_path.exists():
        try:
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            print("\n📊 Demo Statistics:")
            print(f"  ⏱️  Total Time: {stats.get('total_time_s', 0):.1f}s")
            print(f"  🎯 Average FPS: {stats.get('average_fps', 0):.1f}")
            print(f"  👥 Total Detections: {stats.get('total_detections', 0)}")
            print(f"  📍 Total Tracks: {stats.get('total_tracks', 0)}")

            risk_dist = stats.get('average_risk_distribution', {})
            if risk_dist:
                print("  ⚠️  Risk Distribution:")
                for level, count in risk_dist.items():
                    print(f"    {level}: {count:.1f}")

        except Exception as e:
            print(f"⚠️  Could not load stats: {e}")

if __name__ == "__main__":
    main()