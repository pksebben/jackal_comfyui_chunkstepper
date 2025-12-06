#!/usr/bin/env python3
"""Standalone CLI script to concatenate video chunks into a final video.

Usage:
    python stitch.py <chunks_dir> [audio_path] [output_path]

Examples:
    python stitch.py ./chunks
    python stitch.py ./chunks ./audio.mp3
    python stitch.py ./chunks ./audio.wav ./final_output.mp4
"""

import argparse
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

VIDEO_EXTENSIONS = (".mp4", ".webm", ".mov", ".avi", ".mkv")
ANIMATED_EXTENSIONS = (".gif", ".webp")
# Extensions that can be stitched into a video (videos + animated images)
STITCHABLE_EXTENSIONS = VIDEO_EXTENSIONS + ANIMATED_EXTENSIONS
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")


def extract_counter(filename: str, chunk_num: int) -> int | None:
    """Extract the ComfyUI counter number from a filename.

    ComfyUI appends counter suffixes like _00001 or _00001_ to filenames.

    Args:
        filename: Filename stem (without extension)
        chunk_num: Expected chunk number prefix (e.g., 0 for "0000")

    Returns:
        Counter number, or None if no counter pattern found
    """
    base_name = f"{chunk_num:04d}"
    # Pattern: {base_name}_{counter} or {base_name}_{counter}_ or with suffix
    # e.g., 0000_00001.mp4, 0000_suffix_00001_.mp4
    pattern = rf"^{re.escape(base_name)}(?:_[^_]+)?_(\d+)_?$"
    match = re.match(pattern, filename)
    if match:
        return int(match.group(1))
    return None


def find_chunks(chunks_dir: Path) -> dict[int, list[tuple[Path, int | None]]]:
    """Find all video chunk files in the directory, grouped by chunk number.

    Only finds video files and animated formats (gif, webp) that can be stitched.
    Static images like .png are ignored.

    Args:
        chunks_dir: Directory containing chunk files

    Returns:
        Dictionary mapping chunk number to list of (file_path, counter) tuples.
        Counter is None for files without ComfyUI counter suffix.
    """
    chunks: dict[int, list[tuple[Path, int | None]]] = defaultdict(list)

    # Pattern to match chunk files: 4-digit number at start, optionally followed
    # by suffix and/or ComfyUI counter
    # Only matches video and animated image extensions (not static images)
    ext_pattern = "|".join(ext.lstrip(".") for ext in STITCHABLE_EXTENSIONS)
    pattern = re.compile(rf"^(\d{{4}}).*\.({ext_pattern})$", re.IGNORECASE)

    for file in chunks_dir.iterdir():
        if not file.is_file():
            continue
        match = pattern.match(file.name)
        if match:
            chunk_num = int(match.group(1))
            counter = extract_counter(file.stem, chunk_num)
            chunks[chunk_num].append((file, counter))

    return dict(chunks)


def display_thumbnail(video_path: Path, chunk_num: int, index: int) -> None:
    """Display a thumbnail from the video file.

    Args:
        video_path: Path to the video file
        chunk_num: Chunk number for display
        index: Index in the list of candidates
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [{index}] {video_path.name} (could not open for preview)")
        return

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0

        # Read first frame for basic info
        ret, frame = cap.read()
        if ret and frame is not None:
            print(
                f"  [{index}] {video_path.name} "
                f"({width}x{height}, {frame_count} frames, {duration:.1f}s)"
            )
        else:
            print(f"  [{index}] {video_path.name} (could not read frame)")
    finally:
        cap.release()


def select_chunk_file(
    chunk_num: int,
    candidates: list[tuple[Path, int | None]],
    variation_preference: int = -1,
) -> Path:
    """Select which file to use when multiple candidates exist.

    Auto-selects based on variation_preference, or prompts user if -2 (interactive).

    Args:
        chunk_num: The chunk number
        candidates: List of (file_path, counter) tuples
        variation_preference: -1 = lowest counter (default), -2 = interactive,
                              0+ = prefer specific counter

    Returns:
        Selected file path
    """
    if len(candidates) == 1:
        return candidates[0][0]

    # Sort by counter (None treated as -1 to come first, then by counter value)
    sorted_candidates = sorted(
        candidates,
        key=lambda x: (x[1] is not None, x[1] if x[1] is not None else 0),
    )

    # Auto-select based on preference
    if variation_preference == -1:
        # Lowest counter (first after sorting)
        selected = sorted_candidates[0][0]
        print(f"  Chunk {chunk_num:04d}: auto-selected {selected.name} (lowest)")
        return selected
    elif variation_preference >= 0:
        # Look for specific counter
        for path, counter in sorted_candidates:
            if counter == variation_preference:
                print(
                    f"  Chunk {chunk_num:04d}: auto-selected {path.name} "
                    f"(counter {variation_preference})"
                )
                return path
        # Fall back to lowest if preferred not found
        selected = sorted_candidates[0][0]
        print(
            f"  Chunk {chunk_num:04d}: counter {variation_preference} not found, "
            f"using {selected.name}"
        )
        return selected

    # Interactive mode (variation_preference == -2)
    print(f"\nMultiple files found for chunk {chunk_num:04d}:")
    print("Showing video info for each candidate:\n")

    # For interactive display, sort by modification time (newest first)
    display_candidates = sorted(
        candidates, key=lambda x: x[0].stat().st_mtime, reverse=True
    )

    for i, (path, counter) in enumerate(display_candidates):
        display_thumbnail(path, chunk_num, i)
        if counter is not None:
            print(f"       Counter: {counter}")

    while True:
        try:
            choice = input(
                f"\nSelect file for chunk {chunk_num:04d} "
                f"[0-{len(display_candidates) - 1}]: "
            )
            idx = int(choice.strip())
            if 0 <= idx < len(display_candidates):
                return display_candidates[idx][0]
            print(f"Please enter a number between 0 and {len(display_candidates) - 1}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nAborted by user")
            sys.exit(1)


def convert_animated_to_video(
    input_path: Path, output_path: Path, fps: int = 8
) -> bool:
    """Convert an animated WebP/GIF to a video file using Pillow.

    Args:
        input_path: Path to animated image
        output_path: Path for output video
        fps: Frames per second for output video

    Returns:
        True if conversion succeeded, False otherwise
    """
    try:
        img = Image.open(input_path)
    except Exception:
        return False

    # Extract all frames
    frames: list[np.ndarray] = []
    try:
        while True:
            # Convert to RGB and then to BGR for OpenCV
            frame_rgb = img.convert("RGB")
            frame_array = np.array(frame_rgb)
            frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            frames.append(frame_bgr)
            img.seek(img.tell() + 1)
    except EOFError:
        pass  # End of frames

    if not frames:
        return False

    # Get dimensions from first frame
    height, width = frames[0].shape[:2]

    # Write video using OpenCV
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for frame in frames:
        writer.write(frame)

    writer.release()
    return True


def prepare_chunks_for_concat(
    chunks: list[Path], temp_dir: Path
) -> tuple[list[Path], bool]:
    """Prepare chunks for concatenation, converting animated formats if needed.

    Args:
        chunks: List of chunk files
        temp_dir: Temporary directory for converted files

    Returns:
        Tuple of (prepared chunk paths, whether any conversion was done)
    """
    prepared: list[Path] = []
    any_converted = False

    for i, chunk in enumerate(chunks):
        if chunk.suffix.lower() in ANIMATED_EXTENSIONS:
            # Convert animated format to video
            converted_path = temp_dir / f"chunk_{i:04d}.mp4"
            print(f"  Converting {chunk.name} to video...")
            if convert_animated_to_video(chunk, converted_path):
                prepared.append(converted_path)
                any_converted = True
            else:
                print(f"  Warning: Failed to convert {chunk.name}, using original")
                prepared.append(chunk)
        else:
            prepared.append(chunk)

    return prepared, any_converted


def create_concat_file(chunks: list[Path], temp_dir: Path) -> Path:
    """Create FFmpeg concat demuxer file.

    Args:
        chunks: List of chunk files in order
        temp_dir: Temporary directory for the concat file

    Returns:
        Path to the concat file
    """
    concat_file = temp_dir / "concat.txt"
    with open(concat_file, "w") as f:
        for chunk in chunks:
            # Escape single quotes in path for FFmpeg
            escaped_path = str(chunk.absolute()).replace("'", "'\\''")
            f.write(f"file '{escaped_path}'\n")
    return concat_file


def stitch_videos(
    chunks: list[Path],
    output_path: Path,
    audio_path: Path | None = None,
) -> None:
    """Concatenate video chunks and optionally add audio using FFmpeg.

    Args:
        chunks: List of chunk files in order
        output_path: Output file path
        audio_path: Path to audio file (optional)
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Convert animated formats (webp, gif) to video first
        has_animated = any(c.suffix.lower() in ANIMATED_EXTENSIONS for c in chunks)
        if has_animated:
            print("\nConverting animated chunks to video format...")
            prepared_chunks, _ = prepare_chunks_for_concat(chunks, temp_path)
        else:
            prepared_chunks = chunks

        concat_file = create_concat_file(prepared_chunks, temp_path)

        # FFmpeg command for concatenation with H.264 encoding
        # -safe 0: Allow any file path in concat file
        # -c:v libx264 -crf 18: High quality H.264 encoding
        if audio_path is not None:
            # With audio: -c:a aac, -shortest to end when shortest input ends
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output without asking
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-i",
                str(audio_path),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",  # Required for QuickTime/broad player compatibility
                "-crf",
                "18",  # High quality (0=lossless is often problematic)
                "-preset",
                "medium",
                "-vsync",
                "cfr",  # Constant frame rate to fix timing issues
                "-movflags",
                "+faststart",  # Move moov atom to start for QuickTime
                "-c:a",
                "aac",
                "-b:a",
                "320k",
                "-shortest",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                str(output_path),
            ]
            print(f"\nStitching {len(chunks)} chunks with audio...")
        else:
            # Without audio: video only
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output without asking
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",  # Required for QuickTime/broad player compatibility
                "-crf",
                "18",  # High quality (0=lossless is often problematic)
                "-preset",
                "medium",
                "-vsync",
                "cfr",  # Constant frame rate to fix timing issues
                "-movflags",
                "+faststart",  # Move moov atom to start for QuickTime
                "-an",  # No audio
                str(output_path),
            ]
            print(f"\nStitching {len(chunks)} chunks (no audio)...")

        print(f"Output: {output_path}")

        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            print("Stitching complete!")
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e.stderr}", file=sys.stderr)
            sys.exit(1)
        except FileNotFoundError:
            print(
                "Error: FFmpeg not found. Please install FFmpeg and ensure it's in your PATH.",
                file=sys.stderr,
            )
            sys.exit(1)


def main() -> None:
    """Main entry point for the stitch script."""
    parser = argparse.ArgumentParser(
        description="Concatenate video chunks into a final video, optionally with audio.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python stitch.py ./chunks
    python stitch.py ./chunks ./audio.mp3
    python stitch.py ./chunks ./audio.wav ./final_output.mp4
    python stitch.py ./chunks -v -2  # interactive mode for duplicates

The script will:
  1. Scan chunks_dir for files matching pattern {chunk_number:04d}*.*
  2. Process chunks in ascending numerical order (0000, 0001, 0002, ...)
  3. Stop when a chunk number is missing
  4. Auto-select files when duplicates exist (based on --variation setting)
  5. Output a high-quality MP4, with audio if provided
        """,
    )
    parser.add_argument(
        "chunks_dir",
        type=Path,
        help="Directory containing numbered chunk files",
    )
    parser.add_argument(
        "audio_path",
        type=Path,
        nargs="?",
        default=None,
        help="Path to audio file to mux into final video (optional)",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        nargs="?",
        default=Path("output.mp4"),
        help="Output file path (default: output.mp4)",
    )
    parser.add_argument(
        "-v",
        "--variation",
        type=int,
        default=-1,
        help=(
            "ComfyUI counter preference when duplicates exist. "
            "-1 = lowest counter (default), -2 = interactive picker, "
            "0+ = prefer specific counter number"
        ),
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.chunks_dir.exists():
        print(
            f"Error: Chunks directory does not exist: {args.chunks_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.chunks_dir.is_dir():
        print(f"Error: Not a directory: {args.chunks_dir}", file=sys.stderr)
        sys.exit(1)

    if args.audio_path is not None and not args.audio_path.exists():
        print(f"Error: Audio file does not exist: {args.audio_path}", file=sys.stderr)
        sys.exit(1)

    # Find all chunks
    chunk_map = find_chunks(args.chunks_dir)

    if not chunk_map:
        print(f"Error: No chunk files found in: {args.chunks_dir}", file=sys.stderr)
        print(
            "Expected files matching pattern: 0000*.{mp4,webm,mov,...}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Check for chunk 0000
    if 0 not in chunk_map:
        print("Error: Chunk 0000 is missing", file=sys.stderr)
        sys.exit(1)

    # Build ordered list of chunks, stopping at gaps
    ordered_chunks: list[Path] = []
    chunk_num = 0

    while chunk_num in chunk_map:
        candidates = chunk_map[chunk_num]
        selected = select_chunk_file(chunk_num, candidates, args.variation)
        ordered_chunks.append(selected)
        chunk_num += 1

    # Check if we stopped early due to a gap
    max_chunk = max(chunk_map.keys())
    if chunk_num <= max_chunk:
        print(
            f"\nWarning: Gap detected at chunk {chunk_num:04d}. "
            f"Stopping at chunk {chunk_num - 1:04d}.",
            file=sys.stderr,
        )
        print(
            f"(Higher chunks exist up to {max_chunk:04d} but will be skipped)",
            file=sys.stderr,
        )

    # Validate all chunks are the same file type
    extensions = {chunk.suffix.lower() for chunk in ordered_chunks}
    if len(extensions) > 1:
        print(
            f"Error: Mixed file types found: {', '.join(sorted(extensions))}",
            file=sys.stderr,
        )
        print(
            "All chunks must be the same file type (e.g., all .mp4 or all .webm)",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nFound {len(ordered_chunks)} sequential chunks:")
    for i, chunk in enumerate(ordered_chunks):
        print(f"  {i:04d}: {chunk.name}")

    # Stitch videos
    stitch_videos(ordered_chunks, args.output_path, args.audio_path)


if __name__ == "__main__":
    main()
