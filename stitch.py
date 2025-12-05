#!/usr/bin/env python3
"""Standalone CLI script to concatenate video chunks into a final video with audio.

Usage:
    python stitch.py <chunks_dir> <audio_path> [output_path]

Examples:
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

VIDEO_EXTENSIONS = (".mp4", ".webm", ".mov", ".avi", ".mkv", ".gif")


def find_chunks(chunks_dir: Path) -> dict[int, list[Path]]:
    """Find all chunk files in the directory, grouped by chunk number.

    Args:
        chunks_dir: Directory containing chunk files

    Returns:
        Dictionary mapping chunk number to list of matching files
    """
    chunks: dict[int, list[Path]] = defaultdict(list)

    # Pattern to match chunk files: 4-digit number at start, optionally followed
    # by suffix and/or ComfyUI counter
    pattern = re.compile(r"^(\d{4}).*\.(mp4|webm|mov|avi|mkv|gif)$", re.IGNORECASE)

    for file in chunks_dir.iterdir():
        if not file.is_file():
            continue
        match = pattern.match(file.name)
        if match:
            chunk_num = int(match.group(1))
            chunks[chunk_num].append(file)

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
    candidates: list[Path],
) -> Path:
    """Let user select which file to use when multiple candidates exist.

    Args:
        chunk_num: The chunk number
        candidates: List of candidate files

    Returns:
        Selected file path
    """
    if len(candidates) == 1:
        return candidates[0]

    print(f"\nMultiple files found for chunk {chunk_num:04d}:")
    print("Showing video info for each candidate:\n")

    # Sort by modification time (newest first)
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)

    for i, candidate in enumerate(candidates):
        display_thumbnail(candidate, chunk_num, i)

    while True:
        try:
            choice = input(
                f"\nSelect file for chunk {chunk_num:04d} [0-{len(candidates) - 1}]: "
            )
            idx = int(choice.strip())
            if 0 <= idx < len(candidates):
                return candidates[idx]
            print(f"Please enter a number between 0 and {len(candidates) - 1}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nAborted by user")
            sys.exit(1)


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
    audio_path: Path,
    output_path: Path,
) -> None:
    """Concatenate video chunks and add audio using FFmpeg.

    Args:
        chunks: List of chunk files in order
        audio_path: Path to audio file
        output_path: Output file path
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        concat_file = create_concat_file(chunks, temp_path)

        # FFmpeg command for lossless concatenation with audio
        # -safe 0: Allow any file path in concat file
        # -c:v libx264 -crf 0: Lossless H.264 encoding
        # -c:a aac: AAC audio encoding
        # -shortest: End output when shortest input ends
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
            "-crf",
            "0",
            "-preset",
            "ultrafast",
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
        description="Concatenate video chunks into a final video with audio.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python stitch.py ./chunks ./audio.mp3
    python stitch.py ./chunks ./audio.wav ./final_output.mp4

The script will:
  1. Scan chunks_dir for files matching pattern {chunk_number:04d}*.*
  2. Process chunks in ascending numerical order (0000, 0001, 0002, ...)
  3. Stop when a chunk number is missing
  4. Prompt for selection when multiple files match the same chunk number
  5. Output an uncompressed MP4 with the provided audio
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
        help="Path to audio file to mux into final video",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        nargs="?",
        default=Path("output.mp4"),
        help="Output file path (default: output.mp4)",
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

    if not args.audio_path.exists():
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
        selected = select_chunk_file(chunk_num, candidates)
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

    print(f"\nFound {len(ordered_chunks)} sequential chunks:")
    for i, chunk in enumerate(ordered_chunks):
        print(f"  {i:04d}: {chunk.name}")

    # Stitch videos
    stitch_videos(ordered_chunks, args.audio_path, args.output_path)


if __name__ == "__main__":
    main()
