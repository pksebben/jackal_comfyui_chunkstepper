"""VideoChunkStepper node for ComfyUI.

Manages iterative video chunk generation workflow for AnimateDiff and similar tools.
"""

from pathlib import Path

import cv2
import torch

# Common video extensions to check for previous chunks
VIDEO_EXTENSIONS = (".mp4", ".webm", ".mov", ".avi", ".mkv", ".gif")


class VideoChunkStepper:
    """ComfyUI node for iterative video chunk generation.

    When generating long videos with AnimateDiff, you typically render in chunks
    (e.g., 16-32 frames at a time), using the last frame of chunk N-1 as the
    init frame for chunk N. This node manages that workflow.
    """

    @classmethod
    def INPUT_TYPES(s):  # noqa: N802, N804
        return {
            "required": {
                "chunks_directory": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "/path/to/chunks",
                    },
                ),
                "chunk_number": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 9999,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "init_image": ("IMAGE",),
            },
            "optional": {
                "suffix": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "_draft",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("output_path", "frame")
    FUNCTION = "execute"
    CATEGORY = "video/chunk"
    DESCRIPTION = (
        "Manages iterative video chunk generation. Outputs the path for the "
        "current chunk and the appropriate starting frame (init_image for chunk 0, "
        "or last frame of previous chunk for chunk N)."
    )

    def execute(
        self,
        chunks_directory: str,
        chunk_number: int,
        init_image: torch.Tensor,
        suffix: str = "",
    ) -> tuple[str, torch.Tensor]:
        """Execute the node logic.

        Args:
            chunks_directory: Base directory where chunks are saved
            chunk_number: Current chunk index (0-indexed)
            init_image: Starting frame for chunk 0
            suffix: Optional string appended to chunk filename

        Returns:
            Tuple of (output_path, frame)
        """
        # Ensure chunks directory exists
        chunks_path = Path(chunks_directory)
        if not chunks_path.exists():
            chunks_path.mkdir(parents=True, exist_ok=True)

        # Build output path (without extension)
        output_path = str(chunks_path / f"{chunk_number:04d}{suffix}")

        # Determine which frame to output
        if chunk_number == 0:
            # For chunk 0, use the provided init_image directly
            frame = init_image
        else:
            # For chunk N > 0, load the last frame from chunk N-1
            frame = self._load_last_frame_from_previous_chunk(
                chunks_path, chunk_number, suffix
            )

        return (output_path, frame)

    def _load_last_frame_from_previous_chunk(
        self,
        chunks_path: Path,
        chunk_number: int,
        suffix: str,
    ) -> torch.Tensor:
        """Load the last frame from the previous chunk video.

        Args:
            chunks_path: Base directory where chunks are saved
            chunk_number: Current chunk index
            suffix: Suffix used in filenames

        Returns:
            Last frame as IMAGE tensor [1, H, W, C]

        Raises:
            FileNotFoundError: If previous chunk file not found
            ValueError: If video can't be read or has no frames
        """
        prev_chunk_num = chunk_number - 1
        prev_chunk_base = f"{prev_chunk_num:04d}{suffix}"

        # Find previous chunk file (check all video extensions)
        prev_chunk_file = self._find_video_file(chunks_path, prev_chunk_base)

        if prev_chunk_file is None:
            expected_path = chunks_path / f"{prev_chunk_base}.<video_ext>"
            raise FileNotFoundError(
                f"Previous chunk not found. Expected file matching: {expected_path}\n"
                f"Checked extensions: {VIDEO_EXTENSIONS}"
            )

        # Load the video and extract last frame
        return self._extract_last_frame(prev_chunk_file)

    def _find_video_file(self, directory: Path, base_name: str) -> Path | None:
        """Find a video file matching the base name with any video extension.

        Also handles ComfyUI's counter suffix (e.g., 0001_00001.mp4).

        Args:
            directory: Directory to search in
            base_name: Base filename without extension

        Returns:
            Path to found video file, or None if not found
        """
        # First, try exact match with each extension
        for ext in VIDEO_EXTENSIONS:
            candidate = directory / f"{base_name}{ext}"
            if candidate.exists():
                return candidate

        # Check for files with ComfyUI counter suffix (e.g., 0001_00001.mp4)
        # We'll take the most recently modified one if multiple exist
        matching_files: list[Path] = []
        for ext in VIDEO_EXTENSIONS:
            pattern = f"{base_name}_*{ext}"
            matching_files.extend(directory.glob(pattern))

        if matching_files:
            # Sort by modification time (newest first) and return the newest
            matching_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return matching_files[0]

        return None

    def _extract_last_frame(self, video_path: Path) -> torch.Tensor:
        """Extract the last frame from a video file.

        Args:
            video_path: Path to the video file

        Returns:
            Last frame as IMAGE tensor [1, H, W, C] with values in [0, 1]

        Raises:
            ValueError: If video can't be read or has no frames
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        try:
            # Get total frame count
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                raise ValueError(f"Video has no frames: {video_path}")

            # Seek to last frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)

            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"Could not read last frame from: {video_path}")

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to tensor with shape [1, H, W, C] and normalize to [0, 1]
            frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
            frame_tensor = frame_tensor.unsqueeze(0)  # Add batch dimension

            return frame_tensor

        finally:
            cap.release()

    @classmethod
    def IS_CHANGED(  # noqa: N802
        s,  # noqa: N804
        chunks_directory: str,
        chunk_number: int,
        init_image: torch.Tensor,
        suffix: str = "",
    ) -> float:
        """Always re-execute when chunk_number > 0 since previous chunk may have changed."""
        if chunk_number > 0:
            # Return NaN to always re-execute (previous chunk file may have changed)
            return float("NaN")
        # For chunk 0, use default caching behavior
        return 0.0


# Node mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "VideoChunkStepper": VideoChunkStepper,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoChunkStepper": "Video Chunk Stepper",
}
