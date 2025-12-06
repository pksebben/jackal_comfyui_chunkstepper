"""VideoChunkStepper node for ComfyUI.

Manages iterative video chunk generation workflow for AnimateDiff and similar tools.
"""

import re
from pathlib import Path

import cv2
import folder_paths
import torch

# Common video extensions to check for previous chunks
VIDEO_EXTENSIONS = (".mp4", ".webm", ".mov", ".avi", ".mkv", ".gif")

# Image extensions that ComfyUI may save frames as
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")

# Combined extensions for file search
ALL_EXTENSIONS = VIDEO_EXTENSIONS + IMAGE_EXTENSIONS


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
                "variation_preference": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 99999,
                        "step": 1,
                        "display": "number",
                        "tooltip": (
                            "ComfyUI appends counter numbers to filenames. "
                            "-1 = use lowest number (default), "
                            "0+ = prefer specific variation number"
                        ),
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
        variation_preference: int = -1,
    ) -> tuple[str, torch.Tensor]:
        """Execute the node logic.

        Args:
            chunks_directory: Base directory where chunks are saved
            chunk_number: Current chunk index (0-indexed)
            init_image: Starting frame for chunk 0
            suffix: Optional string appended to chunk filename
            variation_preference: ComfyUI counter preference (-1 = lowest, 0+ = specific)

        Returns:
            Tuple of (output_path, frame)
        """
        # Resolve chunks directory path
        # If relative, resolve against ComfyUI's output directory
        chunks_path = Path(chunks_directory)
        if not chunks_path.is_absolute():
            output_dir = Path(folder_paths.get_output_directory())
            chunks_path = output_dir / chunks_path

        # Ensure chunks directory exists
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
                chunks_path, chunk_number, suffix, variation_preference
            )

        return (output_path, frame)

    def _load_last_frame_from_previous_chunk(
        self,
        chunks_path: Path,
        chunk_number: int,
        suffix: str,
        variation_preference: int = -1,
    ) -> torch.Tensor:
        """Load the last frame from the previous chunk video/image.

        Args:
            chunks_path: Base directory where chunks are saved
            chunk_number: Current chunk index
            suffix: Suffix used in filenames
            variation_preference: ComfyUI counter preference (-1 = lowest, 0+ = specific)

        Returns:
            Last frame as IMAGE tensor [1, H, W, C]

        Raises:
            FileNotFoundError: If previous chunk file not found
            ValueError: If video can't be read or has no frames
        """
        prev_chunk_num = chunk_number - 1
        prev_chunk_base = f"{prev_chunk_num:04d}{suffix}"

        # Find previous chunk file (check all video and image extensions)
        prev_chunk_file = self._find_media_file(
            chunks_path, prev_chunk_base, variation_preference
        )

        if prev_chunk_file is None:
            expected_path = chunks_path / f"{prev_chunk_base}[_NNNNN_].<ext>"
            raise FileNotFoundError(
                f"Previous chunk not found. Expected file matching: {expected_path}\n"
                f"Checked extensions: {ALL_EXTENSIONS}"
            )

        # Load the media and extract last frame (or the image itself)
        return self._extract_last_frame(prev_chunk_file)

    def _find_media_file(
        self, directory: Path, base_name: str, variation_preference: int = -1
    ) -> Path | None:
        """Find a media file matching the base name with any supported extension.

        Handles ComfyUI's counter suffix pattern (e.g., 0000_00001_.webp).

        Args:
            directory: Directory to search in
            base_name: Base filename without extension
            variation_preference: -1 for lowest counter, 0+ for specific counter

        Returns:
            Path to found media file, or None if not found
        """
        # First, try exact match with each extension (no counter suffix)
        for ext in ALL_EXTENSIONS:
            candidate = directory / f"{base_name}{ext}"
            if candidate.exists():
                return candidate

        # Check for files with ComfyUI counter suffix
        # ComfyUI pattern: {base_name}_{counter}_.ext (note trailing underscore)
        matching_files: list[tuple[int, Path]] = []

        for ext in ALL_EXTENSIONS:
            # Pattern matches both {base}_{counter}.ext and {base}_{counter}_.ext
            for pattern in [f"{base_name}_*{ext}", f"{base_name}_*_{ext}"]:
                for file_path in directory.glob(pattern):
                    counter = self._extract_counter_from_filename(
                        file_path.stem, base_name
                    )
                    if counter is not None:
                        matching_files.append((counter, file_path))

        if not matching_files:
            return None

        # Remove duplicates (same file matched by multiple patterns)
        seen_paths: set[Path] = set()
        unique_files: list[tuple[int, Path]] = []
        for counter, file_path in matching_files:
            if file_path not in seen_paths:
                seen_paths.add(file_path)
                unique_files.append((counter, file_path))

        if variation_preference >= 0:
            # Look for specific variation number
            for counter, file_path in unique_files:
                if counter == variation_preference:
                    return file_path
            # If preferred variation not found, fall back to lowest

        # Sort by counter number (lowest first) and return
        unique_files.sort(key=lambda x: x[0])
        return unique_files[0][1]

    def _extract_counter_from_filename(self, stem: str, base_name: str) -> int | None:
        """Extract the ComfyUI counter number from a filename stem.

        Args:
            stem: Filename without extension (e.g., "0000_00001_" or "0000_00001")
            base_name: Expected base name (e.g., "0000")

        Returns:
            Counter number, or None if pattern doesn't match
        """
        # Pattern: {base_name}_{counter} or {base_name}_{counter}_
        # Counter is typically 5 digits but we accept any number
        pattern = rf"^{re.escape(base_name)}_(\d+)_?$"
        match = re.match(pattern, stem)
        if match:
            return int(match.group(1))
        return None

    def _extract_last_frame(self, media_path: Path) -> torch.Tensor:
        """Extract the last frame from a video file, or load an image file.

        Args:
            media_path: Path to the video or image file

        Returns:
            Frame as IMAGE tensor [1, H, W, C] with values in [0, 1]

        Raises:
            ValueError: If media can't be read or has no frames
        """
        ext = media_path.suffix.lower()

        # Handle image files directly
        if ext in IMAGE_EXTENSIONS:
            return self._load_image(media_path)

        # Handle video files
        return self._extract_last_video_frame(media_path)

    def _load_image(self, image_path: Path) -> torch.Tensor:
        """Load an image file as a tensor.

        Args:
            image_path: Path to the image file

        Returns:
            Image as IMAGE tensor [1, H, W, C] with values in [0, 1]

        Raises:
            ValueError: If image can't be read
        """
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Could not read image file: {image_path}")

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to tensor with shape [1, H, W, C] and normalize to [0, 1]
        frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
        frame_tensor = frame_tensor.unsqueeze(0)  # Add batch dimension

        return frame_tensor

    def _extract_last_video_frame(self, video_path: Path) -> torch.Tensor:
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
            # Iterate through all frames to reliably get the last one
            # Note: cap.set(CAP_PROP_POS_FRAMES) is unreliable for many codecs
            last_frame = None
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                last_frame = frame

            if last_frame is None:
                raise ValueError(f"Video has no frames: {video_path}")

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)

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
        variation_preference: int = -1,
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
