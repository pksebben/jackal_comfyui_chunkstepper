# jackal-comfyui-chunkstepper

ComfyUI custom node for iterative video chunk generation. Replaces manual node spaghetti for stepping through video generation in sequential chunks.

## Purpose

When generating long videos with AnimateDiff, you typically render in chunks (e.g., 16-32 frames at a time), using the last frame of chunk N-1 as the init frame for chunk N. This node manages that workflow:

- Outputs a standardized file path for the current chunk
- Provides the correct starting frame (init image for chunk 0, last frame of previous chunk for chunk N)

## Node: VideoChunkStepper

### Inputs

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `chunks_directory` | STRING | Yes | Base directory where chunks are saved |
| `chunk_number` | INT | Yes | Current chunk index (0-indexed) |
| `init_image` | IMAGE | Yes | Starting frame for chunk 0 |
| `suffix` | STRING | No | Optional string appended to chunk filename (e.g., `_draft`, `_v2`) |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `output_path` | STRING | Full file path for current chunk, without extension |
| `frame` | IMAGE | Single frame to use as init for this chunk |

### Behavior

**Output path format:**
```
{chunks_directory}/{chunk_number:04d}{suffix}
```

Examples:
- Chunk 0, no suffix: `/path/to/chunks/0000`
- Chunk 42, suffix `_draft`: `/path/to/chunks/0042_draft`

The path excludes file extension — let your video saver node handle that.

**Frame output logic:**
- If `chunk_number == 0`: output `init_image` directly
- If `chunk_number > 0`: load the video at chunk N-1, extract the last frame, output that

**Previous chunk discovery:**
The node should find the previous chunk file regardless of extension. Check for common video formats (mp4, webm, mov, etc.).

### Error Handling

- If `chunks_directory` doesn't exist, create it
- If `chunk_number > 0` and previous chunk file not found: raise clear error with expected path
- If previous chunk video can't be read or has no frames: raise clear error

## Stitching Script

Standalone CLI script to concatenate chunks into a final video with audio.

### Usage

```
python stitch.py <chunks_dir> <audio_path> [output_path]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `chunks_dir` | Yes | Directory containing numbered chunk files |
| `audio_path` | Yes | Path to audio file to mux into final video |
| `output_path` | No | Output file path. Defaults to `output.mp4` in current working dir |

### Behavior

**Chunk discovery:**
- Scan directory for files matching pattern `{chunk_number:04d}*.*`
- Process chunks in ascending numerical order (0000, 0001, 0002, ...)
- Stop when a chunk number is missing (i.e., don't skip gaps)

**Handling duplicates:**
ComfyUI appends its own counter to filenames (e.g., `0001_00001.mp4`, `0001_00002.mp4`). When multiple files match the same chunk number:
- Display thumbnail preview of each candidate
- Prompt user to select which one to use
- Stretch goal: show actual video playback for selection, not just thumbnail

**Output:**
- Uncompressed MP4 (e.g., using `libx264` with `-crf 0` or a lossless codec)
- Audio from provided file muxed in

### Error Handling

- If `chunks_dir` doesn't exist or is empty: clear error
- If `audio_path` doesn't exist: clear error
- If chunk 0000 is missing: clear error
- If a gap in chunk numbers is detected: warn and stop at the gap

## Installation

Drop into `ComfyUI/custom_nodes/` and restart ComfyUI.

## Future Considerations

- Additional output formats/codecs for stitching script
- `VideoChunkInfo` utility node that outputs metadata without loading frames (for conditional workflows)
