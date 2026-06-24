#!/bin/bash
# Automated YouTube playlist → transcription pipeline
# Usage: ./youtube_pipeline.sh <playlist_url>
#   or set PLAYLIST_URL below to run without arguments

# ── Configuration ─────────────────────────────────────────────────────────────
PLAYLIST_URL="${1:-}"
OUTPUT_DIR="$HOME/Podcast-Project/output"
PROCESSED_LOG="$HOME/Podcast-Project/processed.txt"
WORK_DIR="$HOME/Podcast-Project/tmp"
HF_TOKEN="${HF_TOKEN:-YOUR_HF_TOKEN_HERE}"   # export HF_TOKEN=... or replace inline

WHISPER_MODEL="large-v2"
WHISPER_LANGUAGE="en"
COMPUTE_TYPE="float16"
DEVICE="cuda"
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

if [[ -z "$PLAYLIST_URL" ]]; then
  echo "Usage: $0 <youtube_playlist_url>"
  echo "  or:  PLAYLIST_URL=<url> $0"
  exit 1
fi

if [[ "$HF_TOKEN" == "YOUR_HF_TOKEN_HERE" ]]; then
  echo "ERROR: Set your HuggingFace token via:  export HF_TOKEN=your_token_here"
  exit 1
fi

# Activate whisperx environment
source /usr/local/bin/activate_whisperx.sh

mkdir -p "$OUTPUT_DIR" "$WORK_DIR"
touch "$PROCESSED_LOG"

echo "=== Fetching video list from playlist ==="
# Get list of video IDs from the playlist
mapfile -t VIDEO_IDS < <(yt-dlp --flat-playlist --print id "$PLAYLIST_URL")

if [[ ${#VIDEO_IDS[@]} -eq 0 ]]; then
  echo "No videos found in playlist. Check the URL and try again."
  exit 1
fi

echo "Found ${#VIDEO_IDS[@]} video(s) in playlist."

for VIDEO_ID in "${VIDEO_IDS[@]}"; do
  VIDEO_URL="https://www.youtube.com/watch?v=${VIDEO_ID}"

  # Skip already-processed videos
  if grep -qxF "$VIDEO_ID" "$PROCESSED_LOG"; then
    echo "--- Skipping $VIDEO_ID (already processed)"
    continue
  fi

  echo ""
  echo "=== Processing $VIDEO_ID ==="

  VIDEO_FILE="$WORK_DIR/${VIDEO_ID}.mp4"
  AUDIO_FILE="$WORK_DIR/${VIDEO_ID}.mp3"

  # Download video
  echo "  Downloading..."
  yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best" \
    -o "$VIDEO_FILE" \
    "$VIDEO_URL"

  # Extract audio
  echo "  Extracting audio..."
  ffmpeg -y -i "$VIDEO_FILE" \
    -vn -acodec libmp3lame -q:a 4 \
    "$AUDIO_FILE"

  # Delete video file now that audio is extracted
  rm -f "$VIDEO_FILE"

  # Transcribe
  echo "  Transcribing..."
  whisperx "$AUDIO_FILE" \
    --model "$WHISPER_MODEL" \
    --device "$DEVICE" \
    --compute_type "$COMPUTE_TYPE" \
    --language "$WHISPER_LANGUAGE" \
    --output_format vtt \
    --output_dir "$OUTPUT_DIR" \
    --diarize \
    --hf_token "$HF_TOKEN"

  # Clean up audio file
  rm -f "$AUDIO_FILE"

  # Mark as processed
  echo "$VIDEO_ID" >> "$PROCESSED_LOG"

  echo "=== Done with $VIDEO_ID — VTT saved to $OUTPUT_DIR ==="
done

echo ""
echo "=== Pipeline complete ==="
