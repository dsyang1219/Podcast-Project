#!/bin/bash
# Automated YouTube playlist → transcription pipeline
# Usage: ./youtube_pipeline.sh <playlist_url> [playlist_url2 ...]

# ── Configuration ─────────────────────────────────────────────────────────────
PROCESSED_LOG="$HOME/Podcast-Project/processed.txt"
QUEUE_DIR="$HOME/Podcast-Project/queue"
HF_TOKEN="${HF_TOKEN:-YOUR_HF_TOKEN_HERE}"
COOKIES="$HOME/Podcast-Project/cookies.txt"

MIN_DURATION=2700   # 45 minutes
YTDLP="$HOME/.local/bin/yt-dlp"
export PATH="$HOME/.deno/bin:$HOME/.local/bin:$PATH"
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <youtube_playlist_url> [playlist_url2 ...]"
  exit 1
fi

if [[ "$HF_TOKEN" == "YOUR_HF_TOKEN_HERE" ]]; then
  echo "ERROR: Set your HuggingFace token via:  export HF_TOKEN=your_token_here"
  exit 1
fi

mkdir -p "$QUEUE_DIR"
touch "$PROCESSED_LOG"
rm -f "$QUEUE_DIR/STOP"

sanitize_title() {
  echo "$1" | sed 's/[\/\\:*?"<>|]/_/g' | sed 's/  */ /g' | sed 's/^ //;s/ $//'
}

for PLAYLIST_URL in "$@"; do
  echo ""
  echo "=== Fetching video list from: $PLAYLIST_URL ==="

  mapfile -t VIDEO_LINES < <(
    $YTDLP --flat-playlist \
      --match-filter "duration>=${MIN_DURATION}" \
      --cookies "$COOKIES" \
      --print "%(id)s|||%(title)s" \
      "$PLAYLIST_URL"
  )

  if [[ ${#VIDEO_LINES[@]} -eq 0 ]]; then
    echo "No videos found (or none meet the duration filter). Skipping."
    continue
  fi

  echo "Found ${#VIDEO_LINES[@]} video(s) meeting duration filter."

  for LINE in "${VIDEO_LINES[@]}"; do
    VIDEO_ID="${LINE%%|||*}"
    VIDEO_TITLE="${LINE#*|||}"
    SAFE_TITLE=$(sanitize_title "$VIDEO_TITLE")
    VIDEO_URL="https://www.youtube.com/watch?v=${VIDEO_ID}"

    if grep -qxF "$VIDEO_ID" "$PROCESSED_LOG"; then
      echo "--- Skipping $VIDEO_ID ($SAFE_TITLE) — already processed"
      continue
    fi

    # Wait if queue is backed up (transcriber is slower than downloader)
    while [[ $(ls "$QUEUE_DIR"/*.meta 2>/dev/null | wc -l) -ge 2 ]]; do
      sleep 5
    done

    echo "  Downloading: $SAFE_TITLE"
    AUDIO_FILE="$QUEUE_DIR/${VIDEO_ID}.%(ext)s"

    if ! $YTDLP -f "bestaudio" \
      --no-playlist \
      --cookies "$COOKIES" \
      -o "$AUDIO_FILE" \
      "$VIDEO_URL"; then
      echo "  WARNING: Download failed for $VIDEO_ID — skipping"
      rm -f "$QUEUE_DIR/${VIDEO_ID}".*
      continue
    fi

    # Write meta file — signals transcriber that this file is ready
    echo "${VIDEO_ID}|||${SAFE_TITLE}" > "$QUEUE_DIR/${VIDEO_ID}.meta"
    echo "  Queued: $SAFE_TITLE"
  done
done

# Signal transcriber to stop once queue drains
touch "$QUEUE_DIR/STOP"
echo ""
echo "=== All downloads complete. Waiting for transcriber to finish... ==="

# Wait for transcriber to drain the queue
while [[ $(ls "$QUEUE_DIR"/*.meta 2>/dev/null | wc -l) -gt 0 ]]; do
  sleep 10
done

echo "=== Pipeline complete ==="
