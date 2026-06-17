#!/bin/bash

source /usr/local/bin/activate_whisperx.sh

for num in 2491 2492 2493 2494; do
  echo "=== Processing joerogan${num} ==="
  
  ffmpeg -y -i ~/Projects/MyProject/joerogan${num}.mp4 \
    -vn -acodec libmp3lame -q:a 4 \
    ~/Projects/MyProject/joerogan${num}.mp3
  
  whisperx ~/Projects/MyProject/joerogan${num}.mp3 \
    --model large-v2 \
    --device cuda \
    --compute_type float16 \
    --language en \
    --output_format vtt \
    --output_dir ~/Projects/MyProject \
    --diarize \
    --hf_token YOUR_HF_TOKEN_HERE
  
  echo "=== Done with joerogan${num} ==="
done
