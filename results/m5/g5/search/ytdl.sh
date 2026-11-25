#!/bin/zsh
cd results/m5/g5
for id in "$@"; do
  [ -f raw_yt/$id.mp4 ] && continue
  yt-dlp -q --no-warnings --extractor-args "youtube:player_client=android" -f 18 -o "raw_yt/$id.%(ext)s" "https://www.youtube.com/watch?v=$id" 2>&1 | tail -2
  echo "$id $(ls -la raw_yt/$id.mp4 2>/dev/null | awk '{print $5}')"
  sleep 3
done
