#!/bin/bash

mkdir -p audio_data

for dir in */; do
	file=$(find "$dir" -type f -name "*.m4a" | head -n 1)
	if [ -n "$file" ]; then
		speaker=$(basename "$dir")
		ffmpeg -i "$file" -ac 1 -ar 22050 "audio_data/${speaker}.wav"
		echo "Converted $speaker"
	fi
done

