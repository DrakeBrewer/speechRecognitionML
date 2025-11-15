#!/bin/bash

curr_dir="$(dirname "$0")"
audio_dir="${curr_dir}/../data/audio/raw"

for dir in $audio_dir; do
	file=$(find "$dir" -type f -name "*.m4a" | head -n 1)
	if [ -n "$file" ]; then
		speaker=$(basename "$dir")
		output="${audio_dir}/${speaker}.wav"
		
		if [ -f "$output" ]; then
			echo "skipping $speaker"
			continue
		fi

		ffmpeg -i "$file" -ac 1 -ar 22050 "$output"
		echo "Converted $speaker"
	fi
done

