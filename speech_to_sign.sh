#!/usr/bin/env bash

. ~/.bashrc

if (( $# != 1 ))
then
    echo "Usage: speech_to_sign.sh URL_OF_AUDIO"
fi

conda activate speech
text="$(python speech_to_text.py "$1")"
echo "Text: $text"
conda deactivate

http -vvv --form localhost:8889/text query="$text"

filename="Models/Base/test_videos/$(ls -t Models/Base/test_videos | head -n1)"
new_filename="Models/Base/test_videos/out.mp4"
ffmpeg -hide_banner -loglevel error -y -i "$filename" "$new_filename"

echo "File: $new_filename"
#clapper "$new_filename"
