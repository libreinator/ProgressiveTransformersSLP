#!/usr/bin/env python3

import sys
import subprocess
import speech_recognition as sr
from urllib.request import urlretrieve

if __name__ == "__main__":
    urlretrieve(sys.argv[1], "input.opus")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            "input.opus",
            "input.wav",
        ]
    )

    r = sr.Recognizer()
    with sr.AudioFile("input.wav") as source:
        audio = r.record(source)
    text = r.recognize_google(audio)
    print(text)
