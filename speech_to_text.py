#!/usr/bin/env python3

import sys
import subprocess
import speech_recognition as sr
from urllib.request import urlretrieve

if __name__ == "__main__":

    if "http" in sys.argv[1]:
        # print("url downloading")
        urlretrieve(sys.argv[1], "input.ogg")
        filename = "input.ogg"
    else:
        filename = sys.argv[1]
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                sys.argv[1],
                "input.wav",
            ],
            check=True,
        )
    except:
        pass

    r = sr.Recognizer()
    with sr.AudioFile("input.wav") as source:
        audio = r.record(source)
    text = r.recognize_google(audio)
    print(text)
