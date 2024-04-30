#!/usr/bin/env python3

import glob
import os
import urllib.request
import base64
import subprocess
import json
from flask import Flask, request, send_file

app = Flask(__name__)


@app.route("/speech", methods=["GET", "POST"])
def speech():
    if request.method == "POST":
        # print(request.data)
        data = str(request.data)
        data_start = data.find(",") + 1
        actual_data = data[data_start:]
        real_data = base64.b64decode(actual_data)
        print(actual_data[:20])
        with open("file.ogg", "wb") as f:
            f.write(real_data)
            subprocess.run(
                [
                    "./speech_to_sign.sh",
                    "file.ogg",
                ]
            )

        return "Hello, World!"


@app.route("/speechurl", methods=["GET", "POST"])
def speechurl():
    if request.method == "POST":
        print(request.data)
        url = json.loads(request.data)[0]
        print(url)

        subprocess.run(
            [
                "./speech_to_sign.sh",
                url,
            ]
        )

        return "Hello, World!"


@app.route("/video")
def video():
    list_of_files = glob.glob("Models/Base/test_videos/*.mp4")
    latest_file = max(list_of_files, key=os.path.getctime)
    return send_file(latest_file)
