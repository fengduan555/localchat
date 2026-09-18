#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 A/B 对比样音: 本地 Kokoro 晓伊 vs edge 晓伊 (都经 RVC e360)。"""
import io
import json
import os
import subprocess
import sys
import urllib.request

import numpy as np
import soundfile as sf

OUT = "/home/fengduan/kokoro-tts/rvc_work/试听"
SENTS = [
    ("1问候", "你好，我是艾莉。今天想和你聊聊天，可以吗？"),
    ("2日常", "今天天气不错，要不要一起出去走走？"),
    ("3晚安", "已经很晚了，记得早点休息哦。"),
]


def host():
    out = subprocess.check_output(["ip", "route"], text=True)
    for line in out.splitlines():
        if line.startswith("default"):
            return "http://%s:8766" % line.split()[2]
    raise RuntimeError("no host")


def post(url, data, headers, timeout=120):
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def convert(base_wav_bytes, base):
    return post(base + "/convert_bytes", base_wav_bytes,
                {"Content-Type": "audio/wav", "X-Index-Rate": "0"})


def main():
    base = host()
    os.makedirs(OUT, exist_ok=True)

    from kokoro import KPipeline
    zh = KPipeline(lang_code='z')

    for tag, text in SENTS:
        # A) 本地 Kokoro
        audio = np.concatenate([a.cpu().numpy()
                                for _, _, a in zh(text, voice='zf_xiaoyi')])
        buf = io.BytesIO()
        sf.write(buf, audio, 24000, format='WAV')
        out = convert(buf.getvalue(), base)
        p = os.path.join(OUT, f"新方案_本地Kokoro_{tag}.wav")
        with open(p, "wb") as f:
            f.write(out)
        print("OK", p)

        # B) edge (Windows 侧)
        wav = post(base + "/base_tts",
                   json.dumps({"text": text, "rate": "+0%"}).encode(),
                   {"Content-Type": "application/json"})
        out2 = convert(wav, base)
        p2 = os.path.join(OUT, f"旧方案_edge_{tag}.wav")
        with open(p2, "wb") as f:
            f.write(out2)
        print("OK", p2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
