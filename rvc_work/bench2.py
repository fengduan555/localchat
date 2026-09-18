#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A 方案优化后基准: 端到端 (AIRI 视角) + 分段耗时。每句都不同, 避免假象。"""
import io
import json
import subprocess
import time
import urllib.request

import numpy as np
import soundfile as sf

SENTENCES = [
    "你好，我是艾莉。今天想和你聊聊天，可以吗？",
    "今天天气不错，要不要一起出去走走？",
    "我刚刚在看一本书，里面讲了很多有趣的事情。",
    "你最喜欢吃什么？我比较喜欢甜一点的东西。",
    "已经很晚了，记得早点休息哦。",
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


def main():
    base = host()
    print("=" * 62)
    print("A 方案基准 (每句不同, 热态)")
    print("=" * 62)

    # 预热一次
    post("http://localhost:8765/v1/audio/speech",
         json.dumps({"model": "kokoro", "input": "预热。",
                     "voice": "airi"}).encode(),
         {"Content-Type": "application/json"})

    print("\n【端到端】AIRI 视角 (tts_server 8765)")
    totals = []
    for i, s in enumerate(SENTENCES):
        t0 = time.time()
        data = post("http://localhost:8765/v1/audio/speech",
                    json.dumps({"model": "kokoro", "input": s,
                                "voice": "airi"}).encode(),
                    {"Content-Type": "application/json"})
        dt = time.time() - t0
        totals.append(dt)
        audio, sr = sf.read(io.BytesIO(data))
        print("  %d) %5.2fs  (音频 %.2fs)  %s"
              % (i + 1, dt, len(audio) / sr, s[:18]))
    print("  → 平均 %.2fs, 最快 %.2fs, 最慢 %.2fs"
          % (np.mean(totals), min(totals), max(totals)))

    print("\n【分段】本地 Kokoro 中文 vs RVC 转换")
    from kokoro import KPipeline
    p = KPipeline(lang_code='z')
    for s in SENTENCES[:3]:
        t0 = time.time()
        audio = np.concatenate([a.cpu().numpy()
                                for _, _, a in p(s, voice='zf_xiaoyi')])
        t_tts = time.time() - t0
        buf = io.BytesIO()
        sf.write(buf, audio, 24000, format='WAV')
        t0 = time.time()
        post(base + "/convert_bytes", buf.getvalue(),
             {"Content-Type": "audio/wav", "X-Index-Rate": "0"})
        t_rvc = time.time() - t0
        print("  基础语音 %.2fs + RVC %.2fs = %.2fs  (音频 %.2fs)"
              % (t_tts, t_rvc, t_tts + t_rvc, len(audio) / 24000))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
