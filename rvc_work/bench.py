#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分段计时: AIRI 请求全链路 vs 各环节耗时。只测不改。"""
import json
import os
import subprocess
import sys
import time
import urllib.request

WORK = "/home/fengduan/kokoro-tts/rvc_work"
SENTENCE = "你好，我是艾莉。今天想和你聊聊天，可以吗？"


def host():
    out = subprocess.check_output(["ip", "route"], text=True)
    for line in out.splitlines():
        if line.startswith("default"):
            return "http://%s:8766" % line.split()[2]
    raise RuntimeError("no host")


def wsl_to_unc(p):
    return subprocess.check_output(["wslpath", "-w", p], text=True).strip()


def post_json(base, path, payload, raw=False, timeout=180):
    req = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read() if raw else json.loads(r.read())


def timed(label, fn):
    t0 = time.time()
    try:
        out = fn()
        print("  %-34s %5.2fs" % (label, time.time() - t0))
        return out
    except Exception as e:
        print("  %-34s 失败: %s" % (label, str(e)[:80]))
        return None


def main():
    import soundfile as sf
    base = host()
    print("RVC 服务:", base)
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    # 预热一次
    post_json(base, "/base_tts", {"text": "预热。", "rate": "+0%"}, raw=True)

    for i in range(rounds):
        print("第 %d 轮:" % (i + 1))

        def do_base():
            data = post_json(base, "/base_tts",
                             {"text": SENTENCE, "rate": "+0%"}, raw=True)
            p = os.path.join(WORK, "bench_base.wav")
            with open(p, "wb") as f:
                f.write(data)
            return p, len(data)

        r = timed("① 中文基础语音 /base_tts (Windows edge-tts)", do_base)
        if not r:
            return 1
        base_path, base_size = r

        def do_convert():
            return post_json(base, "/convert",
                             {"input": wsl_to_unc(base_path),
                              "index_rate": 0.0}, raw=True)

        out = timed("② RVC 转换 /convert (含 UNC 读取)", do_convert)
        if out:
            p = os.path.join(WORK, "bench_out.wav")
            with open(p, "wb") as f:
                f.write(out)
            y, sr = sf.read(p)
            print("     输出 %.2fs 音频, %d Hz" % (len(y) / sr, sr))

        def do_full():
            req = urllib.request.Request(
                "http://localhost:8765/v1/audio/speech",
                data=json.dumps({"model": "kokoro", "input": SENTENCE,
                                 "voice": "airi"}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                return len(resp.read())

        timed("③ 端到端 (tts_server 8765, 同 AIRI)", do_full)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
