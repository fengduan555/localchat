#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""并发测试: 模拟 AIRI 一次发言 3 句, 对比串行/并发耗时。"""
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

SENTS = [
    "你好，我是艾莉。今天想和你聊聊天，可以吗？",
    "今天天气不错，要不要一起出去走走？",
    "已经很晚了，记得早点休息哦。",
]


def call(text):
    t0 = time.time()
    req = urllib.request.Request(
        "http://localhost:8765/v1/audio/speech",
        data=json.dumps({"model": "kokoro", "input": text,
                         "voice": "airi"}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        n = len(r.read())
    return time.time() - t0, n


def main():
    print("预热 ...")
    call("预热。")

    print("\n【串行】3 句依次请求 (AIRI 常见行为)")
    t0 = time.time()
    for s in SENTS:
        dt, n = call(s)
        print("   %.2fs  %s" % (dt, s[:16]))
    serial = time.time() - t0
    print("   → 合计 %.2fs" % serial)

    print("\n【并发】3 句同时请求")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=3) as ex:
        for dt, n in ex.map(call, SENTS):
            print("   %.2fs" % dt)
    parallel = time.time() - t0
    print("   → 合计 %.2fs" % parallel)
    print("\n提速: %.0f%%" % ((serial - parallel) / serial * 100))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
