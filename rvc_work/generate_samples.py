#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为 assets/weights 下每个 RVC 模型生成同一句话的试听样本。

流程: 取一次中文基础语音 (RVC 服务 /base_tts) → 逐模型切换转换。
模型切换靠改写 rvc_work/model.txt, Windows 服务每次请求会自动热加载。
输出到 rvc_work/试听/ 目录 (再手动拷到 E 盘试听)。
"""
import json
import os
import shutil
import time
import urllib.request

SERVICE = None  # 运行时探测
WORK = "/home/fengduan/kokoro-tts/rvc_work"
OUT_DIR = os.path.join(WORK, "试听")
MODEL_TXT = os.path.join(WORK, "model.txt")
SENTENCE = "你好，我是艾莉。今天想和你聊聊天，可以吗？"
TARGET_SR = 24000


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def find_service():
    """Windows 宿主 IP + 端口 8766。"""
    import subprocess
    out = subprocess.check_output(["ip", "route"], text=True)
    for line in out.splitlines():
        if line.startswith("default"):
            return "http://%s:8766" % line.split()[2]
    raise RuntimeError("找不到 Windows 宿主 IP")


def post_json(path, payload, timeout=180, raw=False):
    req = urllib.request.Request(
        SERVICE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read() if raw else json.loads(r.read())


def wsl_to_unc(path):
    import subprocess
    out = subprocess.check_output(["wslpath", "-w", path], text=True).strip()
    return out


def main():
    global SERVICE
    SERVICE = find_service()
    log("RVC 服务: %s" % SERVICE)
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) 中文基础语音 (一次, 所有模型共用)
    base_wav = os.path.join(WORK, "试听_base.wav")
    try:
        data = post_json("/base_tts", {"text": SENTENCE, "rate": "+0%"}, raw=True)
        with open(base_wav, "wb") as f:
            f.write(data)
        log("基础语音 OK: %.1f KB" % (len(data) / 1024))
    except Exception as e:
        log("基础语音失败: %s" % e)
        return 1

    # 2) 待试听模型列表
    weights = "/mnt/e/download/RVC20260723Nvidia50x0/RVC20260718Nvidia50x0/assets/weights"
    manifest = os.path.join(WORK, "downloaded_models.json")
    names = []
    if os.path.exists(manifest):
        with open(manifest, encoding="utf-8") as f:
            names = json.load(f).get("downloaded", [])
    if not names:
        names = sorted(n for n in os.listdir(weights)
                       if n.endswith(".pth") and "_RVC" in n or n in
                       ("Reiden.Shogun.V5.pth", "HU.TAO.V2.pth",
                        "The.LAST.F20.pth", "villagerminecraft.pth"))
    names = [n for n in names if os.path.exists(os.path.join(weights, n))]
    log("试听模型 %d 个: %s" % (len(names), ", ".join(names)))

    base_unc = wsl_to_unc(base_wav)
    index_rate = float(os.environ.get("INDEX_RATE", "0"))

    # 3) 逐模型转换
    ok = []
    for i, name in enumerate(names, 1):
        out = os.path.join(OUT_DIR, os.path.splitext(name)[0] + ".wav")
        if os.path.exists(out) and os.path.getsize(out) > 10000:
            log("[%d/%d] 已有, 跳过 %s" % (i, len(names), name))
            ok.append(out)
            continue
        try:
            with open(MODEL_TXT, "w", encoding="utf-8") as f:
                f.write(name)
            t0 = time.time()
            data = post_json("/convert", {"input": base_unc,
                                          "index_rate": index_rate}, raw=True)
            with open(out, "wb") as f:
                f.write(data)
            log("[%d/%d] %-26s %.1fs -> %.1f KB"
                % (i, len(names), name, time.time() - t0, len(data) / 1024))
            ok.append(out)
        except Exception as e:
            log("[%d/%d] %-26s 失败: %s" % (i, len(names), name, str(e)[:120]))

    log("完成 %d/%d, 输出目录: %s" % (len(ok), len(names), OUT_DIR))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
