#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下载 Hanazar-Games/RVC-voice-models 全部 .pth 模型到 RVC assets/weights。

- 用 api.github.com 枚举 release 资产 (带重试)
- 用 Windows 侧 curl.exe 下载, 直接写 E 盘 (不受 WSL 沙箱限制)
- 下载后校验 torch 检查点结构 (weight/config/f0/version)
"""
import json
import os
import subprocess
import time
import urllib.request

REPO = "Hanazar-Games/RVC-voice-models"
API = "https://api.github.com/repos/%s/releases?per_page=100" % REPO
WEIGHTS_WIN = (r"E:\download\RVC20260723Nvidia50x0\RVC20260718Nvidia50x0"
               r"\assets\weights")
WEIGHTS_LOCAL = ("/mnt/e/download/RVC20260723Nvidia50x0/"
                 "RVC20260718Nvidia50x0/assets/weights")
WIN_CURL = "/mnt/c/Windows/System32/curl.exe"
MANIFEST = "/home/fengduan/kokoro-tts/rvc_work/downloaded_models.json"


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def fetch_json(url, tries=6, timeout=20):
    for i in range(tries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0",
                              "Accept": "application/vnd.github+json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            log("  API 重试 %d/%d: %s" % (i + 1, tries, str(e)[:70]))
            time.sleep(3)
    return None


def download(url, name):
    dest_win = os.path.join(WEIGHTS_WIN, name)
    dest_local = os.path.join(WEIGHTS_LOCAL, name)
    if os.path.exists(dest_local) and os.path.getsize(dest_local) > 40 * 1048576:
        log("已存在跳过: %s (%.1f MB)"
            % (name, os.path.getsize(dest_local) / 1048576))
        return True
    log("下载 %s ..." % name)
    t0 = time.time()
    p = subprocess.run(
        [WIN_CURL, "-L", "--retry", "5", "--retry-delay", "3",
         "--connect-timeout", "25", "--max-time", "900",
         "-sS", "-o", dest_win, url],
        capture_output=True, text=True)
    if p.returncode != 0:
        log("  失败: %s" % (p.stderr.strip()[:200] or "exit %d" % p.returncode))
        return False
    if not os.path.exists(dest_local):
        log("  失败: 文件未生成")
        return False
    log("  完成 %.1f MB (%.0fs)" % (os.path.getsize(dest_local) / 1048576,
                                    time.time() - t0))
    return True


def verify(path):
    try:
        import torch
        cpt = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(cpt, dict) or "weight" not in cpt:
            return "格式异常 keys=%s" % (sorted(cpt.keys())[:6]
                                        if isinstance(cpt, dict) else type(cpt))
        return "OK info=%s sr=%s f0=%s ver=%s keys=%d" % (
            cpt.get("info"), cpt.get("sr"), cpt.get("f0"),
            cpt.get("version"), len(cpt.get("weight", {})))
    except Exception as e:
        return "载入失败: %s" % str(e)[:120]


def main():
    rs = fetch_json(API)
    if not rs:
        log("无法获取 release 列表, 退出")
        return 1
    plan = []
    for r in rs:
        for a in r.get("assets", []):
            if a["name"].lower().endswith(".pth"):
                plan.append({"tag": r["tag_name"], "name": a["name"],
                             "size": a["size"],
                             "url": a["browser_download_url"]})
    log("待下载 %d 个模型, 合计 %.0f MB"
        % (len(plan), sum(p["size"] for p in plan) / 1048576))

    done, failed = [], []
    for item in plan:
        if download(item["url"], item["name"]):
            done.append(item["name"])
        else:
            failed.append(item["name"])

    log("")
    log("=== 校验结果 ===")
    results = []
    for name in done:
        path = os.path.join(WEIGHTS_LOCAL, name)
        info = verify(path)
        log("%-26s %s" % (name, info))
        results.append({"name": name, "verify": info})

    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump({"models": plan, "downloaded": done, "failed": failed,
                   "verify": results}, f, ensure_ascii=False, indent=2)
    log("")
    log("成功 %d, 失败 %d %s" % (len(done), len(failed),
                                ("(" + ", ".join(failed) + ")") if failed else ""))
    log("清单: %s" % MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
