#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""枚举 Hanazar-Games/RVC-voice-models 的 release 资产 (.pth 模型)。"""
import re
import sys
import time
import urllib.request

REPO = "Hanazar-Games/RVC-voice-models"
TAGS = ["2March_7", "3Arlecchino", "4Ayaka", "5Furina", "6Hutao",
        "7Firefly", "8Villager", "9Nahida", "10Shogun", "11Wanderer"]


def fetch(url, tries=4, timeout=25):
    for i in range(tries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            print("  retry %d/%d: %s" % (i + 1, tries, str(e)[:80]), flush=True)
            time.sleep(2)
    return ""


def assets_for(tag):
    html = fetch(
        "https://github.com/%s/releases/expanded_assets/%s" % (REPO, tag))
    if not html:
        return []
    names = re.findall(
        r'href="/%s/releases/download/%s/([^"]+)"' % (REPO, re.escape(tag)),
        html)
    out = []
    for n in dict.fromkeys(names):
        out.append(n)
    return out


def main():
    for tag in TAGS:
        a = assets_for(tag)
        pths = [x for x in a if x.lower().endswith(".pth")]
        zips = [x for x in a if x.lower().endswith(".zip")]
        print("%-14s pth=%s zip=%s" % (tag, pths or "-", zips or "-"),
              flush=True)
        for p in pths:
            print("   URL https://github.com/%s/releases/download/%s/%s"
                  % (REPO, tag, p), flush=True)


if __name__ == "__main__":
    main()
