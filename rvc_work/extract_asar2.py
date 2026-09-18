#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 Electron app.asar 提取应用代码 (排除 node_modules) 以便检索。"""
import json
import os
import sys

ASAR = sys.argv[1] if len(sys.argv) > 1 else "/mnt/d/airi/resources/app.asar"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/airi_new_x"

with open(ASAR, "rb") as f:
    head = f.read(160 << 20)
i = head.find(b'{"files":')
txt = head[i:].decode("latin-1")
tree, end = json.JSONDecoder().raw_decode(txt)
print("header bytes:", end)


def walk(node, path=""):
    for name, meta in (node.get("files") or {}).items():
        p = path + "/" + name if path else name
        if "files" in meta:
            yield from walk(meta, p)
        else:
            yield p, meta.get("size", 0), int(meta.get("offset", 0))


keep = []
for p, size, off in walk(tree):
    low = p.lower()
    if "node_modules" in low or low.startswith("locales/"):
        continue
    if low.endswith((".js", ".mjs", ".json", ".vue", ".ts", ".html")):
        keep.append((p, size, off))
print("candidates:", len(keep), "total MB: %.1f"
      % (sum(s for _, s, _ in keep) / 1048576))

with open(ASAR, "rb") as f:
    for p, size, off in keep:
        f.seek(off)
        data = f.read(size)
        dest = os.path.join(OUT, p)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as g:
            g.write(data)
print("done ->", OUT)
