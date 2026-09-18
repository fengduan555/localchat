# -*- coding: utf-8 -*-
"""把训练检查点 G_NNN.pth 转成 RVC 推理格式 airi_eNNN.pth (Windows runtime 执行)"""
import os, sys, torch
RVC = r"E:\download\RVC20260723Nvidia50x0\RVC20260718Nvidia50x0"
os.chdir(RVC); sys.path.insert(0, RVC)
os.environ["weight_root"] = os.path.join(RVC, "assets", "weights")

def convert(src, dst, epoch, version="v2"):
    ckpt = torch.load(src, map_location="cpu")
    if "model" in ckpt:
        ckpt = ckpt["model"]
    weight = {}
    for k, v in ckpt.items():
        if "enc_q" in k:
            continue
        weight[k] = v.half()
    opt = {
        "weight": weight,
        "config": [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1",
                   [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                   [10, 10, 2, 2], 512, [16, 16, 4, 4], 109, 256, 40000],
        "info": "%sepoch" % epoch,
        "sr": "40k",
        "f0": 1,
        "version": version,
    }
    torch.save(opt, dst)
    print("saved:", dst, os.path.getsize(dst), "bytes")

convert(os.path.join(RVC, "logs", "Airi", "G_360.pth"),
        os.path.join(RVC, "assets", "weights", "airi_e360.pth"), 360)
