# -*- coding: utf-8 -*-
"""Airi RVC 音色转换服务 — Windows runtime python 常驻进程。

由 tts_server.py (WSL) 通过 interop 自动拉起: 该进程是 Windows 原生进程,
直接使用 RTX GPU 与 E:\\ 上的 RVC 环境, 监听 HTTP 提供转换接口。

协议:
  GET  /health            -> {"ok": true, "model": ...}
  POST /convert           -> JSON {"input": "<音频文件路径>"}
                             响应 = WAV 字节 (24000 Hz 单声道)

用法 (参数经环境变量传入, 因为 configs.config 会占用 sys.argv):
    RVC_MODEL=airi.pth RVC_PORT=8766 runtime\\python.exe -I rvc_serve.py
"""
import io
import json
import os
import sys
import threading
import traceback
import tempfile
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

RVC_ROOT = r"E:\download\RVC20260723Nvidia50x0\RVC20260718Nvidia50x0"

# 先清空 sys.argv: RVC 的 configs.config 在 import 时会解析命令行参数
sys.argv = [sys.argv[0]]

os.chdir(RVC_ROOT)
sys.path.insert(0, RVC_ROOT)
os.environ["weight_root"] = os.path.join(RVC_ROOT, "assets", "weights")
os.environ["rmvpe_root"] = os.path.join(RVC_ROOT, "assets", "rmvpe")

import numpy as np
import soundfile as sf

# ---- 安全日志: tts_server 重启后子进程继承的控制台句柄会失效,
#      此时 print 抛 OSError(EINVAL) 会让每个请求都 500。日志必须走自己持有的文件句柄。----
_LOG_FILE = None


def _log_sink():
    global _LOG_FILE
    if _LOG_FILE is None:
        try:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rvc_serve.log")
            _LOG_FILE = open(path, "a", encoding="utf-8", errors="ignore", buffering=1)
        except Exception:
            _LOG_FILE = False
    return _LOG_FILE


def log(msg, **_kw):
    """兼容 print 的写法 (flush=True 之类的参数直接忽略)。"""
    try:
        print(msg, flush=True)
    except Exception:
        f = _log_sink()
        if f:
            try:
                f.write(str(msg) + "\n")
            except Exception:
                pass


def use_own_log_file():
    """启动时把 stdout/stderr 换成自己持有的日志文件, 不再依赖父进程的控制台。"""
    f = _log_sink()
    if f:
        sys.stdout = f
        sys.stderr = f


from configs.config import Config
from infer.vc.modules import VC

from collections import OrderedDict

_lock = threading.Lock()
_vc = None
_model_name = "airi.pth"
_vc_cache = OrderedDict()          # model_name -> VC (LRU, 常驻最近用过的模型)
MODEL_CACHE_MAX = int(os.environ.get("RVC_MODEL_CACHE", "2"))
_state = {"model": None, "device": None, "ready": False, "last_error": None}

MODEL_CFG = r"\\wsl.localhost\Ubuntu\home\fengduan\kokoro-tts\rvc_work\model.txt"

# 训练生成的索引 (音色检索, 显著减小小样本模型的电音/音色漂移)
# 按模型名自动匹配, 避免拿别的模型的索引导致 faiss 维度断言失败 (assert d == self.d)
_index_cache = {}


def index_for_model(model_name):
    """按模型名找配套索引; 找不到返回 "" (不检索)。

    查找顺序: logs/<名>/added_*.index > logs/<名>/*.index > logs/<名>.index
              > assets/indices/<名>*.index
    """
    stem = os.path.splitext(os.path.basename(model_name or ""))[0]
    if not stem:
        return ""
    if stem in _index_cache:
        return _index_cache[stem]

    cands = []
    for sub in (stem, stem.lower(), stem.capitalize()):
        d = os.path.join(RVC_ROOT, "logs", sub)
        if os.path.isdir(d):
            files = [f for f in os.listdir(d) if f.endswith(".index")]
            files.sort(key=lambda f: (0 if f.startswith("added") else 1, f))
            cands += [os.path.join(d, f) for f in files]
    cands.append(os.path.join(RVC_ROOT, "logs", stem + ".index"))
    ind_dir = os.path.join(RVC_ROOT, "assets", "indices")
    if os.path.isdir(ind_dir):
        cands += [os.path.join(ind_dir, f) for f in sorted(os.listdir(ind_dir))
                  if f.endswith(".index") and stem.lower() in f.lower()]

    found = ""
    for c in cands:
        if os.path.isfile(c):
            found = c
            break
    _index_cache[stem] = found
    return found


def read_requested_model():
    """模型名来源: 环境变量 > tts_server 写入的配置文件 > 默认 airi.pth。"""
    model = os.environ.get("RVC_MODEL", "")
    if not model:
        try:
            with open(MODEL_CFG, "r", encoding="utf-8") as f:
                model = f.read().strip()
        except Exception:
            model = ""
    return model or "airi.pth"


def ensure_loaded(model_name):
    """加载/复用 RVC 模型 (带 LRU 缓存: 最近用过的模型常驻, 切回=瞬间)。

    缓存 2 个模型: 每个 net_g 约 60MB(fp16), hubert 约 200MB;
    超出上限时淘汰最久未用的并释放显存。
    """
    global _vc, _model_name
    if _vc is not None and _model_name == model_name:
        return _vc

    cached = _vc_cache.get(model_name)
    if cached is not None:
        _vc_cache.move_to_end(model_name)
        _vc = cached
        _model_name = model_name
        _state["model"] = model_name
        _state["ready"] = True
        log("model from cache: %s" % model_name, flush=True)
        return cached

    config = Config()
    vc = VC(config)
    vc.get_vc(model_name)
    _vc_cache[model_name] = vc
    _vc_cache.move_to_end(model_name)
    _vc = vc
    _model_name = model_name
    _state["model"] = model_name
    _state["device"] = str(config.device)
    _state["ready"] = True
    log("model loaded: %s on %s (cache %d)"
          % (model_name, config.device, len(_vc_cache)), flush=True)

    # 淘汰最久未用的模型, 释放显存
    while len(_vc_cache) > MODEL_CACHE_MAX:
        old_name, old_vc = _vc_cache.popitem(last=False)
        if old_name == _model_name:
            _vc_cache[old_name] = old_vc     # 当前模型不淘汰
            break
        try:
            del old_vc
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        log("model evicted: %s" % old_name, flush=True)
    return vc


def edge_tts_zh_wav(text, rate="+0%"):
    """Windows 侧 edge-tts 中文合成 → 返回 24kHz 单声道 WAV 字节。

    edge-tts 只能出 mp3; 用同目录 ffmpeg.exe 解码成 wav。全在 Windows
    进程内完成 (WSL 侧网络到微软端点不通, 但 Windows 侧可达)。
    """
    import asyncio
    import edge_tts

    tmpdir = tempfile.mkdtemp(prefix="airi_tts_")
    mp3 = os.path.join(tmpdir, "out.mp3")
    wav = os.path.join(tmpdir, "out.wav")
    try:
        asyncio.run(edge_tts.Communicate(
            text, "zh-CN-XiaoyiNeural", rate=rate).save(mp3))
        ffmpeg = os.path.join(RVC_ROOT, "ffmpeg.exe")
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", mp3,
             "-ar", "24000", "-ac", "1", wav], check=True)
        with open(wav, "rb") as f:
            return f.read()
    finally:
        for p in (mp3, wav):
            try:
                os.remove(p)
            except Exception:
                pass
        try:
            os.rmdir(tmpdir)
        except Exception:
            pass


def convert(in_path, index_rate=0.0, resample_sr=24000, f0_method=None):
    """每次请求读模型配置: 变了就热切换 (锁内完成, 防并发竞态)。

    index_rate: 索引检索倍率 0~1。>0 时用训练集的 added 索引做特征检索,
    音色更贴目标、抑制电音; 0 表示不用索引。
    f0_method: pm / rmvpe / fcpe, 默认取环境变量 RVC_F0_METHOD 或 rmvpe。
    """
    if f0_method is None:
        f0_method = os.environ.get("RVC_F0_METHOD", "rmvpe")
    with _lock:
        requested = read_requested_model()
        if requested != _model_name:
            log("model cfg changed: %s -> %s" % (_model_name, requested), flush=True)
        vc = ensure_loaded(requested)
        idx = index_for_model(requested) if index_rate > 0 else ""
        if index_rate > 0 and not idx:
            log("no index for %s, index_rate=0" % requested, flush=True)
        try:
            status, opt = vc.vc_single(
                0, in_path, 0, f0_method,
                idx, index_rate if idx else 0.0,
                resample_sr, 0.25, 0.33
            )
        except AssertionError:
            # 索引维度与模型不符 (faiss: assert d == self.d) -> 降级为不检索
            log("index dim mismatch for %s (%s), fallback to index_rate=0"
                  % (requested, os.path.basename(idx)), flush=True)
            status, opt = vc.vc_single(
                0, in_path, 0, f0_method, "", 0.0, resample_sr, 0.25, 0.33
            )
    if opt is None or opt[0] is None or opt[1] is None:
        raise RuntimeError("conversion failed: %r" % (status,))
    sr, audio = opt
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


def convert_bytes(wav_bytes, index_rate=0.0, f0_method=None):
    """字节直传: 避免 WSL→Windows 的 UNC 文件往返, 用 Windows 本地临时文件。"""
    fd, path = tempfile.mkstemp(prefix="airi_in_", suffix=".wav")
    os.close(fd)
    try:
        with open(path, "wb") as f:
            f.write(wav_bytes)
        return convert(path, index_rate=index_rate, f0_method=f0_method)
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


def warm_up():
    """启动预热: 触发模型/CUDA Graph/f0 提取器的首次开销 (约 3 秒)。

    这样 AIRI 的第一次请求也是热态。
    """
    import numpy as np
    sr = 24000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    tone = (0.05 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, tone, sr, format="WAV")
    try:
        convert_bytes(buf.getvalue())
        log("warm-up 完成 (CUDA Graph/特征提取器已就绪)", flush=True)
        return True
    except Exception:
        _state["last_error"] = traceback.format_exc()
        log("warm-up 失败:\n%s" % _state["last_error"], flush=True)
        return False


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep console quiet
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._json(200, {
                "ok": _state["ready"],
                "model": _state["model"],
                "device": _state["device"],
                "last_error": _state["last_error"],
            })
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/convert_bytes":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                index_rate = float(self.headers.get("X-Index-Rate", "0") or 0)
                data = convert_bytes(body, index_rate=index_rate)
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:
                _state["last_error"] = traceback.format_exc()
                self._json(500, {"error": str(e)})
        elif path == "/convert":
            try:
                length = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(length).decode("utf-8"))
                in_path = req.get("input")
                if not in_path or not os.path.exists(in_path):
                    self._json(400, {"error": "input file not found: %r" % (in_path,)})
                    return
                index_rate = float(req.get("index_rate", 0.0))
                data = convert(in_path, index_rate=index_rate)
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:
                _state["last_error"] = traceback.format_exc()
                self._json(500, {"error": str(e)})
        elif path == "/base_tts":
            try:
                length = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(length).decode("utf-8"))
                text = req.get("text", "")
                rate = req.get("rate", "+0%")
                if not text:
                    self._json(400, {"error": "text required"})
                    return
                data = edge_tts_zh_wav(text, rate=rate)
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:
                _state["last_error"] = traceback.format_exc()
                self._json(500, {"error": str(e)})
        else:
            self._json(404, {"error": "not found"})


def main():
    use_own_log_file()
    global _model_name
    # 模型名来源: 环境变量 > tts_server 写入的配置文件 > 默认 airi.pth
    model = os.environ.get("RVC_MODEL", "")
    if not model:
        try:
            with open(MODEL_CFG, "r", encoding="utf-8") as f:
                model = f.read().strip()
        except Exception:
            model = ""
    if not model:
        model = "airi.pth"
    _model_name = model
    host = "0.0.0.0"
    port = int(os.environ.get("RVC_PORT", "8766"))

    # 预加载 (失败也要继续, health 会显示未就绪)
    try:
        ensure_loaded(_model_name)
        warm_up()   # 把首次 CUDA Graph 捕获等开销挪到启动阶段
    except Exception:
        _state["last_error"] = traceback.format_exc()
        log("preload failed:\n%s" % _state["last_error"], flush=True)

    srv = ThreadingHTTPServer((host, port), Handler)
    log("RVC serve listening on %s:%d model=%s" % (host, port, _model_name), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
