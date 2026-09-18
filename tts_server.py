#!/usr/bin/env python3
"""Local TTS server with high-precision voice cloning — OpenAI-compatible API.

音色链路 (v3, 已移除 OpenVoice):
  - 基础语音: 本地 Kokoro 中文 (离线 CPU) / Kokoro 英文
  - 音色转换: Windows 侧 RVC GPU 服务 (13 个角色模型, 热切换 + LRU 缓存)
  - 中文合成由 tts_server 本地完成; 音色转换经 HTTP 交给 RVC 服务

Usage:
    python3 tts_server.py              # Start on port 8765
    python3 tts_server.py --port 8080  # Custom port

AIRI configuration:
    - Provider: OpenAI Compatible Audio Speech
    - Base URL: http://localhost:8765/v1
    - API Key:  anything (ignored)
    - Model:    kokoro
    - Voice:    airi  (若叶睦音色: Kokoro 基础语音 → Windows RVC GPU 转换)
                af_heart / 其他注册克隆音色照旧可用
"""

import sys

import os

# HuggingFace 缓存放在工作区内 (沙箱只允许写工作区, 且中文声线要下载)
os.environ.setdefault('HF_HOME', '/home/fengduan/kokoro-tts/hf_cache')

# ── 离线模式 ─────────────────────────────────────────────────────────
# 跳过 HuggingFace 联网校验 (必须在本文件 import kokoro / transformers 之前设置)。
# 实测收益: 中文 pipeline 加载 34~37s → 10.3s (网络校验每次都要超时重试)。
# 声线/模型缺失时不再自动下载 —— 需要用新声线时, 先临时关掉离线补齐缓存。
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import io
import json
import time
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
import torch
import librosa
import threading

# 限制 torch CPU 线程数: 避免 Kokoro 推理与 Windows 侧 RVC/其它任务抢占 CPU
# 导致偶发的"合成 30 秒"现象 (默认会用满所有核心)
try:
    _cpu = os.cpu_count() or 4
    _threads = max(2, min(6, _cpu - 2))
    torch.set_num_threads(_threads)
    print(f'[CPU] torch 线程数限制为 {_threads} (共 {_cpu} 核)')
except Exception:
    pass

from kokoro import KPipeline
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.responses import Response
import uvicorn

# ── Config ──────────────────────────────────────────────────────────
PROJECT_DIR = Path('/home/fengduan/kokoro-tts')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAMPLE_RATE_KOKORO = 24000   # Kokoro TTS native sample rate

# Kokoro voice list
VOICES_US = [
    'af_heart', 'af_bella', 'af_sarah', 'af_nicole', 'af_sky',
    'af_alloy', 'af_aoede', 'af_jessica', 'af_kore', 'af_river',
    'af_nova', 'am_adam', 'am_michael', 'am_fenrir', 'am_puck',
    'am_echo', 'am_eric', 'am_liam', 'am_onyx'
]

FEMALE_BASE_VOICES = [
    'af_heart', 'af_bella', 'af_sarah', 'af_nicole', 'af_sky',
    'af_alloy', 'af_aoede', 'af_jessica', 'af_kore', 'af_river', 'af_nova'
]
MALE_BASE_VOICES = [
    'am_adam', 'am_michael', 'am_fenrir', 'am_puck',
    'am_echo', 'am_eric', 'am_liam', 'am_onyx'
]

# ── RVC (若叶睦/Airi) ────────────────────────────────────────────────
# RVC 音色转换跑在 Windows 侧 (RTX GPU), 由本服务自动拉起 Windows
# runtime python 进程 (rvc_work/rvc_serve.py), 通过 HTTP 调用。
# 基础语音 (说什么) 与 RVC 音色 (像谁) 分离:
#   zh: edge-tts 晓伊 (中文, 微软在线) ; en: Kokoro af_heart
RVC_VOICE_NAME = 'airi'          # /v1/audio/speech 里的 voice 名
RVC_MODEL = 'airi_e360.pth'      # assets/weights 下模型 (A/B 选定: e360)
RVC_BASE_LANG = 'zh'             # 'zh' 中文 (edge-tts) | 'en' 英语 (Kokoro)
RVC_BASE_VOICE_EN = 'af_heart'   # Kokoro 英语基础音色
EDGE_TTS_VOICE = 'zh-CN-XiaoyiNeural'   # 备用: 微软晓伊 (Windows 侧 edge-tts)
KOKORO_ZH_VOICE = os.environ.get('KOKORO_ZH_VOICE', 'zf_xiaoyi')  # 本地中文声线
ZH_PROVIDER = os.environ.get('ZH_PROVIDER', 'kokoro')  # kokoro | edge
# 本地 Kokoro 中文: 离线、无网络依赖; edge 作为兜底 (Windows 侧服务 /base_tts)
RVC_SERVICE_PORT = 8766
RVC_EXE = ('/mnt/e/download/RVC20260723Nvidia50x0/RVC20260718Nvidia50x0/'
           'runtime/python.exe')
RVC_SCRIPT = '/home/fengduan/kokoro-tts/rvc_work/rvc_serve.py'
RVC_LOG = '/home/fengduan/kokoro-tts/rvc_work/rvc_serve.log'
RVC_MODEL_FILE = '/home/fengduan/kokoro-tts/rvc_work/model.txt'   # 热切换模型名落盘
RVC_SERVICE_URL = None           # 探测后填充, 如 http://192.168.x.x:8766
_rvc_proc: subprocess.Popen = None
_rvc_start_lock = threading.Lock()   # 防并发启动出重复实例

# Global state
pipeline: KPipeline = None
pipeline_zh: KPipeline = None    # 本地中文 Kokoro (懒加载)


def init_models():
    """初始化 Kokoro 英文 pipeline。

    OpenVoice 已移除: 音色转换统一由 RVC (Windows GPU 服务) 提供。
    """
    global pipeline
    print(f'Device: {DEVICE}')
    pipeline = KPipeline(lang_code='a')
    print(f'Kokoro ready. {len(VOICES_US)} built-in voices; '
          f'音色转换由 RVC 提供。')


def warm_up_backend():
    """后台预热: Windows RVC 服务 + 本地中文 Kokoro + 完整链路各跑一次。

    把"首次请求要 30~40 秒"的冷启动开销挪到启动阶段。
    """
    start_rvc_service()
    if RVC_BASE_LANG == 'zh' and ZH_PROVIDER != 'edge':
        try:
            t0 = time.time()
            kokoro_zh_tts('预热。')
            print(f'[ZH] 中文 pipeline 预热完成 {time.time() - t0:.2f}s')
        except Exception as e:
            print(f'[ZH] 中文预热失败: {str(e)[:150]}')
    # 完整链路预热 (Kokoro -> RVC), 用一句稍长的真实文本
    try:
        t0 = time.time()
        audio = generate_rvc_tts('预热一下，今天天气不错。')
        print('[WARM] 完整链路预热完成 %.2fs (%.2fs 音频)'
              % (time.time() - t0, len(audio) / SAMPLE_RATE_KOKORO))
    except Exception as e:
        print(f'[WARM] 链路预热失败 (不影响使用): {str(e)[:150]}')


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_models()
    # 后台预热 (不阻塞启动)
    threading.Thread(target=warm_up_backend, daemon=True).start()
    # 后台体检: RVC 服务挂了/僵死会自动重启
    threading.Thread(target=rvc_health_monitor, daemon=True).start()
    yield


app = FastAPI(title='Kokoro TTS Server (High-Precision)', lifespan=lifespan)

# CORS: AIRI 等 Electron 客户端从渲染进程直连时需要预检通过
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*'],
    expose_headers=['*'],
)


@app.middleware('http')
async def log_requests(request, call_next):
    """所有请求带时间戳记日志, 便于排查客户端 (AIRI) 行为。"""
    t0 = time.time()
    response = await call_next(request)
    print('[HTTP %s] %s %s -> %d (%.2fs)'
          % (time.strftime('%H:%M:%S'), request.method, request.url.path,
             response.status_code, time.time() - t0))
    return response


# ── Audio Processing ────────────────────────────────────────────────

def generate_tts(text: str, voice: str, speed: float = 1.0) -> np.ndarray:
    """Generate TTS audio using Kokoro. Voice can be a blend."""
    result = pipeline(text, voice=voice, speed=speed)
    return np.concatenate([a.cpu().numpy() for _, _, a in result])


def save_audio_bytes(audio: np.ndarray, fmt: str = 'wav') -> bytes:
    """Convert numpy audio to the requested format bytes."""
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE_KOKORO, format=fmt.upper())
    buf.seek(0)
    return buf.read()


# ── RVC 音色转换 (Windows GPU 服务) ─────────────────────────────────

def find_windows_host_ip() -> str:
    """WSL2 里 Windows 宿主 IP = 默认网关地址。"""
    try:
        out = subprocess.check_output(['ip', 'route'], text=True)
        for line in out.splitlines():
            if line.startswith('default'):
                return line.split()[2]
    except Exception:
        pass
    try:
        with open('/etc/resolv.conf') as f:
            for line in f:
                if line.startswith('nameserver'):
                    ip = line.split()[1]
                    if ip != '127.0.0.53':
                        return ip
    except Exception:
        pass
    return None


def wsl_to_unc(path: str) -> str:
    """把 WSL 路径转成 Windows 可读的 \\\\wsl.localhost\\... 路径。"""
    try:
        out = subprocess.check_output(
            ['wslpath', '-w', path], text=True).strip()
        if out:
            return out
    except Exception:
        pass
    return '\\\\wsl.localhost\\Ubuntu' + path.replace('/', '\\')


_alive_cache = {'ts': 0.0, 'val': False}
_fail_streak = {'n': 0}          # 连续健康检查失败次数 (防误杀忙碌实例)


def rvc_service_alive(force: bool = False) -> bool:
    """带缓存的服务存活检查 (5 秒内复用结果, 避免每次请求都发 HTTP)。"""
    now = time.time()
    if not force and now - _alive_cache['ts'] < 5.0:
        return _alive_cache['val']
    val = False
    if RVC_SERVICE_URL:
        try:
            with urllib.request.urlopen(RVC_SERVICE_URL + '/health',
                                        timeout=2) as r:
                val = json.loads(r.read()).get('ok') is True
        except Exception:
            val = False
    _alive_cache['ts'] = now
    _alive_cache['val'] = val
    return val


def _rvc_port_owner_pid():
    """查 8766 端口占用进程 PID (Windows 侧)。"""
    try:
        out = subprocess.check_output(
            ['powershell.exe', '-NoProfile', '-Command',
             "(Get-NetTCPConnection -LocalPort %d -State Listen "
             "-ErrorAction SilentlyContinue | Select-Object -First 1).OwningProcess"
             % RVC_SERVICE_PORT],
            text=True, timeout=25, stderr=subprocess.DEVNULL)
        pid = out.strip().splitlines()[0].strip() if out.strip() else ''
        return pid if pid and pid != '0' else None
    except Exception:
        return None


def _rvc_orphan_pids():
    """找所有跑 rvc_serve.py 的 Windows python 进程 (含僵死/未监听的)。"""
    try:
        out = subprocess.check_output(
            ['powershell.exe', '-NoProfile', '-Command',
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -like '*rvc_serve.py*' } | "
             "ForEach-Object { $_.ProcessId }"],
            text=True, timeout=25, stderr=subprocess.DEVNULL)
        return [p.strip() for p in out.split() if p.strip().isdigit()]
    except Exception:
        return []


def force_restart_rvc_service(reason: str = ''):
    """强杀僵死的 RVC 服务实例 (端口占用者 + 所有 rvc_serve.py 孤儿进程)。"""
    pids = []
    owner = _rvc_port_owner_pid()
    if owner:
        pids.append(owner)
    for p in _rvc_orphan_pids():
        if p not in pids:
            pids.append(p)
    if not pids:
        print(f'[RVC] 自愈: 未发现残留实例 {reason}')
        return False
    for pid in pids:
        try:
            subprocess.run(['taskkill.exe', '/PID', pid, '/F'],
                           capture_output=True, timeout=25)
            print(f'[RVC] 自愈: 强杀僵死实例 PID {pid} {reason}')
        except Exception as e:
            print(f'[RVC] 强杀 {pid} 失败: {str(e)[:80]}')
    global _alive_cache
    _alive_cache = {'ts': 0.0, 'val': False}
    time.sleep(2)
    return True


def start_rvc_service() -> bool:
    """确保 Windows 侧 RVC 转换服务在运行 (锁内串行, 防重复实例)。

    若已有线程正在启动, 本次调用会排队等待它完成 (最长 90 秒), 而不是立即失败:
    RVC 冷启动需 30~40 秒, 期间的请求应当等待, 不该报 503/502。
    """
    if _rvc_start_lock.locked():
        print('[RVC] 服务启动中, 本次请求排队等待 ...', flush=True)
        for _ in range(45):
            time.sleep(2)
            if rvc_service_alive(force=True):
                return True
        print('[RVC] 等待启动超时 (90s)', flush=True)
        return False
    with _rvc_start_lock:
        return _start_rvc_service()


def _start_rvc_service() -> bool:
    """(锁内) 探测/拉起 Windows RVC 转换服务 (含僵死实例自愈)。"""
    global RVC_SERVICE_URL, _rvc_proc
    # 仅在首次初始化时写默认模型名; 之后由请求参数 / 前端设置决定
    # (否则每次请求都会把用户切换的模型覆盖回默认值)
    if not os.path.exists(RVC_MODEL_FILE):
        try:
            Path(RVC_MODEL_FILE).write_text(RVC_MODEL)
        except Exception as e:
            print(f'[RVC] 写模型配置失败: {e}')
    if rvc_service_alive():
        return True

    host = find_windows_host_ip()
    if not host:
        print('[RVC] 无法定位 Windows 宿主 IP')
        return False
    url = f'http://{host}:{RVC_SERVICE_PORT}'

    # 健康检查失败但端口被占 / 存在孤儿进程 → 判定为僵死, 先强杀再重启
    if _rvc_port_owner_pid() or _rvc_orphan_pids():
        _fail_streak['n'] += 1
        print('[RVC] 健康检查失败第 %d 次 (需连续 3 次才判定僵死)' % _fail_streak['n'],
              flush=True)
        if _fail_streak['n'] >= 3:
            _fail_streak['n'] = 0
            force_restart_rvc_service('(连续 3 次健康检查失败)')
        else:
            time.sleep(5)
    else:
        _fail_streak['n'] = 0

    # 端口上已有服务但不是我们认的? 先直接探测一次
    try:
        with urllib.request.urlopen(url + '/health', timeout=2) as r:
            if json.loads(r.read()).get('ok'):
                RVC_SERVICE_URL = url
                return True
    except Exception:
        pass

    if not os.path.exists(RVC_EXE):
        print(f'[RVC] 找不到 {RVC_EXE}')
        return False

    print(f'[RVC] 启动 Windows 转换服务: {RVC_EXE}')
    logf = open(RVC_LOG, 'a')
    try:
        _rvc_proc = subprocess.Popen(
            [RVC_EXE, '-I', RVC_SCRIPT],
            stdout=logf, stderr=logf)
    except Exception as e:
        print(f'[RVC] 启动失败: {e}')
        return False

    # 首次启动含模型+hubert 加载 + 预热, 最多等 60 秒
    for _ in range(30):
        time.sleep(2)
        try:
            with urllib.request.urlopen(url + '/health', timeout=2) as r:
                if json.loads(r.read()).get('ok'):
                    RVC_SERVICE_URL = url
                    _alive_cache['ts'] = time.time()
                    _alive_cache['val'] = True
                    _fail_streak['n'] = 0
                    _m = ''
                    try:
                        _m = Path(RVC_MODEL_FILE).read_text(encoding='utf-8').strip()
                    except Exception:
                        pass
                    print(f'[RVC] 服务就绪: {RVC_SERVICE_URL} (模型 {_m or RVC_MODEL})')
                    return True
        except Exception:
            pass
    print(f'[RVC] 等待服务超时, 强杀残留实例后重试一次 (日志: {RVC_LOG})')
    force_restart_rvc_service('(启动超时)')
    try:
        _rvc_proc = subprocess.Popen(
            [RVC_EXE, '-I', RVC_SCRIPT], stdout=logf, stderr=logf)
    except Exception as e:
        print(f'[RVC] 二次启动失败: {e}')
        return False
    for _ in range(30):
        time.sleep(2)
        try:
            with urllib.request.urlopen(url + '/health', timeout=2) as r:
                if json.loads(r.read()).get('ok'):
                    RVC_SERVICE_URL = url
                    _alive_cache['ts'] = time.time()
                    _alive_cache['val'] = True
                    _fail_streak['n'] = 0
                    print(f'[RVC] 服务就绪(自愈成功): {RVC_SERVICE_URL}')
                    return True
        except Exception:
            pass
    print('[RVC] 二次等待仍超时, 放弃本次')
    return False


def rvc_health_monitor():
    """后台体检: 每 60 秒确认 RVC 服务存活, 挂了/僵死自动重启 (自愈)。"""
    while True:
        time.sleep(60)
        try:
            if not rvc_service_alive(force=True):
                print('[RVC] 体检发现服务不可用, 尝试自愈 ...')
                start_rvc_service()
        except Exception as e:
            print(f'[RVC] 体检异常: {str(e)[:120]}')


def edge_tts_zh(text: str, speed: float = 1.0) -> np.ndarray:
    """备用中文方案: 调 Windows RVC 服务的 /base_tts (edge-tts, 24kHz wav)。"""
    if not start_rvc_service():
        raise RuntimeError('RVC 服务不可用 (edge 备用方案需要它)')
    rate = f'{int(round((speed - 1.0) * 100)):+d}%'
    req = urllib.request.Request(
        RVC_SERVICE_URL + '/base_tts',
        data=json.dumps({'text': text, 'rate': rate}).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=90) as r:
        data = r.read()
    audio, sr = sf.read(io.BytesIO(data))
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE_KOKORO:
        audio = librosa.resample(
            y=audio, orig_sr=sr, target_sr=SAMPLE_RATE_KOKORO)
    return audio


def kokoro_zh_tts(text: str, speed: float = 1.0) -> np.ndarray:
    """本地 Kokoro 中文合成 (离线, CPU 约 1.2s/句)。"""
    global pipeline_zh
    if pipeline_zh is None:
        print('[ZH] 加载本地 Kokoro 中文 pipeline ...')
        pipeline_zh = KPipeline(lang_code='z')
        print(f'[ZH] 就绪, 声线 {KOKORO_ZH_VOICE}')
    chunks = [a.cpu().numpy()
              for _, _, a in pipeline_zh(text, voice=KOKORO_ZH_VOICE,
                                         speed=speed)]
    if not chunks:
        raise RuntimeError('Kokoro 中文合成为空')
    audio = np.concatenate(chunks).astype(np.float32)
    if len(audio) / SAMPLE_RATE_KOKORO < 0.15:
        raise RuntimeError('Kokoro 中文合成结果过短')
    return audio


def generate_rvc_base(text: str, speed: float = 1.0,
                      lang: str = RVC_BASE_LANG) -> np.ndarray:
    """按语言生成 RVC 的基础语音 (内容层)。

    中文: 本地 Kokoro 优先 (离线快), 失败自动回退 Windows edge-tts。
    """
    if lang == 'zh':
        if ZH_PROVIDER == 'edge':
            return edge_tts_zh(text, speed=speed)
        try:
            return kokoro_zh_tts(text, speed=speed)
        except Exception as e:
            print(f'[ZH] 本地 Kokoro 失败, 回退 edge-tts: {str(e)[:120]}')
            return edge_tts_zh(text, speed=speed)
    return generate_tts(text, RVC_BASE_VOICE_EN, speed=speed)


def generate_rvc_tts(text: str, speed: float = 1.0,
                     lang: str = RVC_BASE_LANG,
                     index_rate: float = 0.0) -> np.ndarray:
    """基础语音 → Windows RVC 服务转成目标音色 (字节直传, 24 kHz 返回)。

    index_rate: 索引检索倍率 0~1 (压电音用), 透传给 RVC 服务。
    """
    t0 = time.time()
    if not start_rvc_service():
        raise RuntimeError('RVC 服务不可用')

    base_audio = generate_rvc_base(text, speed=speed, lang=lang)
    t1 = time.time()
    buf = io.BytesIO()
    sf.write(buf, base_audio, SAMPLE_RATE_KOKORO, format='WAV')
    req = urllib.request.Request(
        RVC_SERVICE_URL + '/convert_bytes',
        data=buf.getvalue(),
        headers={'Content-Type': 'audio/wav',
                 'X-Index-Rate': str(float(index_rate or 0.0))})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    t2 = time.time()

    audio, sr = sf.read(io.BytesIO(data))
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE_KOKORO:
        audio = librosa.resample(
            y=audio, orig_sr=sr, target_sr=SAMPLE_RATE_KOKORO)
    print('[TIMING] 基础语音 %.2fs + RVC %.2fs + 包装 %.2fs = %.2fs'
          % (t1 - t0, t2 - t1, time.time() - t2, time.time() - t0))
    return audio


def rvc_lang_for_voice(voice: str) -> str:
    """voice 名 → 基础语言: 'airi'/'airi_zh'→zh, 'airi_en'→en。"""
    if voice == 'airi_en':
        return 'en'
    return RVC_BASE_LANG


# ── API Endpoints ──────────────────────────────────────────────────

MODEL_IDS = ['tts-1', 'tts-1-hd', 'kokoro', 'airi', 'tts']


def models_payload():
    """OpenAI 兼容的模型列表 (AIRI 校验 provider 时会调)。"""
    return {
        'object': 'list',
        'data': [
            {'id': i, 'object': 'model', 'created': 0, 'owned_by': 'local'}
            for i in MODEL_IDS
        ],
    }


def voices_payload():
    """音色列表 (兼容不同客户端的探测路径)。"""
    rvc = {
        voice_name: {'model': RVC_MODEL,
                     'lang': lang,
                     'service': RVC_SERVICE_URL or 'starting'}
        for voice_name, lang in (('airi', RVC_BASE_LANG),
                                 ('airi_zh', 'zh'), ('airi_en', 'en'))
    }
    all_voices = sorted(set(VOICES_US) | set(rvc))
    return {
        'object': 'list',
        'data': [{'id': v, 'name': v, 'object': 'voice',
                  'provider': 'local'} for v in all_voices],
        'builtin': VOICES_US,
        'rvc': rvc,
    }


@app.get('/v1/models')
@app.get('/audio/models')
@app.get('/v1/audio/models')
async def list_models():
    return models_payload()


@app.get('/v1/voices')
@app.get('/audio/voices')
@app.get('/v1/audio/voices')
async def list_voices():
    """List all available voices (built-in + cloned + RVC)。"""
    return voices_payload()


@app.get('/v1/audio/speech')
@app.get('/audio/speech')
async def speech_probe():
    """AIRI 校验 baseUrl 时会 GET 该路径, 返回 200 表示服务可用。"""
    return {
        'status': 'ok',
        'service': 'kokoro-rvc-tts',
        'method': 'POST with {"input": ..., "voice": ...}',
        'voices': ['airi', 'airi_zh', 'airi_en'] + VOICES_US[:3],
    }


@app.post('/v1/audio/speech')
@app.post('/audio/speech')
async def audio_speech(request: dict):
    """OpenAI-compatible audio/speech endpoint (同时兼容无 /v1 前缀)。

    Fields:
        model: Model name (ignored, 本地模型)
        input: Text to synthesize
        voice: 'airi' (RVC 若叶睦音色) / 内置 ('af_heart') / 克隆音色
        speed: Playback speed multiplier (0.5 - 2.0, default 1.0)
        response_format: 'wav' (默认) 或 'mp3'
    """
    text = request.get('input', '')
    voice = request.get('voice', 'af_heart')
    response_format = request.get('response_format', 'wav')
    speed = float(request.get('speed', 1.0))

    if not text:
        raise HTTPException(400, 'No input text provided')

    t_req = time.time()
    print('[REQ %s] voice=%s len=%d text=%s'
          % (time.strftime('%H:%M:%S'), voice, len(text), text[:26]))

    # 可选: 热切换 RVC 音色模型 + 索引倍率 (供自建网页调参用)
    req_model = (request.get('rvc_model') or '').strip()
    if req_model and req_model != RVC_MODEL:
        try:
            Path(RVC_MODEL_FILE).write_text(req_model, encoding='utf-8')
            print(f'[RVC] 模型热切换 -> {req_model}')
        except Exception as e:
            print(f'[RVC] 模型切换失败: {e}')
    req_index_rate = float(request.get('rvc_index_rate', 0) or 0)

    # Determine if voice is cloned or built-in
    if voice == RVC_VOICE_NAME or voice in ('airi_zh', 'airi_en'):
        # RVC 音色: 基础语音(本地Kokoro中文/回退edge) → Windows GPU 转换
        # 放到线程池执行, 避免阻塞事件循环 → 多句可并行流水线
        try:
            import anyio
            audio = await anyio.to_thread.run_sync(
                lambda: generate_rvc_tts(
                    text, speed=speed, lang=rvc_lang_for_voice(voice),
                    index_rate=req_index_rate))
        except Exception as e:
            print(f'[RVC] 转换失败: {e}')
            raise HTTPException(503, f'RVC service unavailable: {e}')
    elif voice in VOICES_US or ',' in voice:
        audio = generate_tts(text, voice, speed=speed)
    else:
        available = VOICES_US + [RVC_VOICE_NAME]
        raise HTTPException(
            400,
            f'Unknown voice "{voice}". Available: {available}'
        )

    audio_bytes = save_audio_bytes(audio, response_format)
    print('[DONE %s] %.2fs (%.2fs 音频)'
          % (time.strftime('%H:%M:%S'), time.time() - t_req,
             len(audio) / SAMPLE_RATE_KOKORO))
    return Response(
        content=audio_bytes,
        media_type=f'audio/{response_format}',
        headers={'Content-Disposition': 'inline'}
    )


@app.get('/health')
async def health():
    return {
        'status': 'ok',
        'device': DEVICE,
        'builtin_voices': len(VOICES_US),
        'rvc': {
            'voices': ['airi', 'airi_zh', 'airi_en'],
            'base_lang': RVC_BASE_LANG,
            'zh_provider': 'edge-tts/' + EDGE_TTS_VOICE,
            'model': RVC_MODEL,
            'service_url': RVC_SERVICE_URL,
            'alive': rvc_service_alive(),
        },
        'pipelines': [
            'local_kokoro_zh (离线中文)',
            'kokoro_en (英文)',
            'rvc_timbre_conversion (Windows GPU, 热切换 + LRU 缓存)',
        ],
    }


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--host', default='0.0.0.0')
    args = ap.parse_args()

    print(f'Starting enhanced Kokoro TTS server on '
          f'http://{args.host}:{args.port}')
    print(f'API docs: http://localhost:{args.port}/docs')
    print(f'Enhancements: sample-rate correction, multi-segment SE, '
          f'F0 profile, auto base-voice')
    uvicorn.run(app, host=args.host, port=args.port, log_level='info')
