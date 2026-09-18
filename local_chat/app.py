#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本地聊天管线: LLM(OpenAI兼容) -> 本地 Kokoro 中文 TTS -> RVC(音色) -> 网页播放。

启动: venv/bin/python local_chat/app.py --port 8770
依赖: 已运行的 tts_server (8765) + Windows RVC 服务 (8766, 自动拉起)

设计要点 (v2):
  * 所有阻塞调用 (LLM 流式 / TTS 合成) 都放到线程执行, 绝不阻塞事件循环
    → 合成繁忙时网页依然秒开
  * 单次合成/请求带超时, 失败快速返回错误事件, 不会把服务拖死
  * 并发上限: 最多 N 个合成同时跑 (防 CPU/GPU 抢占)
  * 记忆: 人设 + 时间感知 + 关系状态 + 事实记忆 + 剧情摘要, 后台线程异步抽取

历史: 本文件在 2026-09-17 22:24 被意外清空 (0 字节)。当前内容由
      __pycache__/app.cpython-312.pyc 的字节码反汇编逐函数还原 (提示词原文、
      阈值、默认配置、SSE 事件字段均与原版一致), 之后可正常维护/修改。
"""
import argparse
import asyncio
import base64
import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import anyio
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

import memory as mem

HERE = Path(__file__).parent
ROOT = HERE.parent
WORK = ROOT / 'rvc_work'
CFG_FILE = HERE / 'config.json'
MODEL_TXT = WORK / 'model.txt'
TTS_URL = os.environ.get('TTS_URL', 'http://localhost:8765')
WEIGHTS = Path('/mnt/e/download/RVC20260723Nvidia50x0/RVC20260718Nvidia50x0/assets/weights')

TTS_TIMEOUT = float(os.environ.get('CHAT_TTS_TIMEOUT', '75'))    # 单句合成超时(秒)
LLM_TIMEOUT = float(os.environ.get('CHAT_LLM_TIMEOUT', '60'))    # LLM 请求超时(秒)
MAX_CONCURRENT_TTS = int(os.environ.get('CHAT_TTS_CONCURRENCY', '2'))

WEEKDAY = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']

DEFAULT_CFG = {
    "llm": {
        "base_url": "",
        "api_key": "",
        "model": "",
        "system_prompt": "你是艾莉，一个温柔、活泼、有点黏人的少女。用口语化的中文回答，每次回复不超过两句话。",
        "temperature": 0.85,
        "max_tokens": 300,
    },
    "tts": {
        "voice": "airi",
        "speed": 1.0,
        "rvc_model": "airi_e360.pth",
        "rvc_index_rate": 0.0,
    },
    "history_len": 8,
    "memory": {
        "enabled": True,          # 记忆总开关 (关掉=纯聊天, 不读写数据库)
        "auto_extract": True,     # 每轮后台抽取事实 + 更新关系
        "inject_max_chars": 700,  # 注入 prompt 的记忆字符上限
        "session_gap_hours": 2,   # 超过这么久没聊 → 自动开新会话
        "summary_after": 24,      # 会话消息数达到多少条开始做剧情摘要
        "inject_time": True,      # 注入当前时间 / 距上次对话
        "model": "deepseek-chat", # 记忆专用模型 (别用推理模型: token 会全花在思考上)
        "relationship": True,     # 注入关系状态 (好感/信任/心情)
        "mood_swing": 1.0,        # 情绪波动倍率 (调小=情绪更稳定)
    },
    # 关闭网页后自动停止后台服务; 刷新不会触发 (宽限期内新页面会发心跳取消)
    "auto_stop": True,
    "auto_stop_grace": 20,
}

# ── 记忆抽取提示词 (一次调用同时返回 facts + relationship) ──────────
EXTRACT_PROMPT = '''你是「记忆 + 情绪」分析器。看下面这轮对话, 只输出一个 JSON 对象, 不要解释、不要代码块标记。
{
  "facts": [{"key":"稳定的短键名","value":"值","category":"身份|偏好|关系|计划|禁忌|梗","confidence":0.9,"evidence":"对话原文片段"}],
  "relationship": {"affinity_delta": 1.5, "trust_delta": 0.5, "mood": "开心",
                   "mood_intensity": 0.7, "milestone": "", "note": ""}
}
规则:
- facts: 只记用户本人的稳定信息(称呼/喜好/厌恶/身份/在学什么/重要计划/约定/雷点); 寒暄闲聊情绪波动不要记; 没有就 []
- key 要稳定可复用 (同一含义用同一个 key), 必须带 evidence 原文片段
- affinity_delta: 这轮互动让"艾莉对用户的好感"变化多少, -5~+5 (被关心/夸赞/分享心事→正; 被冷落/敷衍/冒犯→负)
- trust_delta: 信任变化 -3~+3 (用户透露私密/依赖你→正)
- mood: 艾莉此刻心情, 从 开心/兴奋/害羞/平静/温柔/低落/寂寞/生气/担心/撒娇 里选一个
- mood_intensity: 0~1
- milestone: 若这轮发生值得纪念的事(第一次互报名字/约定某件事/吵架又和好), 写一句话, 否则空串
- note: 可选, 艾莉对自己状态的一句内心话, 否则空串
对话:
用户: %s
艾莉: %s'''

SUMMARY_PROMPT = '''把下面的对话压缩成不超过 150 字的中文剧情摘要, 保留对后续聊天有用的事实与进展。
若已有旧摘要, 请把新内容合并进去 (不要重复)。只输出摘要正文。
旧摘要: %s
对话:
%s'''

app = FastAPI(title='Local Chat (LLM + Kokoro + RVC)')

ASSETS = HERE / 'assets'
ASSETS.mkdir(exist_ok=True)
app.mount('/assets', StaticFiles(directory=str(ASSETS)), name='assets')

_tts_sem = threading.BoundedSemaphore(MAX_CONCURRENT_TTS)

# 页面活跃状态: 用于"关闭网页 => 自动停服务"
_activity = {'last_beat': 0.0, 'bye_at': None}
_activity_lock = threading.Lock()


# ── 配置 ────────────────────────────────────────────────────────────

def load_cfg():
    if CFG_FILE.exists():
        try:
            cfg = json.loads(CFG_FILE.read_text(encoding='utf-8'))
            for k, v in DEFAULT_CFG.items():
                if isinstance(v, dict):
                    cfg.setdefault(k, {})
                    for kk, vv in v.items():
                        cfg[k].setdefault(kk, vv)
                else:
                    cfg.setdefault(k, v)
            return cfg
        except Exception:
            pass
    return json.loads(json.dumps(DEFAULT_CFG))


def save_cfg(cfg):
    CFG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    apply_rvc_model(cfg['tts'].get('rvc_model'))


def apply_rvc_model(name):
    """写 model.txt -> Windows RVC 服务下次请求自动热切换模型。"""
    if not name:
        return
    try:
        MODEL_TXT.write_text(name, encoding='utf-8')
    except Exception as e:
        print('[MODEL] 切换失败:', e)


def list_rvc_models():
    if not WEIGHTS.is_dir():
        return []
    skip = ('airi_e180.pth', 'airi_e540.pth', 'airi_e720.pth', 'airi_e900.pth')
    out = []
    for f in sorted(WEIGHTS.glob('*.pth')):
        if f.name in skip:
            continue
        out.append({'file': f.name, 'size_mb': round(f.stat().st_size / 1048576, 1)})
    return out


# ── TTS (阻塞调用, 必须在线程里跑) ──────────────────────────────────

def _synthesize_blocking(text, voice=None, speed=None, index_rate=None):
    cfg = load_cfg()['tts']
    payload = {
        'input': text,
        'voice': voice or cfg.get('voice', 'airi'),
        'speed': float(speed if speed is not None else cfg.get('speed', 1.0)),
        'rvc_model': cfg.get('rvc_model'),
        'rvc_index_rate': float(index_rate if index_rate is not None
                                else cfg.get('rvc_index_rate', 0.0)),
    }
    req = urllib.request.Request(
        TTS_URL + '/v1/audio/speech',
        data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=TTS_TIMEOUT) as r:
        return r.read()


async def synthesize_async(text, voice=None, speed=None, index_rate=None):
    """在线程池里合成 (带并发上限), 不阻塞事件循环。"""
    def job():
        with _tts_sem:
            return _synthesize_blocking(text, voice=voice, speed=speed,
                                        index_rate=index_rate)
    return await anyio.to_thread.run_sync(job)


# ── 分句 ────────────────────────────────────────────────────────────

SENT_END = '。！？!?；;\n'
MAX_SENT = 42
FIRST_MAX_SENT = int(os.environ.get('CHAT_FIRST_SENT_MAX', '18'))   # 首句更短 → 更快出声


def split_sentences(buf, flush=False, first=False):
    """从缓冲区切出完整句子, 返回 (句子列表, 剩余缓冲)。

    first=True 时用更短的阈值 (首句优先出声)。
    """
    limit = FIRST_MAX_SENT if first else MAX_SENT
    out = []
    while True:
        m = re.search('[' + re.escape(SENT_END) + ']', buf)
        if m and m.end() >= 4:
            out.append(buf[:m.end()].strip())
            buf = buf[m.end():]
            continue
        if len(buf) >= limit:
            cut = re.search(r'[，,、\s]', buf[limit // 2:limit])
            idx = (limit // 2 + cut.start() + 1) if cut else limit
            out.append(buf[:idx].strip())
            buf = buf[idx:]
            continue
        break
    if flush and buf.strip():
        out.append(buf.strip())
        buf = ''
    return [s for s in out if s], buf


# ── LLM ─────────────────────────────────────────────────────────────

def llm_stream(messages, cfg):
    """OpenAI 兼容流式对话 (阻塞生成器, 由线程消费); 未配置 key 时演示模式。"""
    llm = cfg['llm']
    if not llm.get('base_url') or not llm.get('model'):
        demo = ('（演示模式）还没有配置 LLM，请在左侧填入 Base URL、API Key 和模型名。'
                '现在你听到的声音，是本地合成再经过音色转换的结果。')
        for ch in re.findall(r'.{1,6}', demo):
            yield ch
            time.sleep(0.02)
        return
    url = llm['base_url'].rstrip('/') + '/chat/completions'
    body = json.dumps({
        'model': llm['model'],
        'messages': messages,
        'stream': True,
        'temperature': float(llm.get('temperature', 0.85)),
        'max_tokens': int(llm.get('max_tokens', 300)),
    }, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(
        url, data=body,
        headers={'Content-Type': 'application/json',
                 'Authorization': 'Bearer ' + (llm.get('api_key') or 'sk-none')})
    try:
        with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as r:
            for raw in r:
                line = raw.decode('utf-8', 'replace').strip()
                if not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    break
                try:
                    obj = json.loads(data)
                    delta = (obj.get('choices') or [{}])[0].get('delta', {})
                    piece = delta.get('content') or ''
                except Exception:
                    piece = ''
                if piece:
                    yield piece
    except urllib.error.HTTPError as e:
        detail = e.read().decode('utf-8', 'replace')[:300]
        yield f'（LLM 请求失败 HTTP {e.code}: {detail}）'
    except Exception as e:
        yield f'（LLM 连接失败: {str(e)[:200]}）'


def _pump_llm(messages, cfg, queue, loop):
    """在线程里跑阻塞的 LLM 流, 把片段投递到 asyncio 队列。"""
    try:
        for piece in llm_stream(messages, cfg):
            loop.call_soon_threadsafe(queue.put_nowait, piece)
    except Exception as e:
        loop.call_soon_threadsafe(queue.put_nowait, f'（LLM 异常: {str(e)[:150]}）')
    finally:
        loop.call_soon_threadsafe(queue.put_nowait, None)   # 结束哨兵


def _llm_once(cfg, messages, max_tokens=400, temperature=0.0, timeout=45, model=None):
    """一次性(非流式)调用 LLM, 返回纯文本 (best-effort)。

    model 为空时用 config.llm.model; 记忆类调用应传 memory.model, 避免命中
    推理模型 (推理模型会把 token 全花在 reasoning_content 上, content 为空)。
    content 为空时尝试从 reasoning_content 里抠出 JSON。
    """
    llm = cfg['llm']
    use_model = model or llm.get('model') or ''
    url = (llm.get('base_url') or '').rstrip('/') + '/chat/completions'
    body = json.dumps({
        'model': use_model,
        'messages': messages,
        'stream': False,
        'temperature': float(temperature),
        'max_tokens': int(max_tokens),
    }, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(
        url, data=body,
        headers={'Content-Type': 'application/json',
                 'Authorization': 'Bearer ' + (llm.get('api_key') or 'sk-none')})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            obj = json.loads(r.read().decode('utf-8', 'replace'))
        ch = (obj.get('choices') or [{}])[0]
        msg = ch.get('message') or {}
        text = (msg.get('content') or '').strip()
        reason = (msg.get('reasoning_content') or '').strip()
        if not text and reason:
            for pat in (r'\{[^{}]*"facts"[\s\S]*\}\s*$', r'\{[\s\S]*"relationship"[\s\S]*\}'):
                m = re.search(pat, reason)
                if m:
                    return m.group(0)
            text = reason
        if not text and ch.get('finish_reason') == 'length':
            print('[MEM] 模型 %s 输出被思考耗尽 (finish=length), 建议换非推理模型'
                  % use_model, flush=True)
        return text
    except Exception as e:
        print('[MEM] LLM 调用失败: %s' % str(e)[:120], flush=True)
        return ''


# ── 记忆: prompt 组装 + 后台抽取 ────────────────────────────────────

def build_system_prompt(cfg, session_id, user_text):
    """人设 + 时间感知 + 会话摘要 + 相关事实记忆。返回 (prompt, 命中列表)。"""
    parts = [cfg['llm']['system_prompt']]
    mcfg = cfg.get('memory', {})
    if not mcfg.get('enabled', True):
        return parts[0], []

    if mcfg.get('inject_time', True):
        now = time.localtime()
        tline = time.strftime('%Y-%m-%d %H:%M', now) + ' ' + WEEKDAY[now.tm_wday]
        ctx = ['[时间] 现在是 %s' % tline]
        last = mem.last_message_ts(session_id)
        if last:
            gap_h = (time.time() - last) / 3600.0
            if gap_h > 0.05:
                ctx.append('[距上次对话] %.1f 小时前' % gap_h if gap_h < 48
                           else '[距上次对话] %.1f 天前' % (gap_h / 24))
        parts.append('\n'.join(ctx))

    sess = mem.get_session(session_id) or {}
    if sess.get('summary'):
        parts.append('[此前剧情摘要]\n' + sess['summary'][:600])

    if mcfg.get('relationship', True):
        rel_block = mem.rel_for_prompt(session_id)
        if rel_block:
            parts.append(rel_block)

    block, hits = mem.facts_for_prompt(user_text, mcfg.get('inject_max_chars', 700))
    if block:
        parts.append('[关于用户的长期记忆]\n' + block +
                     '\n(自然运用这些记忆, 不要生硬复述, 也不要说"根据我的记忆")')

    return '\n\n'.join(parts), hits


def extract_facts_bg(cfg, session_id, user_text, reply_text, source_msg_id=None):
    """后台线程: 抽取事实 + 更新关系状态 + 维护会话摘要 (不阻塞回复)。

    整个流程只用一次 LLM 调用 (EXTRACT_PROMPT 一次返回 facts + relationship),
    因此不会增加回复延迟。
    """
    try:
        raw = _llm_once(
            cfg,
            [{'role': 'user',
              'content': EXTRACT_PROMPT % (user_text[:600], reply_text[:600])}],
            max_tokens=900,
            model=(cfg.get('memory', {}).get('model') or None))
        raw = raw.strip()
        if raw.startswith('```'):
            raw = raw.strip('`')
            if raw.lower().startswith('json'):
                raw = raw.split('\n', 1)[-1]

        facts, rel, obj = [], None, None
        m = re.search(r'\{.*\}', raw, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None
        if isinstance(obj, dict):
            facts = obj.get('facts') or []
            rel = obj.get('relationship') or None
        else:
            m2 = re.search(r'\[.*\]', raw, re.S)
            if m2:
                try:
                    facts = json.loads(m2.group(0))
                except Exception:
                    facts = []

        for item in (facts or [])[:8]:
            if not isinstance(item, dict):
                continue
            k = str(item.get('key', ''))[:60]
            v = str(item.get('value', ''))[:300]
            if not k or not v:
                continue
            cat = item.get('category') or '其他'
            if cat not in mem.CATEGORIES:
                cat = '其他'
            mem.upsert_fact(k, v, cat,
                            float(item.get('confidence', 0.8) or 0.8),
                            str(item.get('evidence', ''))[:200],
                            source_msg_id)
            print(f'[MEM] 记住: {k} = {v} ({cat})', flush=True)

        if isinstance(rel, dict) and cfg.get('memory', {}).get('relationship', True):
            swing = float(cfg.get('memory', {}).get('mood_swing', 1.0) or 1.0)
            r = mem.update_relationship(
                affinity_delta=float(rel.get('affinity_delta', 0) or 0) * swing,
                trust_delta=float(rel.get('trust_delta', 0) or 0) * swing,
                mood=(str(rel.get('mood'))[:20] if rel.get('mood') else None),
                mood_intensity=(float(rel['mood_intensity'])
                                if rel.get('mood_intensity') is not None else None),
                milestone=(str(rel.get('milestone'))[:80] if rel.get('milestone') else None),
                note=(str(rel.get('note'))[:200] if rel.get('note') is not None else None))
            print('[REL] 好感 %.1f 信任 %.1f 心情 %s%s' % (
                r['affinity'], r['trust'], r['mood'],
                (' | 纪念: %s' % r['milestones'][-1]['text']) if r.get('milestones') else ''),
                flush=True)
    except Exception as e:
        print('[MEM] 抽取失败(忽略): %s' % str(e)[:140], flush=True)

    # ── 会话摘要维护 (消息够多时把旧对话压缩进 summary) ──
    try:
        need = int(cfg.get('memory', {}).get('summary_after', 24) or 24)
        if mem.message_count(session_id) >= need:
            older = mem.unsummarized_before(session_id, keep_recent=12)
            if older:
                txt = '\n'.join(
                    f"{'用户' if m['role'] == 'user' else '艾莉'}: {str(m['content'])[:200]}"
                    for m in older)
                old_sum = (mem.get_session(session_id) or {}).get('summary', '')
                new_sum = _llm_once(
                    cfg,
                    [{'role': 'user',
                      'content': SUMMARY_PROMPT % (old_sum[:400], txt[:3000])}],
                    max_tokens=500,
                    model=(cfg.get('memory', {}).get('model') or None))
                if new_sum.strip():
                    mem.set_summary(session_id, new_sum.strip()[:600])
                    mem.mark_summarized([m['id'] for m in older])
                    print('[MEM] 会话摘要已更新 (%d 条并入)' % len(older), flush=True)
    except Exception as e:
        print('[MEM] 摘要维护失败(忽略): %s' % str(e)[:140], flush=True)


# ── HTTP 接口 ───────────────────────────────────────────────────────

@app.get('/')
async def index():
    return HTMLResponse((HERE / 'index.html').read_text(encoding='utf-8'))


@app.post('/api/heartbeat')
async def heartbeat():
    """网页每 5 秒上报一次: 只要有心跳就不会触发自动停止。"""
    with _activity_lock:
        _activity['last_beat'] = time.time()
        _activity['bye_at'] = None          # 页面回来了 (刷新/新开) → 取消关闭
    return {'ok': True,
            'auto_stop': bool(load_cfg().get('auto_stop', True)),
            'grace': float(load_cfg().get('auto_stop_grace', 20))}


@app.post('/api/bye')
async def bye():
    """网页关闭时 (sendBeacon) 上报; 宽限期内没有新心跳才真正停止服务。"""
    with _activity_lock:
        _activity['bye_at'] = time.time()
    return {'ok': True}


def _autostop_watchdog():
    """后台线程: 页面关闭且超过宽限期 → 调 stop_all.sh 停掉所有服务。"""
    while True:
        time.sleep(2)
        try:
            cfg = load_cfg()
            if not cfg.get('auto_stop', True):
                continue
            grace = float(cfg.get('auto_stop_grace', 20))
            with _activity_lock:
                bye = _activity['bye_at']
                beat = _activity['last_beat']
            now = time.time()
            if bye and (now - bye) > grace and (now - beat) > grace:
                print('[AUTOSTOP] 网页已关闭 %.0f 秒, 停止后台服务' % (now - bye),
                      flush=True)
                subprocess.Popen(['bash', str(HERE / 'stop_all.sh')],
                                 start_new_session=True,
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
                time.sleep(4)
                os._exit(0)
        except Exception as e:
            print('[AUTOSTOP] 检查异常: %s' % str(e)[:150], flush=True)


@app.get('/api/config')
async def get_config():
    return {'config': load_cfg(), 'models': list_rvc_models()}


@app.post('/api/config')
async def set_config(request: Request):
    patch = await request.json()
    cfg = load_cfg()
    for sec in ('llm', 'tts', 'memory'):
        if sec in patch:
            cfg[sec].update(patch[sec])
    if 'history_len' in patch:
        cfg['history_len'] = int(patch['history_len'])
    if 'auto_stop' in patch:
        cfg['auto_stop'] = bool(patch['auto_stop'])
    if 'auto_stop_grace' in patch:
        cfg['auto_stop_grace'] = max(3, int(patch['auto_stop_grace']))
    await anyio.to_thread.run_sync(save_cfg, cfg)
    return {'ok': True, 'config': cfg}


@app.post('/api/model')
async def set_model(request: Request):
    body = await request.json()
    name = body.get('name')
    cfg = load_cfg()
    cfg['tts']['rvc_model'] = name
    await anyio.to_thread.run_sync(save_cfg, cfg)
    return {'ok': True, 'model': name}


@app.post('/api/tts')
async def tts_test(request: Request):
    body = await request.json()
    text = (body.get('text') or '你好，我是艾莉。').strip()
    t0 = time.time()
    try:
        wav = await asyncio.wait_for(
            synthesize_async(text, voice=body.get('voice'),
                             speed=body.get('speed'),
                             index_rate=body.get('rvc_index_rate')),
            timeout=TTS_TIMEOUT + 10)
    except asyncio.TimeoutError:
        return JSONResponse({'error': '合成超时（服务可能正在切换模型或预热）'},
                            status_code=504)
    except Exception as e:
        return JSONResponse({'error': str(e)[:300]}, status_code=502)
    print('[TTS] %s | %.2fs | %d bytes' % (text[:20], time.time() - t0, len(wav)),
          flush=True)
    return Response(content=wav, media_type='audio/wav')


# ── 会话 ────────────────────────────────────────────────────────────

@app.get('/api/sessions')
async def api_sessions():
    return {'sessions': await anyio.to_thread.run_sync(mem.list_sessions)}


@app.post('/api/session/new')
async def api_session_new(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    sid = await anyio.to_thread.run_sync(
        lambda: mem.new_session(body.get('title') or ''))
    meet = await anyio.to_thread.run_sync(mem.bump_meet)
    return {'ok': True, 'session_id': sid, 'meet_count': meet}


@app.get('/api/session/{sid}/messages')
async def api_session_messages(sid: str):
    sess = await anyio.to_thread.run_sync(lambda: mem.get_session(sid))
    msgs = await anyio.to_thread.run_sync(lambda: mem.messages_of(sid))
    return {'session': sess, 'messages': msgs}


@app.post('/api/session/rename')
async def api_session_rename(request: Request):
    body = await request.json()
    await anyio.to_thread.run_sync(
        lambda: mem.touch_session(body.get('session_id'), body.get('title')))
    return {'ok': True}


@app.delete('/api/session/{sid}')
async def api_session_delete(sid: str):
    await anyio.to_thread.run_sync(lambda: mem.delete_session(sid))
    return {'ok': True}


# ── 关系状态 ────────────────────────────────────────────────────────

@app.get('/api/relationship')
async def api_relationship():
    r = await anyio.to_thread.run_sync(mem.get_relationship)
    r['days_known'] = await anyio.to_thread.run_sync(mem.days_known)
    r.pop('id', None)
    return r


@app.post('/api/relationship/reset')
async def api_relationship_reset():
    await anyio.to_thread.run_sync(mem.reset_relationship)
    return {'ok': True}


@app.post('/api/relationship/adjust')
async def api_relationship_adjust(request: Request):
    body = await request.json()
    body.pop('id', None)
    r = await anyio.to_thread.run_sync(lambda: mem.update_relationship(**body))
    return r


# ── 事实记忆 ────────────────────────────────────────────────────────

@app.get('/api/memory/facts')
async def api_facts():
    return {'facts': await anyio.to_thread.run_sync(mem.all_facts),
            'categories': mem.CATEGORIES}


@app.post('/api/memory/facts')
async def api_fact_add(request: Request):
    body = await request.json()
    fid = await anyio.to_thread.run_sync(
        lambda: mem.upsert_fact(body.get('key', ''), body.get('value', ''),
                                body.get('category') or '其他',
                                pinned=bool(body.get('pinned', True))))
    return {'ok': bool(fid), 'id': fid}


@app.delete('/api/memory/facts/{fid}')
async def api_fact_delete(fid: int):
    await anyio.to_thread.run_sync(lambda: mem.delete_fact(fid))
    return {'ok': True}


@app.post('/api/memory/facts/{fid}/pin')
async def api_fact_pin(fid: int, request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    await anyio.to_thread.run_sync(
        lambda: mem.pin_fact(fid, bool(body.get('pinned', True))))
    return {'ok': True}


@app.get('/api/memory/export')
async def api_memory_export():
    md = await anyio.to_thread.run_sync(mem.export_markdown)
    return Response(content=md, media_type='text/markdown; charset=utf-8',
                    headers={'Content-Disposition': 'attachment; filename=airi-memory.md'})


@app.post('/api/memory/clear')
async def api_memory_clear():
    await anyio.to_thread.run_sync(mem.clear_facts)
    return {'ok': True}


# ── 聊天主流程 ──────────────────────────────────────────────────────

@app.post('/api/chat')
async def chat(request: Request):
    body = await request.json()
    text = (body.get('text') or '').strip()
    cfg = load_cfg()
    mcfg = cfg.get('memory', {})
    gap = float(mcfg.get('session_gap_hours', 2))

    session_id = None
    hist = []
    hits = []
    if mcfg.get('enabled', True):
        session_id = await anyio.to_thread.run_sync(
            lambda: mem.get_or_create_session(body.get('session_id'), gap_hours=gap))
        sys_prompt, hits = await anyio.to_thread.run_sync(
            lambda: build_system_prompt(cfg, session_id, text))
        hist = await anyio.to_thread.run_sync(
            lambda: mem.recent_messages(session_id, limit=int(cfg.get('history_len', 8))))
        await anyio.to_thread.run_sync(
            lambda: mem.add_message(session_id, 'user', text))
    else:
        sys_prompt = cfg['llm']['system_prompt']
        hist = [{'role': h.get('role', 'user'), 'content': h.get('content', '')}
                for h in (body.get('history') or [])[-int(cfg.get('history_len', 8)):]]

    async def gen():
        t_start = time.time()
        messages = [{'role': 'system', 'content': sys_prompt}]
        messages += [{'role': h['role'], 'content': h['content']} for h in hist]
        messages.append({'role': 'user', 'content': text})

        if session_id:
            yield 'data: ' + json.dumps(
                {'type': 'session', 'session_id': session_id},
                ensure_ascii=False) + '\n\n'

        if hits:
            yield 'data: ' + json.dumps(
                {'type': 'memory', 'count': len(hits),
                 'items': [f"{h['key']}：{h['value']}" for h in hits]},
                ensure_ascii=False) + '\n\n'

        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()
        threading.Thread(target=_pump_llm,
                         args=(messages, cfg, queue, loop), daemon=True).start()

        async def emit_sentence(s, idx):
            t0 = time.time()
            try:
                wav = await asyncio.wait_for(synthesize_async(s),
                                             timeout=TTS_TIMEOUT + 10)
                return {'type': 'audio', 'index': idx, 'text': s,
                        'audio': base64.b64encode(wav).decode(),
                        'elapsed': round(time.time() - t0, 2)}
            except asyncio.TimeoutError:
                return {'type': 'error', 'text': '合成超时（切换模型/预热中？）'}
            except Exception as e:
                return {'type': 'error', 'text': str(e)[:200]}

        buf = ''
        full = ''
        idx = 0
        while True:
            piece = await queue.get()
            if piece is None:
                break
            full += piece
            buf += piece
            yield 'data: ' + json.dumps({'type': 'delta', 'text': piece},
                                        ensure_ascii=False) + '\n\n'
            sentences, buf = split_sentences(buf, first=(idx == 0))
            for s in sentences:
                idx += 1
                ev = await emit_sentence(s, idx)
                yield 'data: ' + json.dumps(ev, ensure_ascii=False) + '\n\n'

        sentences, buf = split_sentences(buf, flush=True)
        for s in sentences:
            idx += 1
            ev = await emit_sentence(s, idx)
            yield 'data: ' + json.dumps(ev, ensure_ascii=False) + '\n\n'

        msg_id = None
        if session_id and full.strip():
            msg_id = await anyio.to_thread.run_sync(
                lambda: mem.add_message(session_id, 'assistant', full))

        yield 'data: ' + json.dumps({'type': 'done', 'text': full,
                                     'session_id': session_id,
                                     'total': round(time.time() - t_start, 2)},
                                    ensure_ascii=False) + '\n\n'
        print('[CHAT] %d 字, %d 句, 总 %.2fs' % (len(full), idx,
                                               time.time() - t_start), flush=True)

        # 后台抽取记忆 + 更新关系 (异步, 不影响本轮回复)
        if (session_id and msg_id and full.strip()
                and mcfg.get('auto_extract', True)):
            threading.Thread(target=extract_facts_bg,
                             args=(cfg, session_id, text, full, msg_id),
                             daemon=True).start()

    return StreamingResponse(gen(), media_type='text/event-stream')


def _warm_up_tts():
    """启动时后台预热一次完整链路 (把冷启动开销挪到启动阶段)。"""
    try:
        t0 = time.time()
        _synthesize_blocking('预热。')
        print('[WARM] TTS 链路预热完成 %.2fs' % (time.time() - t0), flush=True)
    except Exception as e:
        print('[WARM] 预热失败 (不影响使用): %s' % str(e)[:150], flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8770)
    ap.add_argument('--host', default='0.0.0.0')
    args = ap.parse_args()
    mem.init_db()
    apply_rvc_model(load_cfg()['tts'].get('rvc_model'))
    threading.Thread(target=_warm_up_tts, daemon=True).start()
    threading.Thread(target=_autostop_watchdog, daemon=True).start()
    print('本地聊天: http://localhost:%d  (TTS 后端 %s, 合成并发 %d, 关页自动停止=%s)'
          % (args.port, TTS_URL, MAX_CONCURRENT_TTS,
             '开' if load_cfg().get('auto_stop', True) else '关'), flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level='warning')
