#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""记忆存储层 — SQLite 实现（会话 / 消息 / 结构化事实）。

设计要点:
  * 单一 SQLite 文件, 线程安全 (sqlite3 + 全局锁, 后台线程可安全写入)
  * 会话自动切分: 距上次消息超过 session_gap_hours 就自动开新会话
  * 事实记忆: key-value + 分类 + 证据 + 命中次数, 同 key 覆盖更新
  * 全部本地存储, 不上传任何东西 (但注入 prompt 的记忆会随 LLM 请求发出)

表结构:
    sessions(id, title, created_at, updated_at, summary)
    messages(id, session_id, role, content, ts)
    facts(id, key, value, category, confidence, evidence,
          source_msg_id, hits, created_at, updated_at, pinned)
"""
import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path

HERE = Path(__file__).parent
DATA_DIR = HERE / 'data'
DB_PATH = DATA_DIR / 'memory.db'

_lock = threading.RLock()
_conn = None

CATEGORIES = ['身份', '偏好', '关系', '计划', '禁忌', '梗']


def _connect():
    global _conn
    if _conn is None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=10)
        _conn.row_factory = sqlite3.Row
        _conn.execute('PRAGMA journal_mode=WAL')
    return _conn


def init_db():
    with _lock:
        c = _connect()
        c.executescript('''
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT DEFAULT '',
            created_at REAL,
            updated_at REAL,
            summary TEXT DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            ts REAL,
            summarized INTEGER DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_id, id);
        CREATE TABLE IF NOT EXISTS relationship (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            affinity REAL DEFAULT 50,
            trust REAL DEFAULT 50,
            mood TEXT DEFAULT '平静',
            mood_intensity REAL DEFAULT 0.5,
            last_mood_ts REAL,
            meet_count INTEGER DEFAULT 0,
            first_seen REAL,
            milestones TEXT DEFAULT '[]',
            note TEXT DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE,
            value TEXT,
            category TEXT DEFAULT '其他',
            confidence REAL DEFAULT 0.8,
            evidence TEXT DEFAULT '',
            source_msg_id INTEGER,
            hits INTEGER DEFAULT 0,
            created_at REAL,
            updated_at REAL,
            pinned INTEGER DEFAULT 0
        );
        ''')
        c.commit()
        c.execute('INSERT OR IGNORE INTO relationship(id, first_seen) VALUES(1, ?)',
                  (time.time(),))
        c.commit()


# ── 关系状态 (好感度 / 信任 / 心情 / 纪念) ──────────────────────────

DEFAULT_REL = {'affinity': 50.0, 'trust': 50.0, 'mood': '平静',
               'mood_intensity': 0.5, 'meet_count': 0, 'first_seen': None,
               'milestones': [], 'note': ''}


def get_relationship():
    with _lock:
        r = _connect().execute('SELECT * FROM relationship WHERE id=1').fetchone()
    if not r:
        return dict(DEFAULT_REL)
    d = dict(r)
    try:
        d['milestones'] = json.loads(d.get('milestones') or '[]')
    except Exception:
        d['milestones'] = []
    return d


def days_known():
    r = get_relationship()
    if not r.get('first_seen'):
        return 1
    return max(1, int((time.time() - r['first_seen']) / 86400) + 1)


def bump_meet():
    """新会话开始 = 又一次见面。"""
    with _lock:
        c = _connect()
        c.execute('UPDATE relationship SET meet_count = meet_count + 1 WHERE id=1')
        c.commit()
    return get_relationship()['meet_count']


def update_relationship(affinity_delta=0.0, trust_delta=0.0, mood=None,
                        mood_intensity=None, milestone=None, note=None, clamp=True):
    """按增量更新关系状态 (好感度/信任 0~100, 心情文字, 可加纪念)。"""
    with _lock:
        c = _connect()
        r = get_relationship()
        aff = r['affinity'] + float(affinity_delta or 0)
        tru = r['trust'] + float(trust_delta or 0)
        if clamp:
            aff = max(0.0, min(100.0, aff))
            tru = max(0.0, min(100.0, tru))
        ms = r['milestones']
        if milestone:
            text = str(milestone).strip()[:80]
            if text and not any(m.get('text') == text for m in ms):
                ms.append({'ts': time.time(), 'text': text})
                ms = ms[-20:]
        c.execute('''UPDATE relationship SET affinity=?, trust=?, mood=?,
                     mood_intensity=?, last_mood_ts=?, milestones=?, note=?
                     WHERE id=1''',
                  (aff, tru,
                   (mood or r['mood'])[:20],
                   float(mood_intensity if mood_intensity is not None
                         else r['mood_intensity']),
                   time.time() if mood else r.get('last_mood_ts'),
                   json.dumps(ms, ensure_ascii=False),
                   (note if note is not None else r.get('note', ''))[:300]))
        c.commit()
    return get_relationship()


def reset_relationship():
    with _lock:
        c = _connect()
        c.execute("""UPDATE relationship SET affinity=50, trust=50, mood='平静',
                     mood_intensity=0.5, meet_count=0, milestones='[]', note='',
                     first_seen=? WHERE id=1""", (time.time(),))
        c.commit()
    return get_relationship()


def rel_for_prompt(session_id=None, max_chars=280):
    """生成注入 prompt 的「此刻状态」块。"""
    r = get_relationship()
    lines = ['[你此刻的状态]']
    mood = r.get('mood') or '平静'
    inten = r.get('mood_intensity') or 0.5
    grade = '很强烈' if inten > 0.75 else ('较明显' if inten > 0.5 else '淡淡的')
    lines.append('- 心情：%s（%s）' % (mood, grade))
    lines.append('- 与用户：好感度 %.0f/100，信任 %.0f/100，认识 %d 天，共见面 %d 次'
                 % (r.get('affinity', 50), r.get('trust', 50),
                    days_known(), r.get('meet_count', 0)))
    ms = r.get('milestones') or []
    if ms:
        lines.append('- 你们的纪念：' + '；'.join(m['text'] for m in ms[-3:]))
    if r.get('note'):
        lines.append('- 你的私下想法：' + r['note'][:120])
    lines.append('(用符合该状态的语气回应, 直接表达情绪, 但不要报数字、'
                 '也不要说"根据我的状态")')
    return '\n'.join(lines)[:max_chars]


# ── 会话 ────────────────────────────────────────────────────────────

def last_message_ts(session_id):
    with _lock:
        row = _connect().execute(
            'SELECT MAX(ts) t FROM messages WHERE session_id=?', (session_id,)).fetchone()
    return row['t'] if row and row['t'] else None


def get_or_create_session(session_id=None, gap_hours=2.0):
    """取会话; 若距上次消息超过 gap_hours 则自动开新会话。返回 session id。"""
    now = time.time()
    with _lock:
        c = _connect()
        if session_id:
            row = c.execute('SELECT id FROM sessions WHERE id=?', (session_id,)).fetchone()
            if row:
                last = last_message_ts(session_id)
                if last and (now - last) > gap_hours * 3600:
                    return new_session()
                return session_id
        # 没有指定/找不到 → 用最近一个会话, 否则新建
        row = c.execute('SELECT id FROM sessions ORDER BY updated_at DESC LIMIT 1').fetchone()
        if row:
            last = last_message_ts(row['id'])
            if last and (now - last) <= gap_hours * 3600:
                return row['id']
        return new_session()


def new_session(title=''):
    sid = time.strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:4]
    now = time.time()
    with _lock:
        c = _connect()
        c.execute('INSERT INTO sessions(id,title,created_at,updated_at) VALUES(?,?,?,?)',
                  (sid, title or time.strftime('%m-%d %H:%M'), now, now))
        c.commit()
    return sid


def touch_session(session_id, title=None):
    with _lock:
        c = _connect()
        if title:
            c.execute('UPDATE sessions SET updated_at=?, title=? WHERE id=?',
                      (time.time(), title, session_id))
        else:
            c.execute('UPDATE sessions SET updated_at=? WHERE id=?', (time.time(), session_id))
        c.commit()


def list_sessions(limit=50):
    with _lock:
        rows = _connect().execute('''
            SELECT s.id, s.title, s.created_at, s.updated_at, s.summary,
                   (SELECT COUNT(*) FROM messages m WHERE m.session_id=s.id) AS n
            FROM sessions s ORDER BY s.updated_at DESC LIMIT ?''', (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_session(session_id):
    with _lock:
        r = _connect().execute('SELECT * FROM sessions WHERE id=?', (session_id,)).fetchone()
    return dict(r) if r else None


def delete_session(session_id):
    with _lock:
        c = _connect()
        c.execute('DELETE FROM messages WHERE session_id=?', (session_id,))
        c.execute('DELETE FROM sessions WHERE id=?', (session_id,))
        c.commit()


def set_summary(session_id, summary):
    with _lock:
        c = _connect()
        c.execute('UPDATE sessions SET summary=? WHERE id=?', (summary, session_id))
        c.commit()


# ── 消息 ────────────────────────────────────────────────────────────

def add_message(session_id, role, content):
    with _lock:
        c = _connect()
        cur = c.execute('INSERT INTO messages(session_id,role,content,ts) VALUES(?,?,?,?)',
                        (session_id, role, content, time.time()))
        c.commit()
        mid = cur.lastrowid
    touch_session(session_id)
    return mid


def recent_messages(session_id, limit=12):
    with _lock:
        rows = _connect().execute('''
            SELECT id, role, content, ts FROM messages
            WHERE session_id=? ORDER BY id DESC LIMIT ?''',
            (session_id, limit)).fetchall()
    return [dict(r) for r in reversed(rows)]


def messages_of(session_id, limit=500):
    with _lock:
        rows = _connect().execute('''
            SELECT id, role, content, ts FROM messages
            WHERE session_id=? ORDER BY id ASC LIMIT ?''',
            (session_id, limit)).fetchall()
    return [dict(r) for r in rows]


def message_count(session_id):
    with _lock:
        r = _connect().execute('SELECT COUNT(*) n FROM messages WHERE session_id=?',
                               (session_id,)).fetchone()
    return r['n'] if r else 0


def unsummarized_before(session_id, keep_recent=12):
    """取出「较早、尚未摘要」的消息 (留最近 keep_recent 条不做摘要)。"""
    with _lock:
        rows = _connect().execute('''
            SELECT id, role, content FROM messages
            WHERE session_id=? AND summarized=0
            ORDER BY id ASC''', (session_id,)).fetchall()
    rows = [dict(r) for r in rows]
    return rows[:-keep_recent] if len(rows) > keep_recent else []


def mark_summarized(ids):
    if not ids:
        return
    with _lock:
        c = _connect()
        c.executemany('UPDATE messages SET summarized=1 WHERE id=?', [(i,) for i in ids])
        c.commit()


# ── 事实记忆 ────────────────────────────────────────────────────────

def upsert_fact(key, value, category='其他', confidence=0.8, evidence='',
                source_msg_id=None, pinned=None):
    key = (key or '').strip()[:60]
    value = (value or '').strip()[:300]
    if not key or not value:
        return None
    now = time.time()
    with _lock:
        c = _connect()
        row = c.execute('SELECT id, value, pinned FROM facts WHERE key=?', (key,)).fetchone()
        if row:
            if row['value'] == value:
                c.execute('UPDATE facts SET hits=hits+1, updated_at=? WHERE id=?',
                          (now, row['id']))
            else:
                c.execute('''UPDATE facts SET value=?, category=?, confidence=?,
                             evidence=?, source_msg_id=?, updated_at=? WHERE id=?''',
                          (value, category, confidence, evidence, source_msg_id, now,
                           row['id']))
            c.commit()
            return row['id']
        cur = c.execute('''INSERT INTO facts(key,value,category,confidence,evidence,
                            source_msg_id,hits,created_at,updated_at,pinned)
                           VALUES(?,?,?,?,?,?,0,?,?,?)''',
                        (key, value, category, confidence, evidence, source_msg_id,
                         now, now, 1 if pinned else 0))
        c.commit()
        return cur.lastrowid


def all_facts():
    with _lock:
        rows = _connect().execute('''
            SELECT * FROM facts ORDER BY pinned DESC, hits DESC, updated_at DESC''').fetchall()
    return [dict(r) for r in rows]


def search_facts(text, limit=10):
    """极简相关性: 关键词字符重叠 + pinned/hits/新鲜度加权 (不需要向量库)。"""
    facts = all_facts()
    if not facts:
        return []
    text = text or ''
    chars = set(text)
    scored = []
    for f in facts:
        key, val = f['key'], f['value']
        overlap = 0
        for token in (key + val):
            if token in chars:
                overlap += 1
        score = overlap * 2.0
        if f['pinned']:
            score += 6.0
        score += min(f['hits'], 5) * 0.8
        age_days = (time.time() - (f['updated_at'] or 0)) / 86400
        score += max(0.0, 3.0 - age_days * 0.2)
        scored.append((score, f))
    scored.sort(key=lambda x: -x[0])
    picked = [f for _, f in scored[:limit]]
    with _lock:
        c = _connect()
        c.executemany('UPDATE facts SET hits=hits+1 WHERE id=?', [(f['id'],) for f in picked])
        c.commit()
    return picked


def facts_for_prompt(text, max_chars=700):
    """挑出要注入 prompt 的事实, 返回 (文本块, 命中列表)。"""
    picked = search_facts(text, limit=10)
    lines, used, hits = [], 0, []
    for f in picked:
        line = f"- {f['key']}：{f['value']}"
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line) + 1
        hits.append(f)
    return ('\n'.join(lines), hits)


def delete_fact(fid):
    with _lock:
        c = _connect()
        c.execute('DELETE FROM facts WHERE id=?', (fid,))
        c.commit()


def pin_fact(fid, pinned=True):
    with _lock:
        c = _connect()
        c.execute('UPDATE facts SET pinned=? WHERE id=?', (1 if pinned else 0, fid))
        c.commit()


def clear_facts():
    with _lock:
        c = _connect()
        c.execute('DELETE FROM facts')
        c.commit()


def export_markdown():
    out = ['# 艾莉记忆导出', '']
    out.append('## 事实记忆')
    for f in all_facts():
        tag = '（置顶）' if f['pinned'] else ''
        out.append(f"- [{f['category']}] {f['key']}：{f['value']}{tag}"
                   + (f"  ← 证据：{f['evidence']}" if f['evidence'] else ''))
    out.append('')
    out.append('## 会话记录')
    for s in list_sessions():
        out.append(f"\n### {s['title']} ({s['id']}, {s['n']} 条)")
        if s['summary']:
            out.append(f"> 摘要：{s['summary']}")
        for m in messages_of(s['id']):
            who = '我' if m['role'] == 'user' else '艾莉'
            out.append(f"- **{who}**: {m['content']}")
    return '\n'.join(out)


if __name__ == '__main__':
    init_db()
    print('数据库:', DB_PATH)
    sid = new_session('测试会话')
    add_message(sid, 'user', '我叫小风，在学日语')
    add_message(sid, 'assistant', '记住啦，小风！')
    upsert_fact('用户称呼', '小风', '身份', 0.9, '我叫小风')
    print('会话:', list_sessions())
    print('消息:', recent_messages(sid))
    print('事实:', all_facts())
    print('命中:', facts_for_prompt('小风最近怎么样')[0])
