# airi-local-chat

本地 **LLM → TTS → RVC 音色转换 → 网页实时播放** 的语音聊天 demo：模型全部跑在本机（LLM 走 OpenAI 兼容 API），
边生成边合成边播放，带 **长期记忆 + 关系状态（好感/信任/心情）**。

浏览器 → `local_chat/app.py` (8770) → `tts_server.py` (8765, Kokoro 中/英文) → `rvc_work/rvc_serve.py` (8766, Windows GPU 上的 RVC 音色转换)

```
┌──────────┐  SSE   ┌────────────────┐  HTTP   ┌──────────────────┐  HTTP   ┌────────────────────┐
│ 浏览器   │ ─────► │ app.py  (8770) │ ──────► │ tts_server (8765)│ ──────► │ rvc_serve (8766)   │
│ 网页 UI  │ ◄───── │ 分句/记忆/关系 │ ◄────── │ Kokoro 基础语音  │ ◄────── │ RVC 音色 (Windows) │
└──────────┘ 音频   └────────────────┘   wav   └──────────────────┘   wav   └────────────────────┘
                            │
                            └──► OpenAI 兼容 LLM API（流式，如 DeepSeek）
```

## 特性

- **流式低延迟**：LLM 流式输出 → 按标点切句（首句阈值 18 字，之后 42 字）→ 每句立刻合成并排队播放，不等全文生成完
- **长期记忆**（SQLite）：会话 / 消息 / 事实 / 关系四张表
  - 每轮对话后台线程用 **一次** LLM 调用同时抽取「用户事实」+「关系变化」，不增加回复延迟
  - 跨会话召回：按关键词重叠 + 置顶/命中/新鲜度加权挑选要注入的记忆
  - 会话摘要：消息数达到阈值后把旧对话压缩进 summary，避免 prompt 无限增长
- **关系状态**：好感度 / 信任 / 心情（10 种 + 强度）/ 纪念事件 / 内心话，注入 system prompt 影响语气
- **时间感知**：注入当前时间、星期、距上次对话多久，跨天/久别重逢能自然反应
- **音色热切换**：RVC 模型 LRU 缓存（常驻最近 N 个）、CUDA Graph 加速、索引按模型名自动匹配（维度不符自动降级为不检索）
- **服务自愈**：RVC 服务健康检查 + 残留进程清理 + 崩溃自动重启；日志走自持文件句柄，不受父进程控制台失效影响
- **关页自动停止**：网页心跳 + 宽限期，关闭页面后自动停掉后台服务，刷新不会误触发

## 环境要求

| 组件 | 说明 |
|---|---|
| Windows + NVIDIA GPU | 显存 ~2.2GB（RVC 常驻），需自备 RVC 整合包（含 `runtime\python.exe`，torch cu128） |
| WSL2 Ubuntu | Python 3.12 venv（fastapi / uvicorn / anyio / kokoro / soundfile 等） |
| RVC 模型 | `.pth` 权重 + 可选 `.index` 索引，放在 RVC 的 `assets/weights` 与 `logs/` |
| LLM | 任意 OpenAI 兼容接口（示例用 DeepSeek） |

> ⚠️ **路径需要按自己的环境修改**（仓库里是作者本机路径）：
> - `local_chat/app.py` 的 `WEIGHTS`
> - `rvc_work/rvc_serve.py` 的 `RVC_ROOT`、`MODEL_CFG`

## 目录结构

```
.
├── tts_server.py              # WSL: Kokoro 基础语音 + 调用 Windows RVC, OpenAI 兼容 /v1/audio/speech
├── rvc_work/
│   ├── rvc_serve.py           # Windows: RVC GPU 服务 (8766), 模型热切换/索引匹配/自愈
│   ├── start_tts.sh           # 启动/停止/状态 (8765 + 8766)
│   └── 试听/                  # 各音色试听样本
└── local_chat/
    ├── app.py                 # 聊天服务 (8770): SSE 流式、记忆、关系、会话
    ├── memory.py              # SQLite 记忆层 (会话/消息/事实/关系)
    ├── index.html             # 单文件前端 (无框架): 气泡、侧边栏面板、播放队列
    ├── config.json            # 运行配置（需自己创建，见下）
    ├── start_all.sh / stop_all.sh
    └── start_airi.bat / stop_airi.bat   # Windows 侧一键启动（自动开浏览器）
```

## 快速开始

1. **建 venv 装依赖**（WSL 侧）：
   ```bash
   python3 -m venv venv && ./venv/bin/pip install fastapi uvicorn anyio kokoro soundfile
   ```
2. **写配置** `local_chat/config.json`（**不要提交到仓库**）：
   ```json
   {
     "llm": { "base_url": "https://api.deepseek.com/v1", "api_key": "sk-你的key",
              "model": "deepseek-chat", "system_prompt": "你是艾莉，一个温柔活泼的少女。"
              ,"temperature": 0.85, "max_tokens": 300 },
     "tts": { "voice": "airi", "speed": 1.0, "rvc_model": "airi_e360.pth", "rvc_index_rate": 0.05 },
     "memory": { "enabled": true, "auto_extract": true, "model": "deepseek-chat",
                 "inject_max_chars": 700, "session_gap_hours": 2, "summary_after": 24 },
     "auto_stop": true, "auto_stop_grace": 20
   }
   ```
3. **启动**：
   ```bash
   bash local_chat/start_all.sh      # 起 TTS(8765) + 聊天(8770)，RVC 首次请求自动拉起(约 30s 冷启动)
   ```
4. 打开 <http://localhost:8770>（Windows 侧 `start_airi.bat` 会自动开浏览器）

## HTTP 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/chat` | 主对话，SSE 流式返回 |
| POST | `/api/tts` | 单句试听，直接返回 wav |
| GET/POST | `/api/config` | 读取/保存配置（含 memory 段） |
| GET | `/api/sessions` | 会话列表（含消息数） |
| POST | `/api/session/new` | 新会话 |
| GET | `/api/session/{sid}/messages` | 某会话全部消息 |
| POST | `/api/session/rename` · DELETE `/api/session/{sid}` | 重命名 / 删除 |
| GET | `/api/relationship` · POST `/api/relationship/reset` · `/adjust` | 关系状态 |
| GET/POST | `/api/memory/facts` · DELETE `/api/memory/facts/{id}` · POST `/api/memory/facts/{id}/pin` | 事实记忆 |
| GET | `/api/memory/export` | 导出 Markdown |
| POST | `/api/memory/clear` | 清空事实 |
| POST | `/v1/audio/speech` | tts_server：OpenAI 兼容语音合成（`input`/`voice`/`speed`/`rvc_model`/`rvc_index_rate`） |

### SSE 事件

| type | 字段 | 说明 |
|---|---|---|
| `session` | `session_id` | 本轮所属会话 |
| `memory` | `count`, `items` | 命中的记忆条数与内容 |
| `delta` | `text` | LLM 增量文本 |
| `audio` | `index`, `text`, `audio`(base64 wav), `elapsed` | 第 N 句音频 |
| `error` | `text` | 单句合成失败（不中断整轮） |
| `done` | `text`, `session_id`, `total` | 本轮结束 + 总耗时 |

## 记忆是怎么工作的

```
用户发言 ──► build_system_prompt(): 人设 + [时间] + [剧情摘要] + [你此刻的状态] + [关于用户的长期记忆]
回复完成 ──► 后台线程 extract_facts_bg(): 一次 LLM 调用 ─┬─► facts 表 (key/value/类别/证据/置信度)
                                                        ├─► relationship 表 (好感/信任/心情/纪念增量)
                                                        └─► 消息够多时压缩会话摘要
```

记忆注入有字符预算（默认 700），按相关性挑选；关系与摘要各有独立开关，可以只留人设做纯聊天。

## 已知限制

- 需要自备 RVC 权重与索引，本仓库不含任何模型文件
- 作者本机路径硬编码（`E:\...`、`/mnt/e/...`），移植前必须改
- 暂不支持多用户/鉴权，是本机自用 demo
- `assets/` 里的背景图与 `试听/` 音频样本、`data/refs/` 参考音频仅用于个人测试，**版权归原作者**，请勿再分发

## License

## License

**无（No License）**

本仓库未附带任何许可证，**保留所有权利（All rights reserved）**。
未经作者明确许可，不得复制、分发、修改或用于商业用途。

仓库内的第三方素材 —— `local_chat/assets/` 背景图、`rvc_work/试听/` 与
`local_chat/data/refs/` 音频样本 —— 版权归各自原作者，仅用于个人测试，请勿再分发。
