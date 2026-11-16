# 部署说明

## 1. 环境要求

| 项 | 要求 |
| --- | --- |
| Python | 3.9 ~ 3.11（本项目在 3.9.13 与 3.11 验证） |
| 内存 | ≥ 1 GB（Chroma 嵌入式运行） |
| 磁盘 | ≥ 500 MB（依赖 + 向量库） |
| 网络 | 离线可用；使用 DeepSeek 时需要外网 |
| 端口 | 8501（Streamlit 默认） |

## 2. 环境变量

复制 `.env.example` 为 `.env`：

| 变量 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `DEEPSEEK_API_KEY` | 否 | 空 | 留空则进入离线规则模式 |
| `DEEPSEEK_BASE_URL` | 否 | `https://api.deepseek.com` | OpenAI 兼容地址 |
| `DEEPSEEK_MODEL` | 否 | `deepseek-chat` | 模型名 |
| `OPENAI_API_KEY` | 否 | 空 | DeepSeek 未配置时的备选 |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` | 否 | 空 | 配置后自动上报 Trace |
| `MEMORY_DB_PATH` | 否 | `data/agent_memory.db` | SQLite 路径 |
| `CHROMA_DIR` | 否 | `data/chroma` | 向量库目录 |
| `CHROMA_COLLECTION` | 否 | `qa_memory` | collection 名（3~512 字符，字母数字开头） |
| `CLUSTER_THRESHOLD` | 否 | `0.55` | 相似度兜底阈值 |
| `MAX_REFINE_ROUNDS` | 否 | `1` | 自检二次合并轮数 |

> 安全要求：`.env` 已加入 `.gitignore`，仓库只提交 `.env.example`，不得提交真实 Key。

## 3. 本地运行

```bash
python -m pip install -r requirements.txt
cp .env.example .env        # 可选
python -m streamlit run app.py
```

访问 <http://localhost:8501>。若 `streamlit` 命令不在 PATH，请统一使用 `python -m streamlit`。

CLI：

```bash
python cli.py --file data/sample_group_chat.csv --answer-demo
```

评测：

```bash
python eval/run_eval.py
```

## 4. 容器化部署

```bash
docker build -t qa-merge-agent .
docker run -d --name qa-agent -p 8501:8501 --env-file .env \
  -v "$(pwd)/data:/app/data" qa-merge-agent
```

说明：
- `-v data:/app/data` 把 SQLite、Chroma、Trace 挂载到宿主机，容器重建不丢记忆；
- 镜像内置 `HEALTHCHECK`，访问 `/_stcore/health` 返回 `ok` 即为健康；
- 若使用 Docker Compose：

```yaml
services:
  agent:
    build: .
    ports: ["8501:8501"]
    env_file: .env
    volumes: ["./data:/app/data"]
    restart: unless-stopped
```

## 5. 云端部署

**方案 A：Streamlit Community Cloud**
1. 推送仓库到 GitHub（确认未提交 `.env`）；
2. 在 Streamlit Cloud 新建应用，入口文件填 `app.py`；
3. 在 Secrets 中填入 `DEEPSEEK_API_KEY`（Streamlit 会注入为环境变量）；
4. 注意：免费实例文件系统非持久化，SQLite/Chroma 会在重启后丢失，仅适合演示。

**方案 B：云服务器（ECS / CVM）**
1. 安装 Docker，按第 4 节运行，`-v` 挂载数据卷；
2. 安全组放行 8501（建议只放行内网或加 Nginx 反向代理 + Basic Auth）；
3. 生产建议：`docker run --restart=always`，并配置日志轮转。

## 6. 常见问题

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `command not found: streamlit` | 依赖装在用户目录，脚本未进 PATH | 用 `python -m streamlit run app.py` |
| 页面提示「离线规则模式」 | 未配置 `DEEPSEEK_API_KEY` | 属正常降级，功能完整；需要 LLM 时补 key 并重启 |
| 上传 Excel 后 0 条消息 | 内容列名不在候选表且全为空 | 检查表头，或在 `src/parsers.py` 的 `TEXT_COLS` 增加列名 |
| Chroma 报 collection 名非法 | 名称需 3~512 字符、字母数字开头 | 使用默认 `qa_memory` |
| 提交回答后再次点击无效 | 已入库，按钮被禁用 | 点「重置当前会话」开始新一轮 |
| 内存持续增长 | `MemorySaver` 保留所有会话状态 | 长期运行可定期重启进程，或换成 SQLite checkpointer |
| 上传大文件超时 | Streamlit 默认上限 200 MB | `.streamlit/config.toml` 已设 50 MB，可按需调整 |

## 7. 生产化建议

1. **鉴权**：Streamlit 无内置账号体系，建议前置 Nginx + Basic Auth 或 OAuth 网关。
2. **数据隔离**：多团队使用时，把 `session_id` 与租户 ID 绑定，Chroma collection 按租户分库。
3. **模型治理**：为 LLM 增加超时、重试、限流与成本统计（Langfuse 可直接看 token 与费用）。
4. **备份**：定时备份 `data/agent_memory.db` 与 `data/chroma`，二者共同构成长期记忆。
5. **评测回归**：每次修改 Prompt 或词典后跑一次 `python eval/run_eval.py`，防止指标回退。

## 8. 能否直接用于工作 / 跨系统访问

### 8.1 跨系统使用

本项目是 **Web 应用**，不是桌面软件：

| 角色 | 要求 |
| --- | --- |
| 服务端（跑 Agent 的机器） | Windows / macOS / Linux 均可，装 Python 3.9+ 或用 Docker |
| 客户端（同事打开的机器） | 只要有浏览器（Chrome / Edge / Safari / 新版 Firefox），Windows、macOS、Linux、手机、平板都能打开 |
| 访问方式 | 本机 `http://localhost:8501`；局域网 `http://<服务器IP>:8501`；公网需域名 + HTTPS |

因此不需要给每个人装环境，也不需要兼容不同操作系统；只要浏览器能访问到这台服务器即可。
不支持 IE 浏览器。

### 8.2 是否可以直接用于工作

**适合**：团队内部试点、客服/售前小组（1~5 人）、单机或内网部署、人工确认后再回复客户的场景。

**正式生产前需要补齐**：

| 事项 | 现状 | 建议 |
| --- | --- | --- |
| 账号与权限 | 无登录 | 前置 Nginx + Basic Auth，或接企业 SSO |
| 传输安全 | 默认 HTTP | 配 HTTPS 反向代理，禁止公网裸奔 8501 |
| 多用户并发 | Streamlit 单进程；图断点存在进程内 | 单副本运行；多副本需改用 SQLite/Redis checkpointer |
| 数据合规 | 对话与回答默认存本地；启用 LLM 时会把对话发给模型服务 | 敏感对话先脱敏，或改用私有化模型 |
| 规模 | 单文件 ≤ 50 MB、粘贴 ≤ 5 万字符 | 超大数据需分片与批处理 |
| 成本 | 离线模式零调用成本 | 在线模式建议加限流与用量统计（Langfuse 可看） |
| 备份 | 数据在 `data/` 下 | 定期备份 `agent_memory.db` 与 `chroma/` |

结论：**内部试点可以直接用；对外正式提供服务前，至少补上鉴权、HTTPS、数据脱敏和备份。**
