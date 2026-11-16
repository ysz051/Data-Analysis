"""Streamlit 单页应用：上传 → Agent 分析 → 决策轨迹 → 表格填写回答 → 导出/入库。"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from src import config, memory
from src.graph import get_state, run_agent, submit_answers
from src.tools import TOOL_SPECS, export_rows

st.set_page_config(page_title=config.APP_TITLE, page_icon="📊", layout="wide")

SAMPLE_PATH = config.BASE_DIR / "data" / "sample_group_chat.csv"
EDITOR_KEY = "answer_editor"

STATUS_CN = {
    "perceived": "已解析",
    "extracted": "已抽取",
    "clustered": "已归并",
    "annotated": "已标注",
    "drafted": "已生成建议回答",
    "refined": "已二次合并",
    "awaiting_human": "等待人工填写",
    "reviewed": "人工已确认",
    "done": "已完成",
    "empty_input": "输入为空",
    "no_question": "未发现问题",
}

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; max-width: 1280px;}

    /* 隐藏 Streamlit 自带工具栏：Deploy、主菜单、页脚 */
    #MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"],
    [data-testid="stDecoration"], [data-testid="stStatusWidget"] {display: none !important;}

    /* 顶部标题栏 */
    .app-header {display: flex; justify-content: space-between; align-items: flex-start;
                 gap: 16px; padding-bottom: 14px; border-bottom: 1px solid #e5e7eb; margin-bottom: 18px;}
    .app-header .title {font-size: 1.3rem; font-weight: 700; color: #111827;}
    .app-header .desc {margin: 6px 0 0 0; color: #6b7280; font-size: .85rem;}
    .app-header .meta {display: flex; gap: 8px; flex-wrap: wrap; padding-top: 4px;}
    .chip {font-size: .74rem; color: #334155; background: #f8fafc; border: 1px solid #e2e8f0;
           border-radius: 4px; padding: 3px 8px; white-space: nowrap;}
    .chip.ok {color: #047857; background: #f0fdf4; border-color: #bbf7d0;}

    /* 区块标题 */
    .section {display: flex; align-items: center; gap: 8px; margin: 2px 0 10px 0;}
    .section::before {content: ""; width: 3px; height: 15px; background: #2563eb; border-radius: 1px;}
    .section span {font-weight: 600; font-size: .95rem; color: #111827;}

    /* 页签：下划线样式，不使用胶囊背景 */
    .stTabs [data-baseweb="tab-list"] {gap: 26px; border-bottom: 1px solid #e5e7eb;}
    .stTabs [data-baseweb="tab"] {background: transparent; border-radius: 0; padding: 8px 0 10px 0;
                                  color: #6b7280; font-size: .9rem;}
    .stTabs [data-baseweb="tab"]:hover {background: transparent; color: #111827;}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {background: transparent; color: #1d4ed8;
                                                        box-shadow: none;}
    .stTabs [data-baseweb="tab-highlight"] {background: #2563eb;}
    .stTabs [data-baseweb="tab-border"] {display: none;}

    /* 指标卡 */
    div[data-testid="stMetric"] {background: #fafbfc; border: 1px solid #e5e7eb; border-radius: 6px;
                                 padding: 12px 14px; box-shadow: none;}
    div[data-testid="stMetricLabel"] p {color: #6b7280; font-size: .78rem;}
    div[data-testid="stMetricValue"] {color: #111827; font-size: 1.3rem;}

    /* 按钮 */
    .stButton > button, .stDownloadButton > button {border-radius: 6px; font-weight: 500;
                                                    border: 1px solid #d1d5db; box-shadow: none;}
    .stButton > button[kind="primary"] {background: #2563eb; border-color: #2563eb; color: #fff;}
    .stButton > button[kind="primary"]:hover {background: #1d4ed8; border-color: #1d4ed8;
                                              filter: none; transform: none;}

    /* 输入控件 */
    .stTextArea textarea, .stTextInput input {border-radius: 6px !important; border: 1px solid #d1d5db !important;}
    .stTextArea textarea:focus {border-color: #2563eb !important;
                                box-shadow: 0 0 0 2px rgba(37, 99, 235, .12);}

    /* 上传区：隐藏英文提示，按钮改中文 */
    [data-testid="stFileUploaderDropzone"] {background: #fafbfc; border: 1px dashed #cbd5e1; border-radius: 6px;}
    [data-testid="stFileUploaderDropzone"]:hover {border-color: #2563eb;}
    [data-testid="stFileUploaderDropzoneInstructions"] {display: none;}
    [data-testid="stFileUploaderDropzone"] button {font-size: 0; border-radius: 6px; box-shadow: none;}
    [data-testid="stFileUploaderDropzone"] button::after {content: "选择文件"; font-size: .85rem; font-weight: 500;}

    /* 表格与折叠面板 */
    [data-testid="stDataFrame"], [data-testid="stDataEditor"] {border: 1px solid #e5e7eb;
                                                               border-radius: 6px; overflow: hidden;}
    div[data-testid="stExpander"] {border: 1px solid #e5e7eb; border-radius: 6px; background: #fff;}

    /* 提示条 */
    [data-testid="stAlert"] {border-radius: 6px; padding: 10px 12px;}
    [data-testid="stFileUploaderDropzone"]::before {content: "将文件拖拽到此处，或";
        font-size: .85rem; color: #6b7280;}

    /* 侧边栏 */
    [data-testid="stSidebar"] {background: #fafbfc; border-right: 1px solid #e5e7eb;}
    [data-testid="stSidebar"] h3 {font-size: .95rem; color: #111827;}

    /* 页脚 */
    .foot {margin-top: 26px; padding-top: 10px; border-top: 1px solid #eef2f7;
           color: #9ca3af; font-size: .76rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def _init_state() -> None:
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("turns", [])
    st.session_state.setdefault("rows", None)
    st.session_state.setdefault("submitted", False)
    st.session_state.setdefault("sample_bytes", None)
    st.session_state.setdefault("sample_name", "")


def _reset() -> None:
    for key in ("result", "turns", "rows", "submitted"):
        st.session_state[key] = None if key in ("result", "rows") else ([] if key == "turns" else False)
    st.session_state.pop(EDITOR_KEY, None)


def _section(title: str) -> None:
    st.markdown(f'<div class="section"><span>{title}</span></div>', unsafe_allow_html=True)


def _build_excel(rows: pd.DataFrame, turns) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        frame = rows.drop(columns=[c for c in ("cluster_id",) if c in rows.columns])
        frame.to_excel(writer, sheet_name="问题清单", index=False)
        if turns:
            pd.DataFrame(turns).to_excel(writer, sheet_name="原始对话", index=False)
    return buf.getvalue()


def _render_trace(result) -> None:
    if not result.trace:
        return
    with st.expander(f"Agent 决策轨迹（{len(result.trace)} 步，Thought → Action → Observation）", expanded=False):
        frame = pd.DataFrame(
            [
                {
                    "步骤": step.get("step"),
                    "时间": step.get("time"),
                    "思考 Thought": step.get("thought"),
                    "动作 Action": step.get("action"),
                    "观察 Observation": step.get("observation"),
                }
                for step in result.trace
            ]
        )
        st.dataframe(frame, width="stretch", hide_index=True)


def _render_table(result) -> None:
    base_rows = export_rows([c.model_dump() for c in result.clusters])
    if not base_rows:
        st.warning("没有抽取到客户问题，请检查对话内容或调整输入。")
        return

    frame = pd.DataFrame(base_rows)
    editable_cols = ["建议回答", "我的回答"]
    disabled = [c for c in frame.columns if c not in editable_cols and c != "cluster_id"]

    st.subheader("问题清单（可编辑）")
    st.caption("Agent 已按优先级排序；「建议回答」与「我的回答」两列可直接编辑，填写后提交即写入长期记忆。")
    edited = st.data_editor(
        frame,
        key=EDITOR_KEY,
        hide_index=True,
        width="stretch",
        num_rows="fixed",
        disabled=disabled,
        column_config={
            "序号": st.column_config.NumberColumn(width="small"),
            "问题（已合并）": st.column_config.TextColumn(width="large"),
            "出现次数": st.column_config.NumberColumn(width="small"),
            "占比": st.column_config.TextColumn(width="small"),
            "优先级": st.column_config.TextColumn(width="small"),
            "问题类型": st.column_config.TextColumn(width="small"),
            "分类": st.column_config.TextColumn(width="small"),
            "客户原声示例": st.column_config.TextColumn(width="medium"),
            "其他问法": st.column_config.TextColumn(width="medium"),
            "建议回答": st.column_config.TextColumn(width="large"),
            "我的回答": st.column_config.TextColumn(width="large"),
            "cluster_id": None,
        },
    )
    st.session_state["rows"] = edited

    col_a, col_b = st.columns([1, 3])
    with col_a:
        if st.button("提交回答并入库", type="primary", disabled=st.session_state.get("submitted", False)):
            pairs = zip(edited["cluster_id"].tolist(), edited["我的回答"].fillna("").astype(str).tolist())
            answers = {cid: ans for cid, ans in pairs if ans.strip()}
            if not answers:
                st.warning("请先在「我的回答」列填写至少一条回答。")
            else:
                try:
                    with st.spinner("正在写入长期记忆..."):
                        done = submit_answers(result.session_id, answers)
                except ValueError:
                    st.error("会话已失效（服务可能重启过），请点击「重置当前会话」后重新分析。")
                else:
                    st.session_state["submitted"] = True
                    st.success(f"已入库 {len(answers)} 条回答，本次会话状态：{done.status}。下次遇到同类问题会自动命中历史口径。")
    with col_b:
        st.download_button(
            "导出 Excel（问题清单 + 原始对话）",
            data=_build_excel(edited, st.session_state.get("turns") or []),
            file_name=f"问题归并结果_{result.session_id}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.markdown("**按类别查看具体问题**")
    for cluster in result.clusters:
        with st.expander(f"{cluster.cluster_id}｜{cluster.question}（{cluster.count} 条）"):
            for variant in cluster.variants:
                st.markdown(f"- {variant}")


# ------------------------------------------------------------------ 侧边栏
_init_state()
with st.sidebar:
    st.markdown("### 参数设置")
    llm = config.resolve_llm()
    if llm["enabled"]:
        st.success(f"LLM 已启用：{llm['provider']} · {llm['model']}")
    else:
        st.info("未检测到 API Key，当前为离线规则模式")

    business_context = st.text_area(
        "业务背景（可选）",
        placeholder="例如：我们提供企业培训服务，客户主要是 HR 和部门负责人",
        height=120,
    )
    threshold = st.slider("问题合并阈值", min_value=0.30, max_value=0.85, value=float(config.CLUSTER_THRESHOLD), step=0.05,
                          help="越低越容易把相似问题合并；主题归并优先于该阈值")
    suggest_answers = st.toggle("生成建议回答", value=True)
    use_memory = st.toggle("检索历史记忆", value=True)

    st.divider()
    stats = memory.memory_stats()
    st.caption(f"长期记忆：{stats['sessions']} 个会话 / {stats['clusters']} 条问题 / {stats['vectors']} 条向量")
    if st.button("重置当前会话", width="stretch"):
        _reset()
        st.rerun()

    with st.expander("Agent 可用工具（8 个）"):
        for spec in TOOL_SPECS:
            st.markdown(f"**`{spec['name']}`** — {spec['description']}")

# ------------------------------------------------------------------ 主区
st.markdown(
    f"""
    <div class="app-header">
      <div>
        <div class="title">{config.APP_TITLE}</div>
        <div class="desc">上传群聊记录 → 自动抽取并合并高频问题 → 标注优先级 → 表格填写回答并导出</div>
      </div>
      <div class="meta">
        <span class="chip {'ok' if llm['enabled'] else ''}">{'模型已启用' if llm['enabled'] else '离线规则模式'}</span>
        <span class="chip">LangGraph</span>
        <span class="chip">8 个工具</span>
        <span class="chip">本地存储</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

_section("数据输入")
tab_file, tab_text = st.tabs(["上传文件（Excel / CSV / Word）", "直接粘贴对话"])
file_bytes, file_name = None, ""
with tab_file:
    uploaded = st.file_uploader("上传群聊文件（选择文件或拖拽到虚线框内）", type=["xlsx", "xls", "csv", "docx", "txt", "md"])
    st.caption("支持 .xlsx / .xls / .csv / .docx / .txt / .md，单个文件不超过 50 MB。")
    if SAMPLE_PATH.exists() and st.button("载入示例群聊数据"):
        st.session_state["sample_bytes"] = SAMPLE_PATH.read_bytes()
        st.session_state["sample_name"] = SAMPLE_PATH.name
    if uploaded is not None:
        file_bytes, file_name = uploaded.getvalue(), uploaded.name
        st.caption(f"已选择：{uploaded.name}（{len(file_bytes) / 1024:.1f} KB）")
    elif st.session_state.get("sample_bytes"):
        file_bytes, file_name = st.session_state["sample_bytes"], st.session_state["sample_name"]
        st.caption(f"已载入示例数据：{file_name}，点击下方「开始分析」即可运行")

with tab_text:
    pasted = st.text_area(
        "粘贴群聊记录（上限 50,000 字符）",
        height=220,
        max_chars=50000,
        placeholder="在此处粘贴内容即可",
    )
    st.caption("支持「昵称：内容」或微信导出的「时间 + 昵称 / 内容」格式；最多 50,000 字符，输入框右下角会实时显示已输入字数。")

run_clicked = st.button("开始分析", type="primary", width="stretch")

if run_clicked:
    if not file_bytes and not pasted.strip():
        st.error("请上传文件或粘贴对话内容后再开始分析。")
    else:
        pasted_text = pasted
        if file_bytes and pasted.strip():
            # 文件与粘贴同时存在时只取其一，避免同一批对话被重复计入频次
            pasted_text = ""
            st.info("检测到同时上传了文件并粘贴了文本，本次仅使用上传的文件内容。如需粘贴内容，请先清空上传。")
        _reset()
        with st.spinner("Agent 正在解析对话、抽取问题并归并…"):
            result = run_agent(
                file_bytes=file_bytes,
                file_name=file_name,
                pasted_text=pasted_text,
                business_context=business_context,
                settings={
                    "cluster_threshold": threshold,
                    "suggest_answers": suggest_answers,
                    "use_memory": use_memory,
                },
            )
        st.session_state["result"] = result
        st.session_state["turns"] = get_state(result.session_id).get("turns", [])

result = st.session_state.get("result")
if result:
    _section("分析结果")
    if result.status == "empty_input" or not result.clusters:
        st.error("未能从输入中解析出有效问题。" + "；".join(result.errors))
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("有效消息", result.turn_count)
        m2.metric("客户诉求", result.question_count)
        m3.metric("合并后问题", result.cluster_count)
        p0 = sum(1 for c in result.clusters if c.priority == "P0")
        m4.metric("P0 紧急问题", p0)

        _render_trace(result)
        _render_table(result)

        with st.expander("查看 Agent 状态与运行信息"):
            st.json(
                {
                    "会话 ID": result.session_id,
                    "状态": STATUS_CN.get(result.status, result.status),
                    "模型模式": "离线规则模式" if result.llm_mode.startswith("offline") else result.llm_mode,
                    "决策步数": len(result.trace),
                }
            )

    st.divider()
    _section("历史记录")
    with st.expander("历史会话（长期记忆）"):
        sessions = memory.list_sessions(limit=10)
        if not sessions:
            st.caption("暂无历史会话，完成一次回答提交后即可沉淀。")
        else:
            history = pd.DataFrame(sessions).rename(
                columns={
                    "session_id": "会话 ID",
                    "created_at": "创建时间",
                    "file_name": "来源文件",
                    "turn_count": "消息数",
                    "question_count": "诉求数",
                    "cluster_count": "问题簇",
                    "status": "状态",
                    "llm_mode": "模型模式",
                }
            )
            history["状态"] = history["状态"].map(lambda s: STATUS_CN.get(s, s))
            history["模型模式"] = history["模型模式"].map(
                lambda m: "离线规则模式" if str(m).startswith("offline") else m
            )
            st.dataframe(history, width="stretch", hide_index=True)
            session_ids = [s["session_id"] for s in sessions]
            picked = st.selectbox("查看某个历史会话的问题清单", session_ids)
            if picked:
                rows = memory.get_session_clusters(picked)
                if rows:
                    detail = pd.DataFrame(rows)[["question", "category", "count", "priority", "answer"]].rename(
                        columns={
                            "question": "问题（已合并）",
                            "category": "分类",
                            "count": "出现次数",
                            "priority": "优先级",
                            "answer": "我的回答",
                        }
                    )
                    st.dataframe(detail, width="stretch", hide_index=True)
else:
    if SAMPLE_PATH.exists():
        st.info("提示：可直接点击「载入示例群聊数据」→「开始分析」体验完整流程。")

st.markdown(
    '<div class="foot">数据仅保存在本地 SQLite / Chroma，不会上传第三方；回答请人工确认后再对外发送。</div>',
    unsafe_allow_html=True,
)
