# 工具接口说明

所有工具都是纯 Python 函数，统一通过 `src/tools.py: run_tool(name, **kwargs)` 调度；
每个工具的元信息（名称 / 描述 / JSON Schema / 示例）登记在 `TOOL_SPECS` 中，
可直接作为 LLM Function Calling 的工具描述，也可导出为 MCP Server 的 tool 定义。

## 0. 工具总览

| # | 工具 | 作用 | 被哪个节点调用 |
| --- | --- | --- | --- |
| 1 | `parse_chat_input` | 多格式解析为 `Turn[]` | perceive |
| 2 | `extract_questions` | 抽取客户诉求 `RawQuestion[]` | extract |
| 3 | `cluster_questions` | 语义归并 `Cluster[]` | cluster / refine |
| 4 | `annotate_clusters` | 频次 / 优先级 / 标签 | annotate / refine |
| 5 | `retrieve_similar_answers` | 向量库检索历史回答 | draft |
| 6 | `draft_answer` | 生成建议回答 | draft |
| 7 | `export_rows` | 转中文表头表格行 | review |
| 8 | `save_to_memory` | 写入 SQLite + Chroma | finalize |

---

## 1. `parse_chat_input`

**描述**：把 Excel / CSV / Word / 纯文本统一解析成消息序列。Excel 自动识别「内容 / 发言人 / 时间」列，
文本支持「昵称：内容」与微信「时间 昵称 / 内容」两种格式，并过滤 `[图片]`、撤回、入群等系统消息。

**Input**

```json
{
  "type": "object",
  "properties": {
    "file_name":   {"type": "string", "description": "文件名，用于判断格式"},
    "file_bytes":  {"type": "string", "format": "binary", "description": "文件二进制内容"},
    "pasted_text": {"type": "string", "description": "直接粘贴的对话文本"}
  },
  "required": []
}
```

**Output**：`Turn[]`

```json
[{"idx": 0, "speaker": "王女士", "timestamp": "2026-11-10 09:02:11", "text": "请问退款要怎么操作？", "source": "sample.csv"}]
```

**调用示例**

```python
from src.tools import run_tool
turns = run_tool("parse_chat_input", file_name="group.csv", file_bytes=open("group.csv", "rb").read())
turns = run_tool("parse_chat_input", pasted_text="王女士：退款怎么操作？")
```

**异常处理**：CSV 依次尝试 `utf-8-sig / gb18030 / utf-8`；Word 损坏时回退按文本解析；无法识别内容列时取平均长度最大的列。

---

## 2. `extract_questions`

**描述**：识别客户侧诉求（咨询 / 需求 / 故障 / 投诉）。昵称含「客服 / 助手 / 官方 / 销售」的说话人视为我方，
其发言不作为问题；故障描述即使没有问号（如「验证码根本收不到」）也会被识别。

**Input**

```json
{
  "type": "object",
  "properties": {
    "turns": {"type": "array", "items": {"$ref": "Turn"}},
    "business_context": {"type": "string", "description": "业务背景，注入 LLM 提升抽取准确率"}
  },
  "required": ["turns"]
}
```

**Output**：`RawQuestion[]`

```json
[{"qid": "Q001", "text": "退款要怎么操作", "qtype": "咨询", "speaker": "王女士", "timestamp": "", "source": "paste", "turn_idx": 0}]
```

**调用示例**

```python
questions = run_tool("extract_questions", turns=turns, business_context="SaaS 财务软件")
```

---

## 3. `cluster_questions`

**描述**：三级归并。
① 主题意图词典（退款 / 发票 / 登录 / 价格 / 导出 / 导入 / 上传 / 物流 / 子账号）把「同主题同性质」的问题合并并给出规范问法；
② 词典未覆盖时抽取「问题对象」：剥离「坏了 / 不能用 / 不对 / 有bug / 出问题 / 怎么办」等描述词，把「按钮坏了怎么办」「按钮不能用」「按钮有bug」统一归到「按钮相关问题」；
③ 仍无法归类的用字符 n-gram 余弦相似度兜底；
④ 配置了 LLM 时，再由 LLM 做一次跨主题归并，并用「成员覆盖完整性校验」防止漏项（校验失败自动回退）。
所有问题（包括只出现 1 次的）都会保留成行，合并只是「归类」，不会隐藏问题。

**Input**

```json
{
  "type": "object",
  "properties": {
    "questions": {"type": "array", "items": {"$ref": "RawQuestion"}},
    "threshold": {"type": "number", "minimum": 0.3, "maximum": 0.95, "default": 0.55},
    "business_context": {"type": "string"}
  },
  "required": ["questions"]
}
```

**Output**：`Cluster[]`

```json
[{
  "cluster_id": "C01",
  "question": "退款 / 售后政策与操作流程是什么？",
  "category": "售后退款",
  "qtype": "咨询",
  "count": 5,
  "ratio": 0.833,
  "variants": ["退款要怎么操作", "退款流程是什么", "退款大概多久能到账"],
  "speakers": ["王女士", "张先生"],
  "tags": ["退款与售后"],
  "priority": "P0"
}]
```

---

## 4. `annotate_clusters`

**描述**：计算占比、优先级与标签并排序。

优先级公式：

```
score = 0.60 × min(count / 4, 1) + 0.25 × type_weight + 0.15 × urgency
type_weight: 投诉 1.0 / 故障 0.9 / 需求 0.55 / 咨询 0.35
urgency: 命中「急 / 尽快 / 投诉 / 无法 / 退款 …」记 1，否则 0
P0 ≥ 0.72，P1 ≥ 0.50，其余 P2
```

**Input**

```json
{"type": "object",
 "properties": {"clusters": {"type": "array"}, "total_questions": {"type": "integer"}},
 "required": ["clusters"]}
```

**Output**：在 `Cluster` 基础上补齐 `ratio` / `priority` / `priority_score` / `tags` / `quote` / `evidence`，并按优先级降序。

---

## 5. `retrieve_similar_answers`

**描述**：从 Chroma 长期记忆中检索同主题的历史已答问题，返回答案与相似度，用于复用口径。

**Input**

```json
{"type": "object",
 "properties": {"question": {"type": "string"}, "k": {"type": "integer", "default": 3}},
 "required": ["question"]}
```

**Output**

```json
[{"question": "退款流程是什么？", "answer": "可在订单详情页申请退款…", "category": "售后退款", "score": 0.71}]
```

**异常处理**：向量库不可用或为空时返回 `[]`，不阻断主流程；调用方以 `score ≥ 0.45` 作为采纳阈值。

---

## 6. `draft_answer`

**描述**：生成建议回答。LLM 可用时结合「问题 + 其他问法 + 历史口径 + 业务背景」生成 2~4 句中文草稿；
LLM 不可用时按主题选择业务话术模板，并附历史口径参考。不确定信息统一用 `【待确认：xxx】` 标注。

**Input**

```json
{"type": "object",
 "properties": {
   "cluster": {"type": "object"},
   "history": {"type": "array", "items": {"type": "object"}},
   "business_context": {"type": "string"}
 },
 "required": ["cluster"]}
```

**Output**：`string`

```
退款可在「订单详情 → 申请退款」提交，审核通过后按原支付渠道退回。到账时间以支付渠道为准，一般为 1~7 个工作日。【待确认：具体到账时效】
```

---

## 7. `export_rows`

**描述**：把 `Cluster[]` 转成中文表头、可直接交给 `st.data_editor` 编辑与 Excel 导出的行数据。

**Input**

```json
{"type": "object",
 "properties": {"clusters": {"type": "array"}, "answers": {"type": "object"}},
 "required": ["clusters"]}
```

**Output**

```json
[{
  "序号": 1,
  "问题（已合并）": "退款 / 售后政策与操作流程是什么？",
  "出现次数": 5,
  "占比": "16.7%",
  "优先级": "P0",
  "问题类型": "咨询",
  "分类": "售后退款",
  "客户原声示例": "请问退款要怎么操作？",
  "其他问法": "退款流程是什么 / 退款大概多久能到账",
  "建议回答": "…",
  "我的回答": "",
  "cluster_id": "C01"
}]
```

---

## 8. `save_to_memory`

**描述**：把问题簇写入 SQLite（`clusters` 表）并更新会话状态；有回答的记录 upsert 进 Chroma，
形成「下次同类问题可直接复用」的长期记忆。

**Input**

```json
{"type": "object",
 "properties": {
   "session_id": {"type": "string"},
   "clusters": {"type": "array"},
   "meta": {"type": "object", "description": "{file_name, turn_count, question_count, cluster_count, status, llm_mode}"}
 },
 "required": ["session_id", "clusters"]}
```

**Output**

```json
{"saved": 8, "answered": 8, "session_id": "S20261116114634-2afc"}
```

---

## 9. 错误处理约定

| 场景 | 行为 |
| --- | --- |
| 未知工具名 | `run_tool` 抛 `KeyError`，由节点捕获并写入 `errors` |
| LLM 超时 / 额度不足 | `LLMClient.chat` 捕获异常，返回 `None`，自动走规则分支，`llm_mode` 标记为 `offline` |
| LLM 返回非法 JSON | `extract_json` 去 ``` 包裹、截取首个 JSON 片段；仍失败则降级 |
| LLM 归并漏项 | `_apply_llm_merge` 校验成员覆盖，不完整则抛异常并回退规则归并 |
| 空输入 | `clarify` 节点返回 `empty_input` 与提示文案，不抛异常 |
| 向量库异常 | `search_similar` 返回 `[]`，不影响主流程 |
