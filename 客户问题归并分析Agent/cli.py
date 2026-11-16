"""命令行入口：不启动 Web 也能跑完整 Agent，用于评测与现场答辩演示。

用法：
    python cli.py --file data/sample_group_chat.csv
    python cli.py --file data/sample_group_chat.csv --out data/result.xlsx
    python cli.py --text "王女士：退款怎么弄？"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src import memory
from src.graph import run_agent, submit_answers
from src.tools import export_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="客户问题归并分析 Agent（CLI）")
    parser.add_argument("--file", default="", help="Excel/CSV/Word 文件路径")
    parser.add_argument("--text", default="", help="直接粘贴的对话文本")
    parser.add_argument("--context", default="", help="业务背景说明")
    parser.add_argument("--out", default="", help="导出 Excel 路径")
    parser.add_argument("--threshold", type=float, default=0.55, help="聚类相似度阈值")
    parser.add_argument("--no-suggest", action="store_true", help="不生成建议回答")
    parser.add_argument("--answer-demo", action="store_true", help="自动填入示例回答并入库，用于验证记忆闭环")
    args = parser.parse_args()

    if not args.file and not args.text:
        parser.error("请提供 --file 或 --text")

    file_bytes = None
    file_name = args.file
    if args.file:
        path = Path(args.file)
        if not path.exists():
            parser.error(f"文件不存在：{args.file}")
        file_bytes = path.read_bytes()
        file_name = path.name

    result = run_agent(
        file_bytes=file_bytes,
        file_name=file_name,
        pasted_text=args.text,
        business_context=args.context,
        settings={"cluster_threshold": args.threshold, "suggest_answers": not args.no_suggest},
    )

    print("=" * 78)
    print(f"会话 {result.session_id} | 模式 {result.llm_mode} | 状态 {result.status}")
    print(f"消息 {result.turn_count} 条 / 问题 {result.question_count} 条 / 合并后 {result.cluster_count} 簇 / 合并率 {result.merge_rate:.0%}")
    print("=" * 78)
    print("\n[Agent 决策轨迹]")
    for step in result.trace:
        print(f"{step['step']}. [{step['time']}] Thought: {step['thought']}\n   Action: {step['action']}  Observation: {step['observation']}")

    print("\n[问题清单]")
    rows = export_rows([c.model_dump() for c in result.clusters])
    if rows:
        frame = pd.DataFrame(rows).drop(columns=["cluster_id", "其他问法", "客户原声示例"])
        with pd.option_context("display.max_colwidth", 50, "display.width", 200):
            print(frame.to_string(index=False))
    else:
        print("未抽取到问题")

    if args.answer_demo:
        answers = {c.cluster_id: (c.suggested_answer or "已确认，稍后同步。") for c in result.clusters}
        done = submit_answers(result.session_id, answers)
        print(f"\n[记忆闭环] 已提交 {len(answers)} 条回答，状态：{done.status}，库内统计：{memory.memory_stats()}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            pd.DataFrame(rows).to_excel(writer, sheet_name="问题清单", index=False)
        print(f"\n已导出：{out_path}")


if __name__ == "__main__":
    main()
