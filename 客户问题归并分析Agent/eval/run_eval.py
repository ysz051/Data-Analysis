"""离线评测脚本：逐条跑 12 个用例，输出 Markdown 评测报告。

运行：python eval/run_eval.py
结果：docs/eval_results.md
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "eval"))

from cases import CASES  # noqa: E402

from src.parsers import parse_text  # noqa: E402
from src.tools import annotate_clusters, cluster_questions, extract_questions  # noqa: E402


def run_case(case) -> dict:
    started = time.perf_counter()
    turns = [t.model_dump() for t in parse_text(case["input"])]
    questions = extract_questions(turns)
    clusters = annotate_clusters(cluster_questions(questions), len(questions))
    elapsed = (time.perf_counter() - started) * 1000

    corpus = [c["question"] + " " + " ".join(c.get("variants", [])) for c in clusters]
    group_details = []
    groups_ok = True
    for group in case["expect_groups"]:
        hit = next((c for c, text in zip(clusters, corpus) if all(k in text for k in group)), None)
        group_details.append((group, hit))
        groups_ok = groups_ok and hit is not None

    size_ok = len(clusters) <= case["max_clusters"] and len(questions) >= case["min_questions"]
    passed = groups_ok and size_ok

    if passed:
        detail = "通过"
    elif not groups_ok:
        missed = ["/".join(g) for g, hit in group_details if hit is None]
        detail = f"归并缺失：{'、'.join(missed)}"
    else:
        detail = f"规模超限：簇 {len(clusters)} > {case['max_clusters']} 或问题数 {len(questions)} < {case['min_questions']}"

    return {
        "id": case["id"],
        "name": case["name"],
        "focus": case["focus"],
        "turns": len(turns),
        "questions": len(questions),
        "clusters": len(clusters),
        "passed": passed,
        "detail": detail,
        "ms": round(elapsed, 1),
        "top": clusters[0]["question"] if clusters else "-",
        "priority": clusters[0]["priority"] if clusters else "-",
    }


def main() -> None:
    results = [run_case(case) for case in CASES]
    passed = sum(1 for r in results if r["passed"])
    total_ms = sum(r["ms"] for r in results)

    lines = [
        "# 评测记录（自动生成）",
        "",
        f"- 用例数：{len(results)}，通过：{passed}，通过率：{passed / len(results):.0%}",
        f"- 总耗时：{total_ms:.0f} ms（单用例平均 {total_ms / len(results):.0f} ms）",
        "- 运行模式：离线规则模式（未配置 API Key），指标可复现",
        "",
        "| 用例 | 能力点 | 消息 | 诉求 | 簇 | 结果 | 耗时(ms) | 最高优先级问题 |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r['id']} {r['name']} | {r['focus']} | {r['turns']} | {r['questions']} | {r['clusters']} | "
            f"{'✅ 通过' if r['passed'] else '❌ ' + r['detail']} | {r['ms']} | {r['top']}（{r['priority']}） |"
        )

    failures = [r for r in results if not r["passed"]]
    lines += ["", "## 失败用例分析", ""]
    if not failures:
        lines.append("本轮全部通过。已知的失败模式记录在 docs/evaluation.md 的「失败案例复盘」章节。")
    else:
        for r in failures:
            lines.append(f"- **{r['id']} {r['name']}**：{r['detail']}。原因与改进方向见 docs/evaluation.md。")

    out = BASE / "docs" / "eval_results.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n已写入：{out}")


if __name__ == "__main__":
    main()
