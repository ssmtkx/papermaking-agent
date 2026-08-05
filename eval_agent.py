"""Agent evaluation CLI — measure ReAct Agent answer quality across 20 fault scenarios.

Usage::

    python eval_agent.py                  # run all 20 scenarios (LLM calls!)
    python eval_agent.py --dry-run        # validate dataset only, no API calls
    python eval_agent.py --sample 5       # run 5 random scenarios
    python eval_agent.py --verbose        # print full answers
    python eval_agent.py --save res.json  # save results to file

Scoring dimensions (100 points per scenario)::

    Tool Selection   25%  —  expected tools actually called
    Structure        25%  —  三段式 (准备 / 分步教学 / 避坑指南) present
    Source Citation  25%  —  [来源N] markers in answer
    Keyword Coverage 25%  —  expected keywords found in answer

Requirements::

    DEEPSEEK_API_KEY env var must be set (real API calls).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Must be set BEFORE any HF imports ──
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.indexing.vector_store import VectorStore
from src.indexing.indexer import build_hybrid_from_collection
from src.retrieval.reranker import RerankerProcessor
from src.agent.react_agent import PaperReActAgent


# ═══════════════════════════════════════════════════════════════════
#  Scoring logic
# ═══════════════════════════════════════════════════════════════════

def score_tool_selection(
    called_tools: List[str],
    expected_tools: List[str],
) -> Tuple[float, str]:
    """Score tool selection: fraction of expected tools called (0–25)."""
    if not expected_tools:
        return 25.0, "(no expected tools → full)"
    called_set = set(called_tools)
    expected_set = set(expected_tools)
    matched = called_set & expected_set
    ratio = len(matched) / len(expected_set)
    detail = f"called={called_set}, expected={expected_set}, matched={matched}"
    return ratio * 25.0, detail


def score_structure(answer: str) -> Tuple[float, str]:
    """Score answer structure: presence of 3-section format (0–25)."""
    checks = {
        "准备": ["准备", "预备", "材料", "设备"],
        "步骤": ["步骤", "分步", "教学", "操作", "方法", "流程"],
        "避坑": ["避坑", "注意", "常见", "故障", "错误", "避免", "误区"],
    }
    scores = {}
    total = 0.0
    for section, keywords in checks.items():
        hit = any(kw in answer for kw in keywords)
        scores[section] = hit
        if hit:
            total += 25.0 / len(checks)
    detail = f"sections_found={[k for k, v in scores.items() if v]}"
    return total, detail


def score_sources(answer: str) -> Tuple[float, str]:
    """Score source citation: [来源N] markers in answer (0–25)."""
    sources = re.findall(r"\[来源\d+\]", answer)
    count = len(sources)
    # 2+ citations → full score
    ratio = min(count / 2.0, 1.0)
    detail = f"found {count} source markers"
    return ratio * 25.0, detail


def score_keywords(answer: str, expected_keywords: List[str]) -> Tuple[float, str]:
    """Score keyword coverage: fraction found in answer (0–25)."""
    if not expected_keywords:
        return 25.0, "(no expected keywords → full)"
    found = [kw for kw in expected_keywords if kw.lower() in answer.lower()]
    ratio = len(found) / len(expected_keywords)
    missed = [kw for kw in expected_keywords if kw.lower() not in answer.lower()]
    detail = f"found={found}, missed={missed}"
    return ratio * 25.0, detail


# ═══════════════════════════════════════════════════════════════════
#  Eval runner
# ═══════════════════════════════════════════════════════════════════

def evaluate_one(
    agent: PaperReActAgent,
    item: Dict[str, Any],
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run one eval scenario and return scored result."""
    question = item["question"]
    expected_tools = item.get("expected_tools", [])
    expected_keywords = item.get("expected_keywords", [])

    # ── call agent ──
    result = agent.chat(question)
    answer = result.get("answer", "")
    called_tools = [tc["tool"] for tc in result.get("tool_calls", [])]
    iterations = result.get("iterations", 0)

    # ── score ──
    tool_score, tool_detail = score_tool_selection(called_tools, expected_tools)
    structure_score, structure_detail = score_structure(answer)
    source_score, source_detail = score_sources(answer)
    keyword_score, keyword_detail = score_keywords(answer, expected_keywords)

    total = tool_score + structure_score + source_score + keyword_score

    if verbose:
        print(f"\n{'─' * 60}")
        print(f"[{item['id']}] {item['scenario']}")
        print(f"Q: {question}")
        print(f"A: {answer[:500]}{'...' if len(answer) > 500 else ''}")
        print(f"Iterations: {iterations}  Tools: {called_tools}")

    return {
        "id": item["id"],
        "scenario": item["scenario"],
        "question": question,
        "answer": answer,
        "iterations": iterations,
        "called_tools": called_tools,
        "scores": {
            "tool_selection": round(tool_score, 1),
            "structure": round(structure_score, 1),
            "source_citation": round(source_score, 1),
            "keyword_coverage": round(keyword_score, 1),
            "total": round(total, 1),
        },
        "details": {
            "tool_selection": tool_detail,
            "structure": structure_detail,
            "source_citation": source_detail,
            "keyword_coverage": keyword_detail,
        },
    }


def print_report(results: List[Dict[str, Any]]) -> None:
    """Print evaluation summary table."""
    n = len(results)
    avg_total = sum(r["scores"]["total"] for r in results) / n if n else 0.0
    avg_tool = sum(r["scores"]["tool_selection"] for r in results) / n if n else 0.0
    avg_struct = sum(r["scores"]["structure"] for r in results) / n if n else 0.0
    avg_src = sum(r["scores"]["source_citation"] for r in results) / n if n else 0.0
    avg_kw = sum(r["scores"]["keyword_coverage"] for r in results) / n if n else 0.0

    print("\n" + "=" * 72)
    print("                 🤖 ReAct Agent 评测报告")
    print("=" * 72)
    print(f"  评测场景: {n} 个")
    print(f"  平均总分: {avg_total:.1f} / 100")
    print()

    # ── dimension averages ──
    print("  " + "-" * 68)
    print(f"  {'维度':<20} {'权重':>6} {'平均分':>8} {'达标率':>8}")
    print("  " + "-" * 68)
    for label, weight, avg in [
        ("Tool Selection", "25%", avg_tool),
        ("Structure", "25%", avg_struct),
        ("Source Citation", "25%", avg_src),
        ("Keyword Coverage", "25%", avg_kw),
    ]:
        pct = avg / 25.0 * 100 if isinstance(weight, str) else avg / float(weight.strip('%')) * 100
        bar = _mini_bar(avg / 25.0)
        print(f"  {label:<20} {weight:>6} {avg:>7.1f}  {bar} {avg/25.0:.0%}")

    # ── per-item table ──
    print("\n  " + "-" * 68)
    print(f"  {'ID':<6} {'场景':<30} {'工具':>6} {'结构':>6} {'引用':>6} {'关键词':>6} {'总分':>6}")
    print("  " + "-" * 68)
    for r in results:
        s = r["scores"]
        print(
            f"  {r['id']:<6} {r['scenario'][:28]:<30} "
            f"{s['tool_selection']:>5.1f} {s['structure']:>5.1f} "
            f"{s['source_citation']:>5.1f} {s['keyword_coverage']:>5.1f} "
            f"{s['total']:>5.1f}"
        )

    # ── tier breakdown ──
    tiers = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
    for r in results:
        t = r["scores"]["total"]
        if t >= 80:
            tiers["excellent"] += 1
        elif t >= 60:
            tiers["good"] += 1
        elif t >= 40:
            tiers["fair"] += 1
        else:
            tiers["poor"] += 1

    print("\n  " + "-" * 68)
    print(f"  等级分布: 优秀(>=80): {tiers['excellent']}  |  "
          f"良好(>=60): {tiers['good']}  |  "
          f"一般(>=40): {tiers['fair']}  |  "
          f"需改进(<40): {tiers['poor']}")

    # ── worst performers ──
    worst = sorted(results, key=lambda r: r["scores"]["total"])[:3]
    if worst and worst[0]["scores"]["total"] < 60:
        print(f"\n  ⚠️  需改进的场景:")
        for r in worst:
            if r["scores"]["total"] < 60:
                print(f"  [{r['id']}] {r['scenario'][:50]}")
                print(f"       总分: {r['scores']['total']:.1f}  "
                      f"工具: {r['scores']['tool_selection']:.1f}  "
                      f"结构: {r['scores']['structure']:.1f}  "
                      f"引用: {r['scores']['source_citation']:.1f}  "
                      f"关键词: {r['scores']['keyword_coverage']:.1f}")

    print("\n" + "=" * 72 + "\n")


def _mini_bar(val: float, width: int = 10) -> str:
    filled = round(val * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


# ═══════════════════════════════════════════════════════════════════
#  CLI entry
# ═══════════════════════════════════════════════════════════════════

def build_retriever(collection_name: str = "paper_knowledge"):
    """Set up the three-stage retrieval chain."""
    store = VectorStore()
    collection = store.get_or_create_collection(collection_name)

    if collection.count() == 0:
        from scripts.seed_data import seed
        seed()
        collection = store.get_or_create_collection(collection_name)

    hybrid = build_hybrid_from_collection(collection, alpha=0.3)
    reranker = RerankerProcessor(hybrid, candidate_pool=20)
    return reranker


def main():
    parser = argparse.ArgumentParser(
        description="ReAct Agent evaluation on 20 fault scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate dataset only (no API calls)",
    )
    parser.add_argument(
        "--sample", type=int, default=0,
        help="Evaluate only N random scenarios (0 = all)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print full answers",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save full results to JSON file",
    )
    parser.add_argument(
        "--dataset", type=str, default="data/agent_eval.json",
        help="Path to agent eval dataset",
    )
    args = parser.parse_args()

    # ── load dataset ──
    with open(args.dataset, "r", encoding="utf-8") as fh:
        dataset = json.load(fh)

    print(f"[*] Loaded {len(dataset)} agent eval scenarios")
    cats = {}
    for item in dataset:
        cats[item["category"]] = cats.get(item["category"], 0) + 1
    print(f"[*] Categories: {cats}")

    if args.dry_run:
        # validate schema
        for item in dataset:
            for key in ("id", "scenario", "question", "expected_tools", "expected_keywords", "category", "difficulty"):
                assert key in item, f"Missing key '{key}' in {item.get('id', '?')}"
        print("[OK] Dataset schema valid — all 20 scenarios ready")
        print("[*] Run without --dry-run to execute evaluation (API calls required)")
        return

    # ── sample if requested ──
    scenarios = dataset
    if args.sample and args.sample < len(dataset):
        scenarios = random.sample(dataset, args.sample)
        print(f"[*] Sampled {len(scenarios)} scenarios")

    # ── check API key ──
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("[!] DEEPSEEK_API_KEY not set in .env — cannot call LLM.")
        print("[!] Set it in .env and retry, or use --dry-run to validate the dataset.")
        return

    # ── build retriever + agent ──
    print("[*] Building retriever ...")
    retriever = build_retriever()
    agent = PaperReActAgent(retriever, max_iterations=10, max_history=0, top_k=5)
    print("[*] Agent ready\n")

    # ── evaluate ──
    results = []
    for i, item in enumerate(scenarios, 1):
        qid = item["id"]
        print(f"[{i}/{len(scenarios)}] {qid}: {item['scenario'][:50]}...", end=" ", flush=True)
        try:
            r = evaluate_one(agent, item, verbose=args.verbose)
            results.append(r)
            total = r["scores"]["total"]
            emoji = "✅" if total >= 80 else ("⚠️" if total >= 60 else "❌")
            print(f"{emoji} {total:.0f}/100 (tools={r['called_tools']})")
        except Exception as exc:
            print(f"💥 ERROR: {exc}")
            results.append({
                "id": qid,
                "scenario": item["scenario"],
                "question": item["question"],
                "answer": f"ERROR: {exc}",
                "iterations": 0,
                "called_tools": [],
                "scores": {"tool_selection": 0, "structure": 0, "source_citation": 0, "keyword_coverage": 0, "total": 0},
                "details": {"error": str(exc)},
            })

        # clear history between scenarios
        agent.clear_history()

    # ── report ──
    print_report(results)

    if args.save:
        with open(args.save, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
        print(f"[OK] Results saved to {args.save}")

    # ── milestone check ──
    avg_total = sum(r["scores"]["total"] for r in results) / len(results) if results else 0
    if avg_total >= 70:
        print("  ✅ Agent 评测通过 (平均 >= 70/100)")
    else:
        print(f"  ⚠️  Agent 评测未达标 (平均 {avg_total:.1f}/100, 目标 70)")


if __name__ == "__main__":
    main()
