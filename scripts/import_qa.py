"""将 Q&A 对载入 paper_agent 知识库

Usage::

    python scripts/import_qa.py                          # 使用内置示例
    python scripts/import_qa.py --file my_qa.json        # 从 JSON 文件导入
    python scripts/import_qa.py --stdin                  # 从标准输入读取 JSON

JSON 格式::

    [
        {
            "question": "打浆温度多少合适？",
            "answer": "打浆温度一般控制在25-35°C...",
            "category": "打浆"
        },
        ...
    ]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

# ── Must be set BEFORE any HF imports ──
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.indexing.vector_store import VectorStore


def format_qa_chunk(qa: dict, index: int) -> str:
    """将单条 Q&A 格式化为可供检索的知识块。

    格式化策略：
    - 问答合一，问题部分用来命中检索，答案部分用来生成回复
    - 加入 [Q&A] 标记让 LLM 知道这是专家经验而非教材原文
    """
    question = qa.get("question", "").strip()
    answer = qa.get("answer", "").strip()

    if not question or not answer:
        raise ValueError(f"Q&A #{index} 缺少 question 或 answer 字段")

    lines = [
        f"【专家经验问答 #{index}】",
        f"问题：{question}",
        f"回答：{answer}",
    ]

    # 可选字段
    if qa.get("keywords"):
        lines.append(f"关键词：{qa['keywords']}")

    if qa.get("note"):
        lines.append(f"备注：{qa['note']}")

    return "\n".join(lines)


def load_from_json(filepath: str) -> List[dict]:
    """从 JSON 文件读取 Q&A 列表。"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON 顶层必须是数组 [{...}, {...}]")
    return data


def load_from_stdin() -> List[dict]:
    """从标准输入读取 JSON。"""
    raw = sys.stdin.read()
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("JSON 顶层必须是数组 [{...}, {...}]")
    return data


def import_qa_pairs(
    qa_list: List[dict],
    collection_name: str = "paper_knowledge",
    source: str = "user_qa",
) -> int:
    """将 Q&A 列表导入向量库。

    Returns:
        导入的条数
    """
    store = VectorStore()
    collection = store.get_or_create_collection(collection_name)

    chunks = []
    metadatas = []

    for i, qa in enumerate(qa_list, 1):
        try:
            chunk = format_qa_chunk(qa, i)
        except ValueError as e:
            print(f"[WARN] 跳过 #{i}: {e}", file=sys.stderr)
            continue

        chunks.append(chunk)
        metadatas.append({
            "source": source,
            "category": qa.get("category", "general"),
            "type": "qa_pair",
        })

    if not chunks:
        print("[!] 没有有效的 Q&A 可导入", file=sys.stderr)
        return 0

    store.add_documents(collection, chunks, metadatas, source=source)

    print(f"[OK] 已导入 {len(chunks)} 条 Q&A 到集合 '{collection_name}' (source='{source}')")
    print(f"[*] 集合总条数: {collection.count()}")

    # 打印预览 (使用 ascii 安全模式避免 Windows GBK 编码问题)
    print("\n--- Preview (first 200 chars of first 2 entries) ---")
    for i, chunk in enumerate(chunks[:2], 1):
        print(f"[{i}] {chunk[:200]}...")
        print("---")

    return len(chunks)


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

_BUILTIN_SAMPLES = [
    {
        "question": "打浆时浆料温度应该控制在什么范围？",
        "answer": (
            "打浆温度一般控制在25-35°C。温度过低（<20°C）时纤维润胀不足，"
            "打浆效率下降，能耗增加；温度过高（>45°C）会导致纤维过度水化，"
            "纸张强度反而下降，且容易产生泡沫。高打浆度纸种（如描图纸）可适当"
            "提高温度至35-40°C以促进纤维润胀。"
        ),
        "category": "打浆",
        "keywords": "打浆 温度 纤维润胀 能耗",
    },
    {
        "question": "AKD施胶后纸张为什么需要熟化？熟化需要多长时间？",
        "answer": (
            "AKD在中碱性条件下与纤维素羟基发生酯化反应，这个反应不是瞬间完成的。"
            "AKD分子需要在纤维表面扩散、取向，然后与纤维形成共价键，才能产生"
            "稳定的抗水效果。熟化条件：常温（20-25°C）下需要24-48小时；"
            "如果纸张下机后加热到60-80°C，熟化可以缩短到15-30分钟。"
            "加速熟化的方法：提高烘缸温度、添加熟化促进剂（如阳离子聚合物）、"
            "控制纸张pH在7.5-8.5之间。"
        ),
        "category": "施胶",
        "keywords": "AKD 熟化 施胶 抗水性 酯化反应",
    },
]


def main():
    parser = argparse.ArgumentParser(
        description="将造纸 Q&A 对导入向量知识库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--file", "-f",
        help="从 JSON 文件读取 Q&A 列表",
    )
    group.add_argument(
        "--stdin",
        action="store_true",
        help="从标准输入读取 JSON",
    )
    group.add_argument(
        "--demo",
        action="store_true",
        help="导入内置示例数据（测试用）",
    )
    parser.add_argument(
        "--source",
        default="user_qa",
        help="数据来源标签 (默认: user_qa)",
    )
    parser.add_argument(
        "--collection",
        default="paper_knowledge",
        help="Chroma 集合名 (默认: paper_knowledge)",
    )

    args = parser.parse_args()

    if args.file:
        qa_list = load_from_json(args.file)
    elif args.stdin:
        qa_list = load_from_stdin()
    elif args.demo:
        qa_list = _BUILTIN_SAMPLES
    else:
        parser.print_help()
        print("\n[!] 请指定 --file、--stdin 或 --demo")
        sys.exit(1)

    import_qa_pairs(qa_list, collection_name=args.collection, source=args.source)


if __name__ == "__main__":
    main()
