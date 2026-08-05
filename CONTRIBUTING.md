# Contributing to 造纸智能助手 · 小纸

感谢你的关注！以下是参与本项目的方式。

## 快速开始

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
python scripts/seed_data.py
streamlit run app.py
```

## 项目结构

参阅 [README.md](./README.md) 中的项目结构图。

## 开发约定

- **Python 3.10+**，类型注解使用 `from __future__ import annotations`
- 新增功能请在对应 `src/` 子模块下添加，保持模块职责清晰
- 检索器需实现 `.query(query_texts, n_results) -> dict` 接口
- Agent 工具需同时提供 OpenAI function-calling schema 和 `ToolExecutor` 实现

## 提交规范

```
feat: 添加 XX 功能
fix: 修复 XX 问题
docs: 更新文档
refactor: 重构 XX 模块
test: 添加 XX 测试
```

## 数据贡献

如果你有造纸领域的专业知识，欢迎：

1. **扩充模拟知识库** — 编辑 `scripts/seed_data.py` 中的 `KNOWLEDGE_CHUNKS`
2. **扩充评测集** — 编辑 `data/eval_qa.json` 和 `data/agent_eval.json`
3. **提供真实数据** — 将脱敏后的 PDF 放入 `data/raw/`（注意版权和保密）

## License

MIT — 详见 [LICENSE](./LICENSE)
