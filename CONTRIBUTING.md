# Contributing to 法小律 · 中国法律法规智能问答

感谢你的关注！以下是参与本项目的方式。

## 快速开始

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
cp .env.example .env          # 填入 DEEPSEEK_API_KEY

python scripts/crawl_laws.py  # 采集法规语料
python scripts/build_index.py --rebuild
streamlit run app.py
```

## 项目结构

参阅 [README.md](./README.md) 中的项目结构图。

## 开发约定

- **Python 3.10+**，类型注解使用 `from __future__ import annotations`
- 新增功能请在对应 `src/` 子模块下添加，保持模块职责清晰
- 检索器需实现 `.query(query_texts, n_results) -> dict` 接口，
  **并且必须返回 `metadatas`** —— 评测按 `(law_title, article_no)` 判定相关性，
  只返回文本的检索器会让所有指标为 0
- Agent 工具需同时提供 OpenAI function-calling schema 和 `ToolExecutor` 实现
- 领域相关文案统一放 `src/domain.py`，不要在多个文件里重写同一句话
- 改动分段式提示词（`react_agent._SYSTEM_PROMPT`）时，同步更新
  `eval_agent.score_structure` 的判定词表，否则评分衡量的是一套已不存在的要求

## 提交规范

```
feat: 添加 XX 功能
fix: 修复 XX 问题
docs: 更新文档
refactor: 重构 XX 模块
test: 添加 XX 测试
```

## 测试

```bash
python -m unittest discover -s tests
```

## 数据贡献

### 补充语料

语料由 `scripts/crawl_laws.py` 从国家法律法规数据库（flk.npc.gov.cn）采集，
**不要手工编辑 `data/raw/laws/` 下的文件** —— 重跑采集会覆盖它们。
需要调整采集范围时改 `CATEGORIES` 里的叶节点码。

站点有 WAF 限流，采集器会自动冷却；连续被拦会干净退出，稍后重跑即可续传。

### 补充评测集

评测集**必须来自语料之外的独立来源**。这是本项目的硬约束：

照着语料自己写问题，得到的指标衡量的是"问题与答案有多像"，
而不是检索能力 —— 这类自证循环在早期版本里真实发生过，
表现为一个漂亮但毫无信息量的 100%。

可用的独立来源见 README 的评测章节（EQUALS、CAIL2018 等）。
新增条目请遵守 `data/eval_law.json` 的 schema：

```json
{
  "id": "q001",
  "question": "违约金过高可以请求法院调减吗？",
  "category": "合同",
  "type": "knowledge",
  "source": "equals",
  "relevant": [{"law": "中华人民共和国民法典", "article": "第五百八十五条"}]
}
```

## License

MIT — 详见 [LICENSE](./LICENSE)
