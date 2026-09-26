"""Assistant identity and knowledge-base scope, in one place.

Domain-facing strings used to be scattered: the four-item taxonomy was
hardcoded in six files, the persona prompt existed as two independent copies
(``generation/prompts.py`` and ``agent/react_agent.py``), and the collection
name appeared in eight call sites with a default baked into each. Changing
domain meant finding all of them, and missing one produced a silent
mismatch — a UI card keyed off a label literal, or a stale collection.

Import from here instead of retyping any of it.
"""

from __future__ import annotations

import os

# ──────────────────────────────────────────────────────────────
#  Identity
# ──────────────────────────────────────────────────────────────

ASSISTANT_NAME = "法小律"
ASSISTANT_ROLE = "法律智能助手"
ASSISTANT_TAGLINE = "条文可溯源，回答不乱编"

# ──────────────────────────────────────────────────────────────
#  Knowledge-base scope
# ──────────────────────────────────────────────────────────────

# 语料覆盖的法规层级。评测报告、拒答文案、侧边栏都引用这里。
KB_CATEGORIES = ["法律", "行政法规", "司法解释"]
KB_CATEGORY_TEXT = "、".join(KB_CATEGORIES)

# 法律部门，用于说明知识库的覆盖面
KB_LEGAL_FIELDS = [
    "民法商法", "刑法", "行政法", "经济法", "社会法",
    "生态环境法", "诉讼与非诉讼程序法",
]
KB_FIELD_TEXT = "、".join(KB_LEGAL_FIELDS)

# 一句话说明知识库涵盖什么（拒答时告诉用户边界在哪）
KB_SCOPE_SENTENCE = (
    f"当前知识库涵盖{KB_CATEGORY_TEXT}，"
    f"涉及{KB_FIELD_TEXT}等领域"
)

# ──────────────────────────────────────────────────────────────
#  Storage
# ──────────────────────────────────────────────────────────────

DEFAULT_COLLECTION = os.getenv("COLLECTION_NAME", "law_knowledge")
DEFAULT_CORPUS_DIR = os.getenv("CORPUS_DIR", "data/raw/laws")

# ──────────────────────────────────────────────────────────────
#  Disclaimer — 法律问答有责任边界，前端与提示词都要带上
# ──────────────────────────────────────────────────────────────

DISCLAIMER = "回答基于公开法规条文检索生成，不构成法律意见；具体事务请咨询执业律师。"
