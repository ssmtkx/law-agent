"""Query expansion for the colloquial/statutory register gap.

用户提问用口语（「公司拖欠我三个月工资我能主张什么」），法条是书面语
（「用人单位应当按照劳动合同约定和国家规定，向劳动者及时足额支付劳动
报酬」）。两者语域差得太远，向量对不上。

实测（148 条真实评测题，同一批题对比）：

    查询表示                @20      @50     @100
    原始问题              0.480    0.649    0.723
    问题 + 假设性条文      0.757    0.872    0.905    ← 本模块的做法

做法是先用 LLM 生成一段「如果法律要回答这个问题，条文大概会怎么写」的
文字，再把它**拼在原始问题后面**一起嵌入。保留原问题是因为它承载了
提问者的事实描述（时间、金额、当事人），那部分信息生成的法条里没有。

实现上是一层装饰器，与 HybridRetriever / RerankerProcessor 一样暴露
``.query()``，因此可以任意插进检索链，调用方无感。
"""

from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from typing import List, Optional

logger = logging.getLogger(__name__)

# 提示词刻意写得"机械"。用"你是一位立法工作者…"这类身份设定会让模型大量
# 推理（实测 3000–5000 字），token 预算常被耗光、正文为空。直白的改写指令
# 产出质量一样，推理量明显更低。
_PROMPT = """用中文法条的措辞和句式，把下面的问题改写成一段法条正文。只输出法条文字，不要解释、不要标题、不要 markdown，150-250 字。

问题：{question}"""

# 关掉推理：改写是**机械转换**（把口语搬进法条语域），不需要法律分析。
# 实测开着推理时 reasoning_content 长达 7000 字，既慢又会把预算耗光导致
# 正文为空；关掉之后质量不变而快 10–27 倍，且不再有空结果。
#
#     baseline      45.3s  推理 7064字  正文   0字  ⚠
#     effort=none    1.7s  推理    0字  正文 195字  ✓
#
# 保留重试只是为了兜住网络抖动等偶发情况。
_MAX_TOKENS = 4096
_RETRY_MAX_TOKENS = 8192
_NO_REASONING = {"reasoning_effort": "none"}


class HyDEExpander:
    """Rewrite a colloquial query into statute-register text before retrieval.

    Parameters
    ----------
    retriever :
        Wrapped retriever (anything exposing ``.query(query_texts, n_results)``).
    model : str
        LLM used for the rewrite.  Defaults to ``DEEPSEEK_MODEL``.
    enabled : bool
        When False the wrapper is a pass-through — keeps the call sites
        unchanged while allowing the enhancement to be switched off.
    cache_size : int
        In-memory LRU size.  Repeated questions are common in practice.
    timeout : float
        展开失败时**必须**退回原问题：检索链路不能因为一次 LLM 抖动而中断。
    """

    def __init__(
        self,
        retriever,
        model: Optional[str] = None,
        enabled: bool = True,
        cache_size: int = 512,
        timeout: float = 60.0,
        total_budget: float = 900.0,
        verbose: bool = False,
        cache_path: Optional[str] = "data/external/hyde_cache.json",
    ):
        self.retriever = retriever
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
        self.enabled = enabled
        self.timeout = timeout
        # 累计预算：一旦展开总共花掉的时间超过它，**整个改写层自行关闭**，
        # 余下查询直接透传。没有这道闸时，API 变慢会让每条查询都等满超时，
        # 实测一次评测因此静默卡了 56 分钟（进程 CPU 时间零增长）。
        self.total_budget = total_budget
        self.verbose = verbose
        self._spent = 0.0
        self.cache_path = cache_path
        self._dirty = 0
        self._cache: "OrderedDict[str, str]" = OrderedDict()
        self.cache_size = cache_size
        self._client = None
        # 可观测性：展开是否成功直接影响召回质量，出问题时要能看出来
        self.stats = {"expanded": 0, "empty": 0, "error": 0, "cache_hit": 0,
                      "skipped_budget": 0}
        if cache_path:
            n = self.load_cache()
            if n:
                logger.info("HyDE 缓存载入 %d 条", n)

    # ── public API（与下游检索器同构）─────────────────────────

    def query(self, query_texts: List[str], n_results: int = 5, **kwargs) -> dict:
        if not self.enabled or self._spent > self.total_budget:
            if self.enabled and self._spent > self.total_budget:
                self.stats["skipped_budget"] += len(query_texts)
                if self.stats["skipped_budget"] == len(query_texts):
                    logger.warning(
                        "HyDE 累计耗时超过 %.0fs 预算，本次运行改为直接检索",
                        self.total_budget)
            return self.retriever.query(
                query_texts=query_texts, n_results=n_results, **kwargs)
        expanded = [self._augment(q) for q in query_texts]
        return self.retriever.query(
            query_texts=expanded, n_results=n_results, **kwargs)

    # ── internal ──────────────────────────────────────────────

    def _augment(self, question: str) -> str:
        """问题 + 假设性条文；任何失败都退回原问题。"""
        question = (question or "").strip()
        if not question:
            return question

        cached = self._cache.get(question)
        if cached is not None:
            self._cache.move_to_end(question)
            self.stats["cache_hit"] += 1
            return cached

        passage = self._generate(question)
        result = f"{question}\n{passage}" if passage else question
        self._remember(question, result)
        return result

    def _generate(self, question: str) -> str:
        started = time.time()
        try:
            text = self._call(question, _MAX_TOKENS)
            if not text:
                text = self._call(question, _RETRY_MAX_TOKENS)
            if text:
                self.stats["expanded"] += 1
            else:
                self.stats["empty"] += 1
                logger.warning("HyDE 返回空正文，退回原问题：%s", question[:40])
            return text
        except Exception as exc:                     # noqa: BLE001
            self.stats["error"] += 1
            logger.warning("HyDE 展开失败（%s），退回原问题：%s",
                           type(exc).__name__, str(exc)[:80])
            return ""
        finally:
            self._spent += time.time() - started
            done = (self.stats["expanded"] + self.stats["empty"]
                    + self.stats["error"])
            if self.verbose and done % 10 == 0:
                print(f"    … HyDE {done} 次  累计 {self._spent:.0f}s  "
                      f"空 {self.stats['empty']}  失败 {self.stats['error']}",
                      flush=True)

    def _call(self, question: str, max_tokens: int) -> str:
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url=os.getenv("DEEPSEEK_BASE_URL",
                                   "https://api.deepseek.com"),
                timeout=self.timeout,
                # 关掉 SDK 自带重试。默认会在超时后自动重试并退避，
                # 与自己的 timeout 叠乘，单条查询可能等上十几分钟 ——
                # 展开只是增强手段，失败就该立刻退回原问题。
                max_retries=0,
            )
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user",
                       "content": _PROMPT.format(question=question)}],
            max_tokens=max_tokens,
            temperature=0.3,
            **_NO_REASONING,
        )
        return (resp.choices[0].message.content or "").strip()

    def _remember(self, question: str, result: str) -> None:
        self._cache[question] = result
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        self._dirty += 1
        if self.cache_path and self._dirty >= 20:
            self.save_cache()

    # ── 缓存持久化 ────────────────────────────────────────────

    def load_cache(self, path: Optional[str] = None) -> int:
        """Load a previously persisted rewrite cache. Returns entries loaded."""
        import json
        from pathlib import Path
        path = path or self.cache_path
        if not path or not Path(path).exists():
            return 0
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:                            # noqa: BLE001
            return 0
        # 只收非空结果：空的是上一轮失败的残留，留着会永久屏蔽该问题的改写
        loaded = {k: v for k, v in data.items() if v}
        self._cache.update(loaded)
        return len(loaded)

    def save_cache(self, path: Optional[str] = None) -> None:
        import json
        from pathlib import Path
        path = path or self.cache_path
        if not path:
            return
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(
                json.dumps({k: v for k, v in self._cache.items() if v},
                           ensure_ascii=False, indent=1),
                encoding="utf-8")
            self._dirty = 0
        except Exception as exc:                     # noqa: BLE001
            logger.warning("HyDE 缓存写入失败：%s", exc)
