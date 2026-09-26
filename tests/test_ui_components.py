"""UI 渲染冒烟测试。

用一个假的 streamlit 模块把 ``src.ui.components`` 的每个渲染函数跑一遍。
不验证观感（那要看图），只保证：

1. 每个函数都能执行完不抛异常；
2. 生成的 HTML 里引用的 CSS 变量都真实存在 —— 主题改造最容易出的错就是
   改了调色板却漏改组件里的内联 ``var(--...)``，而这种错**不会报错**，
   只会让元素静默失去颜色。

第 2 条正是这个测试存在的理由：黑白色调重构时，``--xz-ink-muted`` 被机械
替换成 ``--ui-ink-muted``，而新调色板里叫 ``--ui-text-muted``。
"""

import re
from pathlib import Path
import sys
import types
import unittest
import unittest.mock


# ══════════════════════════════════════════════════════════
#  最小 streamlit 替身
# ══════════════════════════════════════════════════════════

class _FakeCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeSt(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.rendered: list[str] = []
        self.session_state = types.SimpleNamespace(feedback={})
        # 置为某个按钮的 key 即可模拟"用户点了它"；None 表示没点任何按钮
        self.click_key: str | None = None

    def markdown(self, body, **kw):
        self.rendered.append(str(body))

    def caption(self, body, **kw):
        self.rendered.append(str(body))

    def button(self, *a, key=None, **kw):
        return key == self.click_key

    def expander(self, *a, **kw):
        return _FakeCtx()

    def columns(self, spec, **kw):
        n = spec if isinstance(spec, int) else len(spec)
        return [_FakeCtx() for _ in range(n)]

    def rerun(self):
        pass


_fake = _FakeSt()
sys.modules.setdefault("streamlit", _fake)

from src.ui import components as ui  # noqa: E402
from src.ui.theme import THEME_CSS  # noqa: E402
from src.utils import feedback as feedback_mod  # noqa: E402

_DEFINED_VARS = set(re.findall(r"(--ui-[a-z0-9-]+)\s*:", THEME_CSS))


class ComponentRenderTest(unittest.TestCase):
    def setUp(self):
        _fake.rendered.clear()

    def test_all_components_render(self):
        ui.brand_banner()
        ui.sidebar_header()
        ui.side_label("知识库统计")
        ui.daily_line()
        ui.render_divider("欢迎提问")
        ui.render_welcome()
        ui.render_mode_intro("qa")
        ui.render_mode_intro("agent")
        ui.render_side_tagline()
        ui.render_stamp(1234, "收录条文")
        ui.render_sources(["《中华人民共和国民法典》第五百八十五条 当事人可以约定…"])
        ui.render_thoughts([
            {"type": "thought", "content": "先查违约责任"},
            {"type": "action", "content": "search_knowledge"},
            {"type": "observation", "content": "命中 3 条"},
            {"type": "answer", "content": "完成"},
        ])
        ui.footer(27672)
        self.assertTrue(_fake.rendered, "组件没有产出任何内容")

    def test_unknown_mode_falls_back_without_error(self):
        ui.render_mode_intro("不存在的模式")
        self.assertTrue(_fake.rendered)

    def test_empty_inputs_are_noops(self):
        before = len(_fake.rendered)
        ui.render_sources(None)
        ui.render_sources([])
        ui.render_thoughts(None)
        ui.render_thoughts([])
        self.assertEqual(len(_fake.rendered), before)


class FeedbackButtonWiringTest(unittest.TestCase):
    """点击 👍/👎 必须真的走到 ``feedback.record``。

    回归测试。此前的写法是::

        def _submit(rating):
            st.session_state.feedback[str(msg_index)] = rating
            msg = msg or {}          # ← 让 msg 变成 _submit 的局部变量

    右侧读到的是尚未赋值的局部名，点击即抛 ``UnboundLocalError``。而写入
    session_state 那行在异常**之前**，界面照样显示"已反馈：有帮助"，所以症状是
    **看起来反馈成功了，``data/feedback.jsonl`` 却始终是空的**。

    原来的用例抓不到它：``_FakeSt.button()`` 恒返回 ``False``，整条点击路径
    从未被执行过 —— 这也是本用例存在的理由。
    """

    def setUp(self):
        _fake.rendered.clear()
        _fake.session_state.feedback = {}
        _fake.click_key = None
        self.calls: list[dict] = []
        self.patcher = unittest.mock.patch.object(
            feedback_mod, "record",
            lambda rating, **kw: self.calls.append({"rating": rating, **kw}) or {},
        )
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.addCleanup(lambda: setattr(_fake, "click_key", None))

    @staticmethod
    def _msg() -> dict:
        return {
            "role": "assistant",
            "content": "根据《中华人民共和国民法典》第五百八十五条……",
            "question": "违约金过高能调减吗？",
            "mode": "条文问答",
            "sources": [{"source": "《中华人民共和国民法典》第五百八十五条"}],
            "iterations": None,
        }

    def test_click_records_question_and_answer(self):
        """核心：记录必须带上问题与回答，否则事后无法复盘。"""
        _fake.click_key = "fb_up_0"
        ui.render_feedback(0, self._msg())
        self.assertEqual(len(self.calls), 1, "点击后没有落盘")
        entry = self.calls[0]
        self.assertEqual(entry["rating"], "up")
        self.assertEqual(entry["question"], "违约金过高能调减吗？")
        self.assertIn("民法典", entry["answer"])
        self.assertEqual(
            entry["sources"], ["《中华人民共和国民法典》第五百八十五条"]
        )

    def test_down_click_records_down(self):
        _fake.click_key = "fb_down_0"
        ui.render_feedback(0, self._msg())
        self.assertEqual([c["rating"] for c in self.calls], ["down"])

    def test_msg_defaulting_to_none_does_not_raise(self):
        """``render_feedback(i)`` 的 msg 默认为 None —— 正是 _submit 想兜的分支。"""
        _fake.click_key = "fb_up_3"
        ui.render_feedback(3)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0]["question"], "")
        self.assertEqual(self.calls[0]["sources"], [])

    def test_no_click_records_nothing(self):
        ui.render_feedback(0, self._msg())
        self.assertEqual(self.calls, [])


class ThemeConsistencyTest(unittest.TestCase):
    """组件里引用的每个 CSS 变量都必须在主题里定义。"""

    def test_components_only_use_defined_css_vars(self):
        source = Path("src/ui/components.py").read_text(encoding="utf-8")
        icons = Path("src/ui/icons.py").read_text(encoding="utf-8")
        used = set(re.findall(r"var\((--ui-[a-z0-9-]+)\)", source + icons))
        self.assertEqual(
            sorted(used - _DEFINED_VARS), [],
            "组件引用了未定义的 CSS 变量 —— 不会报错，元素会静默失去样式",
        )

    def test_every_component_class_has_styles(self):
        comp = Path("src/ui/components.py").read_text(encoding="utf-8")
        classes = set()
        for attr in re.findall(r'class="([^"]+)"', comp):
            classes.update(c for c in attr.split() if c.startswith("ui-"))
        styled = set(re.findall(r"\.(ui-[a-z0-9-]+)", THEME_CSS))
        # ui-ic 由 icons.py 产出，theme 里也有定义
        self.assertEqual(sorted(classes - styled), [],
                         "组件用了没有对应样式的类名")

    def test_no_legacy_theme_tokens(self):
        """旧的 .xz-* / --xz-* 不应残留 —— 残留即为漏改。"""
        for path in ("src/ui/theme.py", "src/ui/components.py",
                     "src/ui/icons.py"):
            source = Path(path).read_text(encoding="utf-8")
            leftovers = re.findall(r"--xz-[a-z0-9-]+|\.xz-[a-z0-9-]+", source)
            self.assertEqual(leftovers, [], f"{path} 残留旧主题标记")

    def test_monochrome_has_no_decorative_gradients(self):
        """简约黑白主题不应有纸纹或渐变。"""
        self.assertNotIn("paper-texture", THEME_CSS)
        self.assertNotIn("linear-gradient", THEME_CSS)
        self.assertNotIn("radial-gradient", THEME_CSS)


if __name__ == "__main__":
    unittest.main()
