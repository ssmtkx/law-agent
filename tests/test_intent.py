"""Unit tests for LawAgent intent detection.

The rule under test is domain-free (it only knows greetings and identity
phrases), so what these tests pin is the *boundary*: a professional question
that happens to open with a greeting must not be routed to chit-chat.
"""

import os
import unittest

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-for-unit-tests")

from src.generation.rag_pipeline import LawAgent  # noqa: E402


class IntentDetectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 跳过 __init__，避免创建 OpenAI client，直接测试纯逻辑方法
        cls.agent = LawAgent.__new__(LawAgent)

    def test_pure_greeting_is_chat(self):
        for q in ["你好", "hello", "你是谁", "介绍一下你自己", "谢谢", "再见"]:
            self.assertTrue(self.agent._is_chat(q), q)

    def test_short_identity_question_is_chat(self):
        for q in ["你好，你是谁", "你是谁啊", "你叫什么名字"]:
            self.assertTrue(self.agent._is_chat(q), q)

    def test_legal_question_with_greeting_is_not_chat(self):
        for q in [
            "你好，劳动合同违约金怎么算？",
            "你好请问民法典对诉讼时效是怎么规定的",
            "谢谢，麻烦讲一下正当防卫的认定标准",
        ]:
            self.assertFalse(self.agent._is_chat(q), q)

    def test_legal_terms_are_not_chat(self):
        for q in ["正当防卫", "善意取得", "公司拖欠工资怎么办", "房屋租赁合同解除"]:
            self.assertFalse(self.agent._is_chat(q), q)


if __name__ == "__main__":
    unittest.main()
