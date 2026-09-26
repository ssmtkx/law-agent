"""Unit tests for statute parsing and article chunking.



This module is the foundation of the whole legal pipeline and is where the

subtle failures live — every case below is a bug that was actually observed

or a boundary that the corpus exercised.

"""



import unittest



from src.utils.cn_numeral import article_number, cn_to_int, parse_article_no

from src.ingestion.law_parser import (

    LawArticleSplitter, LawTextParser, clean_title, sanitize_meta,

)



CIVIL_CODE = "中华人民共和国民法典"





class NumeralTest(unittest.TestCase):

    def test_additive_and_multiplicative_forms(self):

        for text, want in [("十", 10), ("三十", 30), ("一百零二", 102),

                           ("一百一十", 110), ("一千二百", 1200),

                           ("二百零一", 201), ("九百九十九", 999),

                           ("一千二百六十", 1260)]:

            self.assertEqual(cn_to_int(text), want, text)



    def test_article_no_splits_sub_article(self):

        self.assertEqual(parse_article_no("第一百零二条"), (102, 0))

        self.assertEqual(parse_article_no("第十七条之一"), (17, 1))

        self.assertEqual(parse_article_no("第十七条之二"), (17, 2))



    def test_whole_heading_is_not_parsed_as_one_numeral(self):

        """把整串递给数字转换会把结尾的「一」算进去，17 变成 11。



        实测刑法因此看起来"条号断裂"，所以「之N」必须先拆。

        """

        self.assertEqual(article_number("第十七条之一"), 17)

        self.assertNotEqual(article_number("第十七条之一"), 11)



    def test_arabic_digits_accepted(self):

        self.assertEqual(article_number("577"), 577)

        self.assertEqual(article_number(577), 577)

        self.assertEqual(article_number("第577条"), 577)



    def test_unparseable_is_none(self):

        self.assertIsNone(article_number(""))





class CleanTitleTest(unittest.TestCase):

    def test_strips_search_highlight_markup(self):

        """列表接口在带搜索词时会用 <em> 包住命中片段。"""

        raw = ("<em class='highlight'>中华人民共和国</em>"

               "<em class='highlight'>刑法</em>")

        self.assertEqual(clean_title(raw), "中华人民共和国刑法")





class ExtractArticlesTest(unittest.TestCase):

    def test_article_bodies_and_numbers(self):

        paras = [

            "中华人民共和国某法",             # 标题

            "（2020年1月1日通过）",           # 公布信息

            "第一条　为了规范某事项，制定本法。",

            "第二条　本法所称某事项，是指……。",

        ]

        arts = LawTextParser.extract_articles(paras)

        self.assertEqual([a["no"] for a in arts], ["第一条", "第二条"])

        self.assertEqual([a["num"] for a in arts], [1, 2])

        self.assertIn("制定本法", arts[0]["text"])



    def test_preamble_and_toc_are_dropped(self):

        """标题/目录/公布令不属于条文，由 chunk 头部的元数据承担。"""

        paras = [

            "中华人民共和国某法",

            "目　　录",

            "第一章　总　　则",

            "第一条　内容甲。",

        ]

        arts = LawTextParser.extract_articles(paras)

        self.assertEqual(len(arts), 1)

        self.assertEqual(arts[0]["no"], "第一条")



    def test_body_reference_is_not_mistaken_for_an_article_start(self):

        """条首必须行首锚定。



        不锚定的话，正文里的「依照本法第三百零二条规定」也会被当成条首 ——

        实测海商法宽松匹配 382 处 vs 锚定后 310 条。

        """

        paras = [

            "第一条　依照本法第三百零二条规定处理。",

            "第二条　实际只有两条。",

        ]

        arts = LawTextParser.extract_articles(paras)

        self.assertEqual([a["no"] for a in arts], ["第一条", "第二条"])



    def test_container_tracked_for_each_article(self):

        paras = [

            "第一章　总　　则",

            "第一条　内容甲。",

            "第二章　分　　则",

            "第二条　内容乙。",

            "第一节　细　　则",

            "第三条　内容丙。",

        ]

        arts = LawTextParser.extract_articles(paras)

        self.assertEqual(arts[0]["container"], "第一章 总 则")

        self.assertEqual(arts[1]["container"], "第二章 分 则")

        self.assertEqual(arts[2]["container"], "第一节 细 则")



    def test_wrapped_lines_join_the_current_article(self):

        paras = [

            "第一条　这是被换行拆开的",

            "条文的后半句。",

            "第二条　另一条。",

        ]

        arts = LawTextParser.extract_articles(paras)

        self.assertIn("条文的后半句", arts[0]["text"])

        self.assertEqual(len(arts), 2)



    def test_sub_article_keeps_base_number(self):

        paras = ["第十七条　内容甲。", "第十七条之一　内容乙。"]

        arts = LawTextParser.extract_articles(paras)

        self.assertEqual(arts[1]["num"], 17)

        self.assertEqual(arts[1]["sub"], 1)





class FallbackChunkingTest(unittest.TestCase):

    """不用「第X条」编号的文件需要兜底切分，否则完全检索不到。



    这类文件约占语料 18%：早期规范性文件与修改/补充决定用「一、」或

    「（一）」编号，批复/复函则是无编号的散文。

    """



    def test_enum_markers_are_used_when_no_articles(self):

        paras = [

            "最高人民法院关于某问题的补充安排",

            "法释〔2020〕13号",

            "一、本安排自公布之日起施行。",

            "二、将《安排》第一条修改为：“……”。",

            "三、本安排由最高人民法院负责解释。",

        ]

        arts = LawTextParser.extract_articles(paras)

        self.assertEqual([a["no"] for a in arts], ["一", "二", "三"])

        self.assertEqual(arts[0]["chunking"], "enum")



    def test_paren_markers_used_for_older_documents(self):

        paras = [

            "保护海底电缆规定",

            "(1973年5月30日发布)",

            "(一) 保护海底电缆是加强海防建设的重要措施。",

            "(二) 舰艇、商船、外轮等舰船锚泊，要避开敷设有海底电缆的禁区。",

        ]

        arts = LawTextParser.extract_articles(paras)

        self.assertEqual(len(arts), 2)

        self.assertEqual(arts[0]["chunking"], "paren")



    def test_paragraph_fallback_drops_the_masthead(self):

        """版头（公告/文号/通过日期）不含内容，不应进入第一个 chunk。



        只按长度判断会漏掉「…已于…会议通过，现予公布」这种既长又是版头的段落。

        """

        paras = [

            "中华人民共和国最高人民法院",

            "公    告",

            "最高人民法院《关于某问题的批复》已于2013年9月9日由审判委员会"

            "第1590次会议通过，现予公布，自2013年9月12日起施行。",

            "2013年9月12日",

            "法释〔2013〕23号",

            "四川省高级人民法院：",

            "你院《关于某问题的请示》收悉。经研究，批复如下：依据《中华人民"

            "共和国劳动争议调解仲裁法》第二十七条的规定，当事人自知道或者"

            "应当知道其权利被侵害之日起一年内提出。",

        ]

        arts = LawTextParser.extract_articles(paras)

        self.assertEqual(len(arts), 1)

        self.assertEqual(arts[0]["chunking"], "paragraph")

        self.assertIn("你院", arts[0]["text"])

        self.assertNotIn("公    告", arts[0]["text"])

        self.assertNotIn("法释〔2013〕23号", arts[0]["text"])



    def test_single_stray_marker_does_not_fabricate_structure(self):

        """散文里偶然出现一个「一、」不应被当成顶层编号。"""

        paras = [

            "关于某事项的批复",

            "你院请示收悉。经研究，一、该问题应当按照下列原则处理："

            "具体而言，需要结合案件事实综合判断，不能一概而论。",

        ]

        arts = LawTextParser.extract_articles(paras)

        self.assertEqual(len(arts), 1)

        self.assertEqual(arts[0]["chunking"], "paragraph")



    def test_real_articles_are_marked_as_such(self):

        """有「第X条」时不得走兜底，且要标成 article。"""

        paras = [

            "第一条　内容甲。",

            "一、这是条文内部的编号，不是顶层单位。",

            "第二条　内容乙。",

        ]

        arts = LawTextParser.extract_articles(paras)

        self.assertEqual([a["no"] for a in arts], ["第一条", "第二条"])

        self.assertTrue(all(a["chunking"] == "article" for a in arts))





class FallbackSplitterTest(unittest.TestCase):

    def test_chunking_mode_lands_in_metadata(self):

        law = {

            "title": "保护海底电缆规定", "bbbs": "x", "category": "行政法规",

            "sxrq": "1973-05-30", "sxx": 3,

            "articles": [

                {"no": "(一)", "num": 1, "sub": 0, "container": "",

                 "text": "保护海底电缆是加强海防建设的重要措施。",

                 "chunking": "paren"},

            ],

        }

        meta = LawArticleSplitter().split(law)[0]["metadata"]

        self.assertEqual(meta["chunking"], "paren")

        self.assertIn("保护海底电缆规定", meta["law_title"])





class LegacyDocTest(unittest.TestCase):

    """老式 .doc（OLE2）的处理。



    源站对一批司法解释返回老式 .doc，而配套 PDF 是扫描件 —— 两条常规路径

    都失败，实测因此漏掉 56 部法规，恰好是最高频引用的那批（民间借贷、

    买卖合同、医疗损害……）。第三级用 antiword 兜底。

    """



    OLE2_MAGIC = bytes([0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1])



    def test_docx_parser_rejects_ole2_with_a_clear_message(self):

        """直接交给 zipfile 只会得到含糊的"没有 word/document.xml"，

        那样的报错会把人引向错误的排查方向。"""

        with self.assertRaises(ValueError) as ctx:

            LawTextParser.parse_docx(self.OLE2_MAGIC + bytes(64))

        self.assertIn("OLE2", str(ctx.exception))



    def test_legacy_doc_degrades_gracefully(self):

        """antiword 是可选依赖：没装就返回空，不能抛异常打断采集。"""

        out = LawTextParser.parse_legacy_doc(b"not a real doc")

        self.assertEqual(out, [])



    def test_parse_dispatches_by_format(self):
        """三种 fmt 走各自的路径，且失败方式不同（这是有意的）。

        docx / pdf 解析失败会抛异常（由调用方决定是否回落下一级）；
        doc 走 antiword，它是**优雅降级**——提取不出就返回空列表，
        因为 antiword 是可选依赖，缺了不能打断整条采集流程。
        """
        bad = self.OLE2_MAGIC + bytes(64)
        with self.assertRaises(ValueError):
            LawTextParser.parse(bad, fmt="docx")          # 明确认出是 .doc
        with self.assertRaises(Exception):
            LawTextParser.parse(bad, fmt="pdf")           # fitz 打不开
        self.assertEqual(LawTextParser.parse(bad, fmt="doc"), [])   # 优雅降级


class SanitizeMetaTest(unittest.TestCase):

    def test_none_becomes_empty_string(self):

        """Chroma 只接受 str/int/float/bool，None 会直接抛 TypeError。



        实测 117 部法规的施行日期在源接口里是 null，dict.get(k, "") 拿到的是

        None 而不是默认值（键存在）。

        """

        self.assertEqual(sanitize_meta({"sxrq": None})["sxrq"], "")



    def test_scalars_pass_through(self):

        meta = {"a": "x", "b": 1, "c": 1.5, "d": True}

        self.assertEqual(sanitize_meta(meta), meta)



    def test_non_scalars_are_stringified(self):

        self.assertEqual(sanitize_meta({"x": ["a"]})["x"], "['a']")





class SplitterTest(unittest.TestCase):

    LAW = {

        "title": CIVIL_CODE, "bbbs": "abc123", "category": "法律",

        "subcategory": "民法商法", "gbrq": "2020-05-28", "sxrq": "2021-01-01",

        "sxx": 3,

        "articles": [

            {"no": "第五百八十五条", "num": 585, "sub": 0,

             "container": "第三编 合同", "text": "当事人可以约定违约金。"},

        ],

    }



    def test_chunk_header_carries_law_and_article(self):

        """法条多以「本法所称…」开头，脱离法名与条号无法理解。"""

        chunks = LawArticleSplitter().split(self.LAW)

        self.assertEqual(len(chunks), 1)

        text = chunks[0]["text"]

        self.assertIn(f"《{CIVIL_CODE}》", text)

        self.assertIn("第五百八十五条", text)

        self.assertIn("当事人可以约定违约金", text)



    def test_metadata_carries_the_matching_keys(self):

        meta = LawArticleSplitter().split(self.LAW)[0]["metadata"]

        self.assertEqual(meta["law_title"], CIVIL_CODE)

        self.assertEqual(meta["article_no"], "第五百八十五条")

        self.assertEqual(meta["article_num"], 585)

        self.assertEqual(meta["category"], "法律")



    def test_metadata_is_chroma_safe_when_dates_are_missing(self):

        law = dict(self.LAW, sxrq=None, gbrq=None)

        for chunk in LawArticleSplitter().split(law):

            for value in chunk["metadata"].values():

                self.assertIsInstance(value, (str, int, float, bool))



    def test_long_article_splits_but_keeps_article_identity(self):

        law = dict(self.LAW, articles=[

            {"no": "第一条", "num": 1, "sub": 0, "container": "",

             "text": "第一句。" * 200},

        ])

        chunks = LawArticleSplitter(max_chars=300).split(law)

        self.assertGreater(len(chunks), 1)

        # 所有子块共享同一 (法名, 条号)，评测按此匹配

        for chunk in chunks:

            self.assertEqual(chunk["metadata"]["article_no"], "第一条")

            self.assertLessEqual(len(chunk["text"]), 400)





if __name__ == "__main__":

    unittest.main()

