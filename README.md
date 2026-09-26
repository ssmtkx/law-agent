# ⚖ 法小律 · 中国法律法规智能问答

> 语料直接采集自国家法律法规数据库，按**条文**切分；
> 查询先用 LLM 改写进法条语域再检索；评测集取自**语料之外的独立来源**

**「法小律」** 检索现行有效的法律、行政法规与司法解释条文，依据条文回答问题，
并标注具体法规与条号，供查证。

**核心承诺**：只引用检索到的条文，不凭记忆编造条号；检索不到就如实说没有。

---

## ✨ 特性

- 📚 **真实语料**：2,192 部法规 / 86,403 条条文，来自 [国家法律法规数据库](https://flk.npc.gov.cn)
- ✂️ **按条切分**：中国法规天然以「第X条」为语义单元，一条一个知识块
- 🧬 **上下文注入**：每块带《法规名》（公布/施行日期）第X条 —— 法条多以「本法所称…」开头，脱离法名无法理解
- 🗣 **查询改写**：用户口语与法条书面语存在语域差，先用 LLM 把问题搬进法条语域再检索（实测召回 @20 从 0.480 提升到 0.757）
- 🔍 **混合检索**：BM25 关键词 + BGE 语义向量，按名次做 RRF 融合
- 📊 **可信评测**：评测集取自语料之外的独立来源，并单独报告**语料覆盖度**
- ⚡ **低延迟**：端到端 9–17 秒（含两次 LLM 调用）
- 🔁 **反馈闭环**：👍👎 连同问题与回答落盘，`scripts/feedback_stats.py` 可回看差评明细

---

## 🏗 系统架构

```
┌─ 离线：语料与索引 ────────────────────────────────────────┐
│                                                          │
│  flk.npc.gov.cn（国家法律法规数据库）                      │
│        │  scripts/crawl_laws.py        WAF 自适应/熔断/续传 │
│        ▼                                                  │
│  data/raw/laws/*/*.json     2,192 部 / 86,403 条           │
│        │  src/ingestion/law_parser.py                     │
│        │    · 条文解析（行首锚定「第X条」+ 编/章/节追踪）   │
│        │    · 兜底切分（一、/（一）/段落 三级降级）         │
│        ▼                                                  │
│  带上下文头的 chunk：                                      │
│  《法规名》（公布/施行日期）第X条 \n 条文正文              │
│        │  src/indexing/                                   │
│        ▼                                                  │
│  chroma_db（BGE 向量） + bm25_index.pkl（jieba 倒排）      │
│  57,066 条 / 1,504 部现行有效法规                          │
└──────────────────────────────────────────────────────────┘

┌─ 在线：问答 ──────────────────────────────────────────────┐
│                                                          │
│  用户提问                                                 │
│        │  app.py / run.py / run_agent.py                 │
│        ▼                                                  │
│  ┌──────────────────┐   ┌────────────────────────────┐   │
│  │ 条文问答          │   │ Agent 法律分析              │   │
│  │ rag_pipeline.py  │   │ react_agent.py             │   │
│  └────────┬─────────┘   └──────────┬─────────────────┘   │
│           └──────────┬─────────────┘                     │
│                      ▼                                   │
│  ① HyDEExpander     把口语问题改写成法条语域               │
│        │             （reasoning_effort=none，约 3.5 秒）  │
│        ▼                                                  │
│  ② HybridRetriever  BM25 + 语义 → RRF 融合 → Top-K        │
│        ▼                                                  │
│  ③ LLM 生成         reasoning_effort=low，约 9–14 秒       │
│                     ↓                                     │
│        三段式输出：法律依据 → 适用分析 → 风险提示          │
└──────────────────────────────────────────────────────────┘
```

### 检索链路的三个取舍（都是实测逼出来的）

| 取舍 | 结论 | 依据 |
|---|---|---|
| **查询改写** | **默认开** | 同一批 148 条评测题，语义召回 @20 从 0.480 → 0.757 |
| **CrossEncoder 精排** | **默认关** | 两次实测均为负作用：用原问题打分时 Hit@5 从 0.567 压回 0.233；改用**改写文本**打分后仍是 0.520 → 0.280。并已排除截断（有效长度 8192，输入约 500–600 token）。它同时是最慢的一环（约 10 秒/条） |
| **RRF 融合** | 用，但**不是性能来源** | 与旧的 min-max 归一化相比召回无变化，选它是因为不需要分数校准、代码更短 |

精排可用 `factory.build_retriever(rerank=True)` 打开，用于对比实验。

---

## 🚀 快速开始

> ⚠️ **先读这一条：本仓库不含语料与索引。**
>
> `chroma_db/`（548 MB）、`data/raw/`（2,192 部法规，46 MB）、`data/bm25_index.pkl`
> （91 MB）**均未进版本库** —— 几十 MB 的二进制不该进 Git，这是有意的设计。
>
> 所以**直接 clone 下来是跑不起来的**，必须先自备数据，二选一：
>
> | 方式 | 做法 | 代价 |
> |---|---|---|
> | **自己采集**（默认路径） | 走下面第 3、4 步：`crawl_laws.py` → `build_index.py` | 站点有 WAF 限流，全量采集约数小时；建索引约 34 分钟（CPU） |
> | **自带语料** | 按 `data/raw/laws/<分类>/<bbbs>.json` 的结构放入你自己的法规 JSON，再跑第 4 步 | 需自己拿到符合结构的语料 |
>
> 只想**读代码或读文档**的话不必跑起来 —— 建议直接看
> [源码阅读指南.md](docs/源码阅读指南.md) 与 [问题与解决方案.md](docs/问题与解决方案.md)。

### 1. 环境（推荐 uv）

uv 不需要"激活环境"这一步，也就避开了 conda 在 Git Bash 下最常见的一个坑
（见下面的"环境踩坑"）。

```bash
# 安装 uv（Windows PowerShell）
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# 或 pip install uv

cd law-agent
uv sync                 # 按 pyproject.toml 创建 .venv 并装依赖
uv run streamlit run app.py
```

依赖定义在仓库根的 [`pyproject.toml`](pyproject.toml)，版本与 `requirements.txt`
一致（均为实测跑通的锁定版本）。**为什么锁死而不是写 `>=` 下限，理由写在该文件
顶部的注释里 —— 这里不再抄一份。**

> ⚠️ **别在文档里再内联一份 pyproject。** 这里曾经抄过一版"最小写法"，它漏掉了
> 把 `torch` 显式列为**直接依赖** —— 而 `[tool.uv.sources]` 的覆盖只对直接依赖
> 生效，照抄会从 PyPI 拉到约 2.5GB 的 CUDA 版 torch（CPU 环境用不上）。
> 依赖清单只有 `pyproject.toml` 一个来源。
>
> 迁到 uv 之后，`requirements.txt` 可以用
> `uv export --format requirements-txt > requirements.txt` 自动生成，避免手工同步。

### 1b. 或者用 pip

不想装 uv 也可以，`requirements.txt` 里的版本与 `pyproject.toml` 完全一致：

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash；cmd 用 .venv\Scriptsctivate
pip install -r requirements.txt
streamlit run app.py
```

> `requirements.txt` 里已带 `--extra-index-url` 指向 PyTorch 的 CPU 源，并锁定
> `torch==2.13.0+cpu` —— 这个本地版本串只存在于该源上，因此会强制走 CPU 版，
> 不会拉 2.5GB 的 CUDA 包。

### 2. 配置

```bash
cp .env.example .env    # 填入 DEEPSEEK_API_KEY
```

**务必看 `.env.example` 里关于模型的说明**——选错模型会返回空答案。

### 3. 采集语料

```bash
uv run python scripts/crawl_laws.py --categories law,admin,judicial
uv run python scripts/crawl_status.py --watch   # 另开终端看进度
```

先试跑一小批：`uv run python scripts/crawl_laws.py --categories law --limit 20`

### 4. 建立索引

```bash
uv run python scripts/build_index.py --rebuild
```

### 5. 启动

```bash
uv run streamlit run app.py     # http://localhost:8501
uv run python run.py            # 或 CLI
uv run python run_agent.py
```

### 环境踩坑（conda 用户）

**Git Bash 下 `conda activate` 会直接报错**，因为 conda 默认只为 cmd/PowerShell
配置过：

```
CondaError: Run 'conda init' before 'conda activate'
```

激活没生效时，`streamlit` 命令会解析到 **base 环境**——而 base 里装了 streamlit
但**缺 chromadb**，于是症状是"**服务能启动，但一打开页面就报错**"，很容易被误判成
代码问题。三个解法：

```bash
# A. 用 conda run（不需改配置，但会缓冲输出）
conda run -n paper_agent streamlit run app.py

# B. 直接给全路径（输出最干净）
D:/anaconda-environment/envs/paper_agent/python.exe -m streamlit run app.py

# C. 永久修好
conda init bash          # 然后关掉终端重开
conda activate paper_agent
```

### 依赖说明

`antiword` 是**系统命令不是 Python 包**，只在**重新采集语料**时用得上（解源站返回的
老式 `.doc` 司法解释）。缺了不影响现有索引与问答——代码里是优雅降级，那批文档会
退回成采集失败。

---

## 📊 评测

### 评测集从哪来

法律领域的评测集**不能**照着语料自己写 —— 用检索到的条文反推问题，得到的是
循环论证，指标不携带信息。本项目使用**独立于采集语料**的来源：

| 用途 | 来源 |
|---|---|
| 主评测 | [EQUALS](https://github.com/andongBlue/EQUALS)：6,914 个 (问题, 法条, 答案) 三元组；问题来自真实法律问答社区，答案由法学标注者标注 |
| 拒答 / 幻觉 | 自建域外问题集（GDPR、美国专利法、自动驾驶责任等） |
| 语料保真度（**待实现**） | [verified-chinese-law-kb](https://github.com/vickywu97/verified-chinese-law-kb)：逐字核验的条文，可用于校验 docx 解析结果与权威文本是否一致。**该校验脚本尚未编写**，语料目前只经过抽样人工检查 |

```bash
# 下载 EQUALS 后放到 data/external/equals/
python scripts/build_eval_set.py          # 转换 + 核对语料覆盖度
python eval_rag.py --dataset data/eval_law.json --sample 100
python eval_rag.py --dataset data/eval_law.json --hyde      # 带查询改写
```

### 报告里先看覆盖度

```
  语料覆盖度（先看这个，再看指标）
  黄金条文覆盖率: 91.7%   （评测集要求的条文里，语料实际收录的比例）
  可答条目: 6337/6914
```

覆盖率低说明**语料不全**，此时任何 Hit Rate 都只是语料完整度的代理指标，
不是检索能力的度量。

### 实测结果

**配置对比**（30 条评测题，同一批题）：

| 配置 | Hit@5 |
|---|---|
| 无改写 + 无精排 | 0.133 |
| 无改写 + 精排 | 0.233 |
| **有改写 + 无精排** | **0.567** ← 生产配置 |
| 有改写 + 精排（改写文本打分） | 0.280 |

**召回率 vs top_k**（150 条评测题）：

| k | 1 | 3 | 5 | 8 | **12** | 16 | 20 | 30 |
|---|---|---|---|---|---|---|---|---|
| Hit Rate | 0.360 | 0.453 | 0.540 | 0.613 | **0.667** | 0.687 | 0.707 | 0.800 |
| MRR | 0.360 | 0.401 | 0.422 | 0.433 | **0.438** | 0.439 | 0.440 | 0.445 |

> Hit Rate 随 k 持续上升，但 **MRR 从 k=8 起平在 0.44** —— 说明加大 k 是在
> **绕开**排序问题，不是在解决它。排序问题的正解（CrossEncoder 精排）在两种
> 打分信号下均已实测为负作用并被移除。

**完整数字、各实验的样本量与标准误，以及「这些数字不能说明什么」，见
[评测结果.md](docs/评测结果.md)。** 排查过程见
[问题与解决方案.md](docs/问题与解决方案.md)（37 条）。

## 📁 项目结构

```
├── app.py / run.py / run_agent.py    # 入口：Web / CLI / Agent CLI
├── eval_rag.py / eval_agent.py       # 评测 CLI
├── pyproject.toml                    # 依赖定义（uv；锁定理由见文件顶部注释）
├── requirements.txt                  # 同一批版本（pip 路径）
│
├── src/                              # 约 4,700 行
│   ├── domain.py                     # 身份、语料范围、集合名（单一来源）
│   ├── ingestion/
│   │   ├── law_parser.py             #   docx/PDF → 条文 → 带上下文的 chunk
│   │   └── splitter.py / parser.py   #   通用分块与 PDF 解析
│   ├── indexing/
│   │   ├── bm25_index.py             #   自实现 BM25（Counter 词频表）
│   │   ├── vector_store.py           #   Chroma + 分页取数
│   │   └── indexer.py                #   法规 → 索引
│   ├── retrieval/
│   │   ├── query_expander.py         #   ⭐ HyDE 查询改写
│   │   ├── hybrid_retriever.py       #   RRF 融合 + 相关性下限
│   │   ├── reranker.py               #   CrossEncoder（默认关闭）
│   │   └── factory.py                #   检索链组装
│   ├── generation/
│   │   ├── prompts.py                #   身份 / 闲聊 / RAG / 拒答 模板
│   │   └── rag_pipeline.py           #   LawAgent
│   ├── agent/                        #   ReAct 循环 + 3 个工具
│   ├── eval/evaluator.py             #   指标 + 覆盖度 + 拒答率
│   └── ui/                           #   简约黑白主题
│
├── scripts/                          # 约 2,100 行
│   ├── crawl_laws.py                 # 采集（WAF 自适应）
│   ├── crawl_status.py               # 采集进度监控
│   ├── build_index.py                # 建立索引
│   ├── build_eval_set.py             # 构建评测集
│   ├── diagnose_recall.py            # 召回诊断
│   ├── test_hyde.py                  # 查询改写对比实验
│   ├── tune_threshold.py             # 阈值标定
│   └── feedback_stats.py             # 用户反馈汇总
│
├── tests/                            # 104 个测试
└── docs/
    ├── 源码阅读指南.md
    ├── 问题与解决方案.md              ⭐ 本次开发踩过的坑（37 条）
    └── 评测结果.md                    ⭐ 实测数字 + 它们不能说明什么
```

---

## 🛠 技术栈

| 层次 | 技术 | 说明 |
|---|---|---|
| **LLM** | DeepSeek-V4-Pro | 推理模型，`reasoning_effort` 控制推理量 |
| **Embedding** | BAAI/bge-small-zh-v1.5 | 512 token 上限，实测仅 0.8% 的条文被截断 |
| **向量库** | ChromaDB | 嵌入式；**默认 L2 距离而非余弦**，换算见 `hybrid_retriever._to_similarity` |
| **关键词检索** | BM25 + jieba | 自实现，Counter 词频表 + schema 版本校验 |
| **融合** | RRF (k=60) | Cormack et al., SIGIR 2009 |
| **前端** | Streamlit | 简约黑白主题 |

---

## ⚙️ 环境变量

```bash
DEEPSEEK_API_KEY=sk-xxxxx        # 必填
DEEPSEEK_MODEL=deepseek-v4-pro   # 见 .env.example 的说明，选错会返回空答案
HF_ENDPOINT=https://hf-mirror.com
COLLECTION_NAME=law_knowledge
HYDE_ENABLED=1                   # 设为 0 关闭查询改写
```

---

## 🔧 常见问题

**回答是空的**

`DEEPSEEK_MODEL` 选错了，或 `reasoning_effort` 没设。推理模型的推理过程会消耗
token 预算，预算被耗光时正文为空且 `finish_reason="length"`。详见
[问题与解决方案.md](docs/问题与解决方案.md)。

**启动卡十几分钟，然后报 `Unrecognized processing class`**

HuggingFace 缓存里的**过期否定标记**（`.no_exist/`）在作祟：

```bash
find ~/.cache/huggingface/hub -maxdepth 2 -name ".no_exist" -type d -exec rm -rf {} +
```

清除后模型可完全离线加载。想固定为离线模式可设 `HF_HUB_OFFLINE=1`。

**采集一直显示"冷却中"**

站点有 WAF 限流，属正常行为。连续被拦 3 次且中间无成功请求时采集器会干净退出
（而不是空烧几小时），稍后重跑即可续传。

---

## ⚠️ 免责声明

本项目回答基于公开法规条文检索生成，**不构成法律意见**。法律结论高度依赖具体
事实，涉及实际事务请咨询执业律师。

## 📄 数据来源与许可

法律法规文本依据《中华人民共和国著作权法》第五条不适用著作权保护，可自由分发。
本项目仅采集**正式文本**，不含司法解释的学术解读与评注（那部分有版权）。

代码以 MIT 许可发布。

---

## 📝 后续规划

- [x] 全量语料采集（2,192 / 2,195 部，99.9%）
- [ ] **拒答机制重建** —— 唯一未解决的架构问题：查询改写把任何问题都洗成法条腔，
      摧毁了"语料能不能答"的判别信号；余弦阈值与精排两种方案均已实测排除（见问题文档）
- [x] 用户反馈闭环（👍👎 落盘到 `data/feedback.jsonl`，含问题与回答；`scripts/feedback_stats.py` 查看）
- [ ] 法律时效性：同一法规的多版本按生效日期检索（681 部旧版本已保留在磁盘，未索引）
- [ ] 判例检索（裁判文书）
