"""从国家法律法规数据库（flk.npc.gov.cn）采集法规正文，产出结构化 JSON。

正文只能通过 docx 下载获得 —— 详情接口的 ``content`` 字段实测恒为空，
所以流程是：枚举 → 取签名下载地址 → 下载 docx → 解析条文 → 落盘。

Usage::

    python scripts/crawl_laws.py                      # 采集全部三个类别
    python scripts/crawl_laws.py --categories law     # 只采集法律
    python scripts/crawl_laws.py --limit 20           # 每个叶类别只取 20 部（试跑）
    python scripts/crawl_laws.py --rate 2             # 提高限流间隔到 2s/请求

采集可断点续传：已存在的 ``{bbbs}.json`` 会直接跳过，中断后重跑即可。

输出::

    data/raw/laws/{law,admin,judicial}/{bbbs}.json
    data/meta/catalog.json        法规清单（含效力状态）
    data/meta/crawl_report.json   采集统计与失败清单
"""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.law_parser import LawTextParser, clean_title  # noqa: E402

BASE = "https://flk.npc.gov.cn"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": f"{BASE}/",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

# 站点前置了网宿 WAF：连续请求到一定量后会改成返回 302 + Set-Cookie
# (wzws_cid)，不带这个 cookie 就一直是 302/307。带 CookieJar 的 opener
# 会在首次响应里自动存下它，之后原样重发一次即可拿到 200。
_COOKIE_JAR = http.cookiejar.CookieJar()
_OPENER = urllib.request.build_opener(
    urllib.request.HTTPCookieProcessor(_COOKIE_JAR),
)

# 叶节点分类码。父节点（如 101 法律）传进接口会返回 total=0，必须用叶节点。
CATEGORIES: dict[str, tuple[str, list[tuple[str, int]]]] = {
    "law": ("法律", [
        ("宪法相关法", 110), ("民法商法", 120), ("行政法", 130),
        ("经济法", 140), ("社会法", 150), ("生态环境法", 155),
        ("刑法", 160), ("诉讼与非诉讼程序法", 170),
        ("法律解释", 180), ("修正案", 195),
    ]),
    "admin": ("行政法规", [("行政法规", 210)]),
    "judicial": ("司法解释", [
        ("高法司法解释", 320), ("高检司法解释", 330), ("联合发布司法解释", 340),
    ]),
}

STATUS_LABEL = {1: "已废止", 2: "已修改", 3: "现行有效", 4: "尚未生效"}

DEFAULT_OUT = Path("data/raw/laws")
META_DIR = Path("data/meta")
HEARTBEAT_PATH = META_DIR / "crawl_heartbeat.json"

# ── 心跳：让外部能区分「正在冷却」和「进程已死」 ──
# 采集器会长时间静默（WAF 冷却可达 15 分钟），只看文件数或日志无法判断
# 它到底在等还是已经挂了。scripts/crawl_status.py 读这个文件。
_STARTED = time.time()
_STATS = {"collected": 0, "skipped": 0, "failed": 0, "articles": 0}


def _heartbeat(state: str, **extra) -> None:
    """Record current crawler state for the status script."""
    payload = {
        "state": state,            # starting | fetching | cooling | done
        "updated_at": time.time(),
        "elapsed_sec": round(time.time() - _STARTED, 1),
        **_STATS,
        **extra,
    }
    META_DIR.mkdir(parents=True, exist_ok=True)
    try:
        HEARTBEAT_PATH.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass       # 心跳失败不应中断采集


# ═══════════════════════════════════════════════════════════════
#  network
# ═══════════════════════════════════════════════════════════════

class RateLimiter:
    """Enforce a minimum interval between requests, with jitter.

    抖动是必要的：固定间隔的请求序列本身就是机器人特征，容易再次触发
    WAF 挑战。
    """

    def __init__(self, min_interval: float, jitter: float = 0.4):
        self.min_interval = min_interval
        self.jitter = jitter
        self._last = 0.0

    def wait(self) -> None:
        target = self.min_interval * (1.0 + random.uniform(0, self.jitter))
        elapsed = time.time() - self._last
        if elapsed < target:
            time.sleep(target - elapsed)
        self._last = time.time()


class WafChallenge(RuntimeError):
    """The site answered with an anti-bot page instead of JSON."""


class SiteThrottled(RuntimeError):
    """Consecutive challenges with no success in between — stop the run."""


class WafGuard:
    """Adaptive cooldown for the site's anti-bot throttle.

    站点前置的网宿 WAF 有两级反应：轻量时返回 302/307 + ``wzws_cid``
    cookie（带着 cookie 重发即可通过），请求量继续增长后升级为 **JavaScript
    挑战页** —— 那需要执行 JS 才能拿到 cookie，普通 HTTP 客户端过不去。

    关键是这个挑战是**冷却式**的：停手一段时间后站点会恢复正常（实测首页
    从 302 回到 200）。所以正确策略不是硬闯，而是识别到挑战后整体停下来等，
    再慢慢恢复。等待时间随连续命中次数指数增长，成功后逐步衰减。
    """

    def __init__(self, base: float = 180.0, cap: float = 900.0,
                 abort_after: int = 3):
        self.base = base
        self.cap = cap
        self.abort_after = abort_after
        self.hits = 0            # 连续命中次数，成功后衰减
        self._resume_at = 0.0

    def wait(self) -> None:
        """Block until any pending cooldown has elapsed."""
        while True:
            remaining = self._resume_at - time.time()
            if remaining <= 0:
                return
            print(f"    [WAF] 冷却中，还需等待 {remaining/60:.1f} 分钟 …",
                  flush=True)
            time.sleep(min(remaining, 60))

    def penalize(self) -> None:
        self.hits += 1
        if self.hits >= self.abort_after:
            # 连续被挑战且中间一次都没成功 —— 说明站点正在硬限流，再等下去
            # 只是持续戳它。明确停下来，让调用方干净退出，由用户稍后续跑。
            # 不加这个熔断的话，一次请求最多会耗掉 3+6+12+15+15 ≈ 51 分钟，
            # 失败后下一部法规又重新开始，整轮会静默烧掉几个小时。
            raise SiteThrottled(
                f"连续 {self.hits} 次被反爬挑战且期间无成功请求，"
                f"站点正在限流；已停止本轮，稍后重跑即可续传"
            )
        delay = min(self.base * (2 ** (self.hits - 1)), self.cap)
        self._resume_at = time.time() + delay
        print(f"    [WAF] 触发反爬挑战（连续第 {self.hits} 次），"
              f"暂停 {delay/60:.1f} 分钟后重试", flush=True)
        _heartbeat("cooling", waf_hits=self.hits,
                   resume_at=self._resume_at, cooldown_sec=round(delay, 1))

    @property
    def resume_at(self) -> float:
        return self._resume_at

    def reward(self) -> None:
        """A successful request clears the consecutive-challenge counter."""
        self.hits = 0


_GUARD = WafGuard()


def _request(url: str, data: bytes | None = None, *, timeout: int = 60,
             retries: int = 5, expect_json: bool = True) -> bytes:
    """GET/POST with backoff, WAF retry and cooldown. Raises on final failure."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        _GUARD.wait()
        try:
            req = urllib.request.Request(url, data=data, headers={
                **HEADERS,
                **({"Content-Type": "application/json"} if data else {}),
            })
            with _OPENER.open(req, timeout=timeout) as resp:
                body = resp.read()
            if expect_json and body.lstrip()[:1] not in (b"{", b"["):
                # WAF 的 JS 挑战页也是 200 + text/html，不检查就会一路当成
                # JSON 解析失败，把"被反爬拦住"误报成"数据格式不对"
                raise WafChallenge(
                    f"non-JSON response ({len(body)} bytes, "
                    f"content-type={resp.headers.get('Content-Type')})"
                )
            _GUARD.reward()
            return body
        except WafChallenge as exc:
            last_exc = exc
            _GUARD.penalize()
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code in (301, 302, 307):
                # 轻量 WAF 挑战。CookieJar 已从这次响应里存下 wzws_cid，
                # 原样重发一次就能过。307 会保留 POST 方法和 body，
                # 所以重发必须用同一个 data。
                time.sleep(1.5)
                continue
            if exc.code == 429:
                _GUARD.penalize()
                continue
            if 400 <= exc.code < 500:
                raise          # 客户端错误重试无意义
            time.sleep(2 ** attempt)
        except Exception as exc:                     # noqa: BLE001
            last_exc = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"request failed after {retries} attempts: {last_exc}")


def search_page(code: int, page: int, size: int = 100) -> dict:
    """One page of the statute list for a leaf category code."""
    body = json.dumps({
        "searchRange": 1, "searchType": 2, "xgzlSearch": False, "searchContent": "",
        "orderByParam": {"order": "-1", "sort": ""},
        "flfgCodeId": [code], "zdjgCodeId": [], "gbrqYear": [],
        "pageNum": page, "pageSize": size,
    }).encode()
    return json.loads(_request(f"{BASE}/law-search/search/list", body))


def fetch_document(bbbs: str, limiter: RateLimiter, fmt: str = "docx") -> bytes:
    """Resolve the signed OSS url and download the document bytes."""
    limiter.wait()
    meta = json.loads(_request(
        f"{BASE}/law-search/download/pc?format={fmt}&bbbs={bbbs}"
    ))
    url = (meta.get("data") or {}).get("url")
    if not url:
        raise RuntimeError(f"no download url for {bbbs}")
    # OSS 是 CDN，不占源站配额，也无需限流；返回的是二进制 docx，不能按 JSON 校验
    return _request(url, timeout=120, expect_json=False)


def fetch_articles(bbbs: str, limiter: RateLimiter) -> tuple[list[dict], str]:
    """Download and parse one statute. Returns ``(articles, format_used)``.

    三级兜底，按可靠性排序：

    1. **docx** —— 绝大多数文件走这条
    2. **pdf** —— 少数最高检文件不以 docx 提供
    3. **antiword** —— 源站对一批司法解释返回老式 ``.doc``（OLE2），而配套
       PDF 是扫描件（无文字层）。实测两条常规路径都失败时共有 56 部漏采，
       恰好是最高频引用的那批（民间借贷、买卖合同、医疗损害……）。
       antiword 是可选依赖，未安装时这一级直接跳过。

    只有全部失败才抛错，且错误里带上每一级的原因 —— 否则会把人引向错方向。
    """
    errors: list[str] = []

    # 1) docx
    try:
        return LawTextParser.parse(fetch_document(bbbs, limiter, "docx")), "docx"
    except SiteThrottled:
        raise
    except Exception as exc:                         # noqa: BLE001
        errors.append(f"docx: {type(exc).__name__}: {str(exc)[:90]}")

    # 2) pdf
    try:
        paragraphs = LawTextParser.parse_pdf(fetch_document(bbbs, limiter, "pdf"))
        if paragraphs:
            return LawTextParser.extract_articles(paragraphs), "pdf"
        errors.append("pdf: no text layer (scanned image)")
    except SiteThrottled:
        raise
    except Exception as exc:                         # noqa: BLE001
        errors.append(f"pdf: {type(exc).__name__}: {str(exc)[:90]}")

    # 3) antiword（老式 .doc）
    try:
        data = fetch_document(bbbs, limiter, "docx")   # 同一份内容，交给 antiword
        paragraphs = LawTextParser.parse_legacy_doc(data)
        if paragraphs:
            return LawTextParser.extract_articles(paragraphs), "doc"
        errors.append("antiword: 提取为空或未安装 antiword")
    except SiteThrottled:
        raise
    except Exception as exc:                         # noqa: BLE001
        errors.append(f"antiword: {type(exc).__name__}: {str(exc)[:90]}")

    raise RuntimeError(" | ".join(errors))


# ═══════════════════════════════════════════════════════════════
#  crawl
# ═══════════════════════════════════════════════════════════════

def enumerate_category(code: int, limiter: RateLimiter, limit: int | None) -> list[dict]:
    """All statutes under one leaf category code."""
    rows: list[dict] = []
    page = 1
    while True:
        limiter.wait()
        data = search_page(code, page)
        batch = data.get("rows") or []
        rows.extend(batch)
        total = data.get("total") or 0
        if limit and len(rows) >= limit:
            return rows[:limit]
        if not batch or len(rows) >= total:
            return rows
        page += 1


def build_record(row: dict, cat_key: str, cat_label: str, sub_label: str,
                 articles: list[dict]) -> dict:
    """Assemble the per-statute JSON record."""
    record = {
        "bbbs": row.get("bbbs", ""),
        "title": clean_title(row.get("title")),
        "category": cat_label,
        "subcategory": sub_label,
        "issuing_body": row.get("zdjgName", ""),
        "gbrq": row.get("gbrq") or "",        # 公布日期
        "sxrq": row.get("sxrq") or "",        # 施行日期
        "sxx": row.get("sxx", 0),           # 1废止 2已修改 3现行有效 4尚未生效
        "status": STATUS_LABEL.get(row.get("sxx", 0), "未知"),
        "source_url": f"{BASE}/detail?id={row.get('bbbs', '')}",
        "articles": articles,
    }
    if not articles:
        # 修正案类文档（"一、将第一条修改为……"）没有可切分的条文，
        # 标记出来让索引阶段跳过，而不是静默产出空记录
        record["parse_warning"] = "no_articles"
    return record


def _flush(catalog: list[dict], report: dict) -> None:
    """Persist catalog + report so an interrupted run keeps its diagnostics."""
    META_DIR.mkdir(parents=True, exist_ok=True)
    (META_DIR / "catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=1), encoding="utf-8")
    (META_DIR / "crawl_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="采集国家法律法规数据库的法规正文",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--categories", default="law,admin,judicial",
                    help="逗号分隔: law,admin,judicial（默认全部）")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="输出根目录")
    ap.add_argument("--limit", type=int, default=None,
                    help="每个叶类别最多采集多少部（试跑用）")
    ap.add_argument("--rate", type=float, default=1.5,
                    help="对 flk.npc.gov.cn 的最小请求间隔秒数（默认 1.5；"
                         "调低容易触发 WAF 挑战）")
    ap.add_argument("--current-only", action="store_true",
                    help="只采集现行有效的法规（sxx=3）")
    args = ap.parse_args()

    wanted = [c.strip() for c in args.categories.split(",") if c.strip()]
    unknown = [c for c in wanted if c not in CATEGORIES]
    if unknown:
        ap.error(f"未知类别: {unknown}（可选: {list(CATEGORIES)}）")

    out_root = Path(args.out)
    limiter = RateLimiter(args.rate)

    print("=" * 66)
    print("  国家法律法规数据库 · 语料采集")
    print("=" * 66)
    print(f"  类别: {wanted}    限流: {args.rate}s/请求"
          f"{'    仅现行有效' if args.current_only else ''}")
    print()
    catalog: list[dict] = []
    report: dict = {"categories": {}, "failures": [], "warnings": []}

    _heartbeat("starting", categories=wanted)
    # 立刻用空报告覆盖上一次运行的结果。不覆盖的话，新的一轮在跑到第一个
    # 子类别结束前都不会 flush，而监控脚本读到的是上一轮的报告 ——
    # 刚启动的运行会显示成"失败 1976 部"，完全是误导。
    _flush(catalog, report)

    try:
        return _crawl(wanted, args, out_root, limiter, catalog, report)
    except SiteThrottled as exc:
        # 干净退出：落盘已有成果，让用户稍后续跑，而不是默默耗几个小时
        report["totals"] = {
            "articles": sum(c.get("articles", 0)
                            for c in report["categories"].values()),
            "catalog_size": len(catalog),
            "failures": len(report["failures"]),
            "aborted": True,
        }
        _flush(catalog, report)
        _heartbeat("throttled", reason=str(exc))
        print()
        print("=" * 66)
        print(f"  ⏹ 已停止：{exc}")
        print(f"  本轮已落盘 {len(catalog)} 部。站点冷却一段时间后，"
              f"重跑同样的命令即可续传。")
        print("=" * 66)
        return 2


def _crawl(wanted: list[str], args, out_root: Path, limiter: RateLimiter,
           catalog: list[dict], report: dict) -> int:
    """The crawl loop proper; may raise SiteThrottled."""
    grand_total = 0

    for cat_key in wanted:
        cat_label, subs = CATEGORIES[cat_key]
        cat_dir = out_root / cat_key
        cat_dir.mkdir(parents=True, exist_ok=True)

        cat_stat = {"label": cat_label, "laws": 0, "articles": 0,
                    "skipped_existing": 0, "failed": 0, "no_articles": 0}
        print(f"── {cat_label} ({cat_key}) ──")

        for sub_label, code in subs:
            try:
                rows = enumerate_category(code, limiter, args.limit)
            except Exception as exc:                 # noqa: BLE001
                print(f"  [!] 枚举失败 {sub_label}({code}): {exc}")
                report["failures"].append({"stage": "enumerate", "code": code,
                                           "sub": sub_label, "error": str(exc)})
                continue

            print(f"  {sub_label:20s} code={code:<4d} 共 {len(rows):>5d} 部")
            for row in rows:
                bbbs = row.get("bbbs")
                if not bbbs:
                    continue

                if args.current_only and row.get("sxx") != 3:
                    continue

                target = cat_dir / f"{bbbs}.json"
                prev = None
                if target.exists():
                    try:
                        prev = json.loads(target.read_text(encoding="utf-8"))
                    except Exception:                # noqa: BLE001
                        prev = None

                # 只有"已落盘且确实切出了内容"才算完成。`articles` 为空说明上一轮
                # 的解析器没能切出东西（可能是后来才支持的格式，例如不用「第X条」
                # 编号的老文件），重取一次即可，不需要额外的回填步骤。
                if prev is not None and prev.get("articles"):
                    cat_stat["skipped_existing"] += 1
                    _STATS["skipped"] += 1
                    cat_stat["articles"] += len(prev["articles"])
                    catalog.append({k: prev.get(k, "") for k in (
                        "bbbs", "title", "category", "subcategory",
                        "issuing_body", "gbrq", "sxrq", "sxx", "status")} | {
                            "article_count": len(prev["articles"])})
                    continue

                _heartbeat("fetching", current=clean_title(row.get("title")))
                try:
                    articles, fmt_used = fetch_articles(bbbs, limiter)
                    if fmt_used == "pdf":
                        report["warnings"].append({
                            "bbbs": bbbs, "title": clean_title(row.get("title")),
                            "reason": "docx_unavailable_used_pdf"})
                except SiteThrottled:
                    # 必须放行。此前它被下面的 except Exception 吞掉，导致熔断
                    # 只是把后续每一部法规都标成失败然后继续跑 —— 实测一次
                    # 发作积压了 117 条假的"失败"记录，而整轮并没有停下来。
                    raise
                except Exception as exc:             # noqa: BLE001
                    cat_stat["failed"] += 1
                    _STATS["failed"] += 1
                    report["failures"].append({
                        "stage": "download", "bbbs": bbbs,
                        "title": clean_title(row.get("title")),
                        "error": f"{type(exc).__name__}: {exc}"[:200],
                    })
                    print(f"    [!] {clean_title(row.get('title'))[:34]} 失败: "
                          f"{type(exc).__name__}", flush=True)
                    continue

                record = build_record(row, cat_key, cat_label, sub_label, articles)
                if not articles:
                    cat_stat["no_articles"] += 1
                    report["warnings"].append({
                        "bbbs": bbbs, "title": record["title"],
                        "reason": "no_articles"})

                target.write_text(
                    json.dumps(record, ensure_ascii=False, indent=1),
                    encoding="utf-8",
                )
                cat_stat["laws"] += 1
                cat_stat["articles"] += len(articles)
                grand_total += len(articles)
                _STATS["collected"] += 1
                _STATS["articles"] += len(articles)

                # 每 25 部打一行进度。子类别动辄上百部，没有这个的话中途
                # 完全看不出进展（原来只在子类别结束时汇总一次）
                if _STATS["collected"] % 25 == 0:
                    elapsed = time.time() - _STARTED
                    rate = _STATS["collected"] / max(elapsed, 1) * 60.0
                    print(f"    … 已采 {_STATS['collected']} 部 / "
                          f"{_STATS['articles']:,} 条  "
                          f"({rate:.1f} 部/分钟)", flush=True)

                catalog.append({k: record[k] for k in (
                    "bbbs", "title", "category", "subcategory",
                    "issuing_body", "gbrq", "sxrq", "sxx", "status")} | {
                        "article_count": len(articles)})

        report["categories"][cat_key] = cat_stat
        print(f"  → 新采 {cat_stat['laws']} 部 / {cat_stat['articles']} 条，"
              f"跳过已存在 {cat_stat['skipped_existing']} 部，"
              f"失败 {cat_stat['failed']} 部")
        print()

        # 增量落盘：崩溃或被中断时诊断信息不丢失（上一次中断就把失败原因
        # 留在了内存里，只能重跑才知道发生了什么）
        _flush(catalog, report)

    # ── persist metadata ──
    report["totals"] = {
        "articles": grand_total,
        "catalog_size": len(catalog),
        "failures": len(report["failures"]),
        "no_articles": sum(1 for w in report["warnings"]
                           if w.get("reason") == "no_articles"),
        "pdf_fallback": sum(1 for w in report["warnings"]
                            if w.get("reason") == "docx_unavailable_used_pdf"),
    }
    _flush(catalog, report)
    _heartbeat("done")

    print("=" * 66)
    print(f"  条文总计: {grand_total}   收录法规: {len(catalog)} 部")
    print(f"  失败: {report['totals']['failures']}    "
          f"无可切分条文: {report['totals']['no_articles']}    "
          f"PDF 兜底: {report['totals']['pdf_fallback']}")
    print(f"  产出: {out_root}/   清单: {META_DIR}/catalog.json")
    print("=" * 66)

    return 0 if not report["failures"] else 1


if __name__ == "__main__":
    sys.exit(main())
