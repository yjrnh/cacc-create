# -*- coding: utf-8 -*-
"""抓取 GitHub 仓库 Traffic 数据（仅保留 14 天），合并写入 metrics/ 永久存档。

输出：
  metrics/traffic_daily.csv    日期, 浏览次数, 独立访客, 克隆次数, 独立克隆者
  metrics/traffic_referrers.csv 快照日期, 来源站, 浏览次数, 独立访客
  metrics/traffic_summary.md   汇总（近 14 天 + 累计）
"""
import csv
import json
import os
import urllib.request
from datetime import datetime, timezone

REPO = os.environ.get("REPO", "yjrnh/cacc-create")
TOKEN = os.environ["GITHUB_TOKEN"]
BASE = "https://api.github.com/repos/%s/traffic" % REPO
OUT = "metrics"


def get(path):
    req = urllib.request.Request(
        "%s/%s" % (BASE, path),
        headers={
            "Authorization": "Bearer %s" % TOKEN,
            "Accept": "application/vnd.github+json",
            "User-Agent": "traffic-archive",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def read_csv(path, key="date"):
    rows = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows[row[key]] = row
    return rows


def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for k in sorted(rows):
            w.writerow(rows[k])


def main():
    os.makedirs(OUT, exist_ok=True)

    views = get("views")
    clones = get("clones")
    referrers = get("popular/referrers")

    # ---- 每日数据合并（按日期 upsert，重复运行不产生脏行）----
    daily_path = os.path.join(OUT, "traffic_daily.csv")
    daily = read_csv(daily_path)
    by_date = {}
    for item in views.get("views", []):
        d = item["timestamp"][:10]
        by_date.setdefault(d, {})["views"] = item["count"]
        by_date.setdefault(d, {})["uniques"] = item["uniques"]
    for item in clones.get("clones", []):
        d = item["timestamp"][:10]
        by_date.setdefault(d, {})["clones"] = item["count"]
        by_date.setdefault(d, {})["clone_uniques"] = item["uniques"]

    for d, v in by_date.items():
        old = daily.get(d, {})
        daily[d] = {
            "date": d,
            "views": v.get("views", old.get("views", 0)),
            "unique_visitors": v.get("uniques", old.get("unique_visitors", 0)),
            "clones": v.get("clones", old.get("clones", 0)),
            "unique_cloners": v.get("clone_uniques", old.get("unique_cloners", 0)),
        }

    write_csv(
        daily_path,
        daily,
        ["date", "views", "unique_visitors", "clones", "unique_cloners"],
    )

    # ---- 来源快照（每次运行追加一行，保留历史）----
    ref_path = os.path.join(OUT, "traffic_referrers.csv")
    snap = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ref_rows = read_csv(ref_path, key="snapshot")
    for r in referrers:
        key = "%s|%s" % (snap, r["referrer"])
        ref_rows[key] = {
            "snapshot": snap,
            "referrer": r["referrer"],
            "views": r["count"],
            "uniques": r["uniques"],
        }
    write_csv(ref_path, ref_rows, ["snapshot", "referrer", "views", "uniques"])

    # ---- 汇总 ----
    last14 = sorted(daily)[-14:]
    tv = sum(int(daily[d]["views"]) for d in last14)
    tu = sum(int(daily[d]["unique_visitors"]) for d in last14)
    tc = sum(int(daily[d]["clones"]) for d in last14)
    tcu = sum(int(daily[d]["unique_cloners"]) for d in last14)
    all_v = sum(int(r["views"]) for r in daily.values())
    all_c = sum(int(r["clones"]) for r in daily.values())

    md = [
        "# 仓库流量存档",
        "",
        "仓库：`%s`　最后更新：%s (UTC)" % (REPO, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")),
        "",
        "## 近 14 天（GitHub 窗口期）",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        "| 页面浏览 | %d |" % tv,
        "| 独立访客 | %d |" % tu,
        "| 克隆次数 | %d |" % tc,
        "| 独立克隆者 | %d |" % tcu,
        "",
        "## 累计（自首次归档起）",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        "| 累计浏览 | %d |" % all_v,
        "| 累计克隆 | %d |" % all_c,
        "| 覆盖天数 | %d |" % len(daily),
        "",
        "## 最近明细",
        "",
        "| 日期 | 浏览 | 独立访客 | 克隆 | 独立克隆者 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for d in sorted(daily)[-14:]:
        r = daily[d]
        md.append("| %s | %s | %s | %s | %s |" % (d, r["views"], r["unique_visitors"], r["clones"], r["unique_cloners"]))
    md += ["", "> 数据由 GitHub Actions 每周自动抓取。GitHub 官方只保留 14 天每日明细，本分支为永久存档。", ""]

    with open(os.path.join(OUT, "traffic_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print("近14天：浏览 %d / 克隆 %d；累计：浏览 %d / 克隆 %d" % (tv, tc, all_v, all_c))


if __name__ == "__main__":
    main()
