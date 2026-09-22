#!/usr/bin/env python3
"""把技术报告 markdown 转成 PDF（封面 + 目录 + 正文 + 图）。

依赖：python-markdown（已在 managed venv 安装）。
转换链路：md -> HTML -> Edge headless --print-to-pdf。
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import markdown

WORKSPACE = Path("C:/Users/LENOVO/WorkBuddy/2026-09-14-16-24-50")
MD_PATH = WORKSPACE / ".workbuddy/artifacts/CACC技术报告_草稿v1.md"
HTML_PATH = WORKSPACE / ".workbuddy/artifacts/CACC技术报告_v1.html"
PDF_PATH = WORKSPACE / ".workbuddy/artifacts/CACC技术报告_v1.pdf"
FIGS_DIR = WORKSPACE / "CACC_vecsearch/figures"

# 让打印引擎能用到的封面信息（学生信息留空，交付前手动填入）
COVER_INFO = {
    "title": "算法技术报告",
    "subtitle": "可预算约束的近邻向量检索索引改造",
    "competition": "CACC 创业赛道 2026 第一轮",
    "candidate": "（姓名）",
    "school": "成都信息工程大学",
    "major": "区块链工程专业",
    "phone": "（联系电话）",
    "email": "（邮箱）",
    "date": "2026-09-14",
}


def _latex_to_unicode(text: str) -> str:
    """把简单 LaTeX 数学内容转成 HTML 可渲染的 unicode + 上下标。"""
    # 先处理上下标（保留为 HTML 标签）
    text = re.sub(r"\^{\s*([^}]+)\s*}", lambda m: f"<sup>{m.group(1)}</sup>", text)
    text = re.sub(r"_\{\s*([^}]+)\s*}", lambda m: f"<sub>{m.group(1)}</sub>", text)
    # 分式
    text = re.sub(
        r"\\frac\{([^}]+)\}\{([^}]+)\}",
        lambda m: f"({m.group(1)})/({m.group(2)})",
        text,
    )
    # 简单命令 → unicode（用 replace 防止 regex replacement 转义问题）
    cmd_map = {
        r"\varepsilon": "ε", r"\epsilon": "ε",
        r"\ge": "≥", r"\geq": "≥",
        r"\le": "≤", r"\leq": "≤",
        r"\ne": "≠", r"\neq": "≠",
        r"\times": "×", r"\cdot": "·",
        r"\in": "∈", r"\infty": "∞",
        r"\Delta": "Δ",
        r"\phi": "φ", r"\varphi": "φ",
        r"\tau": "τ", r"\mu": "μ", r"\sigma": "σ", r"\gamma": "γ",
        r"\alpha": "α", r"\beta": "β", r"\theta": "θ", r"\lambda": "λ", r"\pi": "π",
        r"\setminus": "∖",
    }
    for old, new in cmd_map.items():
        text = text.replace(old, new)
    # \text{} / \mathit{} / \mathrm{}
    text = re.sub(r"\\text\{([^}]*)\}", lambda m: m.group(1), text)
    text = re.sub(r"\\mathit\{([^}]*)\}", lambda m: m.group(1), text)
    text = re.sub(r"\\mathrm\{([^}]*)\}", lambda m: m.group(1), text)
    # 去掉残留反斜杠
    text = text.replace("\\", "")
    # 清理多余空白
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _process_math(md_text: str) -> str:
    """处理 $$...$$ 行间数学与 $...$ 行内数学。"""
    def display_repl(m: re.Match) -> str:
        inner = _latex_to_unicode(m.group(1))
        return f'<div style="text-align:center; margin:0.8em 0">{inner}</div>'

    def inline_repl(m: re.Match) -> str:
        return _latex_to_unicode(m.group(1))

    # 先处理行间 $$...$$
    md_text = re.sub(r"\$\$([^$]+?)\$\$", display_repl, md_text)
    # 再处理行内 $...$
    md_text = re.sub(r"\$([^$]+)\$", inline_repl, md_text)
    return md_text


def _prepare_markdown(md_text: str) -> str:
    # 去掉工作稿提示
    md_text = re.sub(r">\s*本文件是工作稿.*\n?", "", md_text)
    md_text = re.sub(r">\s*交付前需转 PDF.*\n?", "", md_text)
    # 把标题从“草稿 v1”改成正式标题
    md_text = md_text.replace("# 算法技术报告（草稿 v1）", "# 算法技术报告")

    # 删除待办清单章节（内部交付检查单，不应出现在提交 PDF 中）
    md_text = re.split(r"\n## 待办清单[（(].*", md_text)[0].rstrip() + "\n"

    # 数学转换
    md_text = _process_math(md_text)

    # 在摘要前插入目录标记
    if "## 摘要" in md_text:
        md_text = md_text.replace("## 摘要", "[TOC]\n\n## 摘要")
    return md_text


def _cover_html() -> str:
    return f"""
<div class="cover">
  <div class="cover-title">{COVER_INFO['title']}</div>
  <div class="cover-subtitle">{COVER_INFO['subtitle']}</div>
  <div class="cover-competition">{COVER_INFO['competition']}</div>
  <div class="cover-meta">
    <p><strong>参赛人：</strong>{COVER_INFO['candidate']}</p>
    <p><strong>学校：</strong>{COVER_INFO['school']}</p>
    <p><strong>专业：</strong>{COVER_INFO['major']}</p>
    <p><strong>电话：</strong>{COVER_INFO['phone']}</p>
    <p><strong>邮箱：</strong>{COVER_INFO['email']}</p>
    <p><strong>提交日期：</strong>{COVER_INFO['date']}</p>
  </div>
</div>
"""


def _back_cover_html() -> str:
    return """
<div class="page-break"></div>
<div class="back-cover">
  <p>— 报告结束 —</p>
</div>
"""


def _css() -> str:
    return """
@page {
  size: A4;
  margin: 2cm 2cm 2.5cm 2cm;
}

* {
  box-sizing: border-box;
}

body {
  font-family: "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", sans-serif;
  font-size: 10.5pt;
  line-height: 1.65;
  color: #1a1a1a;
}

/* 封面 */
.cover {
  min-height: 75vh;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
  page-break-after: always;
}
.cover-title {
  font-size: 28pt;
  font-weight: bold;
  margin-bottom: 0.6em;
  letter-spacing: 0.05em;
}
.cover-subtitle {
  font-size: 16pt;
  color: #333;
  margin-bottom: 2em;
}
.cover-competition {
  font-size: 12pt;
  color: #555;
  margin-bottom: 4em;
}
.cover-meta {
  text-align: left;
  font-size: 12pt;
  line-height: 2;
  margin-top: 2em;
}
.cover-meta p {
  margin: 0.2em 0;
}

/* 目录 */
.toc {
  page-break-after: always;
}
.toc ul {
  list-style: none;
  padding-left: 0;
}
.toc li {
  margin: 0.35em 0;
}
.toc a {
  text-decoration: none;
  color: #000;
}

/* 分页 */
.page-break {
  page-break-after: always;
}
.back-cover {
  height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
  color: #666;
}

/* 标题 */
h1, h2, h3, h4 {
  font-weight: bold;
  color: #111;
  page-break-after: avoid;
}
h1 { font-size: 18pt; margin-top: 1.5em; }
h2 { font-size: 14pt; margin-top: 1.3em; border-bottom: 1.5px solid #ccc; padding-bottom: 0.2em; }
h3 { font-size: 12pt; margin-top: 1.1em; }
h4 { font-size: 10.5pt; margin-top: 1em; }

p {
  margin: 0.6em 0;
  text-align: justify;
}

/* 表格 */
table {
  border-collapse: collapse;
  width: 100%;
  font-size: 9pt;
  margin: 1em 0;
  page-break-inside: auto;
}
thead {
  display: table-header-group;
}
tr {
  page-break-inside: avoid;
}
th, td {
  border: 0.75pt solid #999;
  padding: 4pt 6pt;
  text-align: center;
  vertical-align: middle;
}
th {
  background: #f2f2f2;
  font-weight: bold;
}

/* 代码块 */
code {
  font-family: Consolas, "Courier New", monospace;
  background: #f5f5f5;
  padding: 0.1em 0.3em;
  border-radius: 2pt;
  font-size: 9pt;
}
pre {
  background: #f8f8f8;
  border: 0.75pt solid #ddd;
  padding: 8pt;
  overflow-x: auto;
  font-size: 8.5pt;
  line-height: 1.4;
  page-break-inside: avoid;
}
pre code {
  background: transparent;
  padding: 0;
}

/* 图片 */
img {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 1em auto;
  page-break-inside: avoid;
}

/* 引用块 */
blockquote {
  margin: 0.8em 0;
  padding: 0.5em 1em;
  border-left: 3pt solid #2b6cb0;
  background: #f7fafc;
  color: #2d3748;
  font-size: 10pt;
}

/* 列表 */
ul, ol {
  margin: 0.5em 0;
  padding-left: 1.8em;
}
li {
  margin: 0.25em 0;
}

/* 水平线 */
hr {
  border: none;
  border-top: 0.75pt solid #ccc;
  margin: 1.5em 0;
}

/* 摘要 */
#摘要 {
  page-break-before: always;
}

/* 防止大图/表格单独跨页时难看 */
img, table, figure {
  page-break-inside: avoid;
}
"""


def build_html() -> str:
    md_text = MD_PATH.read_text(encoding="utf-8")
    md_text = _prepare_markdown(md_text)

    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc"],
        extension_configs={"toc": {"toc_depth": 3}},
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>{COVER_INFO['title']}</title>
<style>{_css()}</style>
</head>
<body>
{_cover_html()}
{html_body}
{_back_cover_html()}
</body>
</html>"""


def print_pdf() -> None:
    HTML_PATH.write_text(build_html(), encoding="utf-8")
    edge = "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
    cmd = [
        edge,
        "--headless",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        "--print-to-pdf=" + str(PDF_PATH),
        str(HTML_PATH.as_uri()),
    ]
    print("Running Edge headless print-to-pdf...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Edge stderr:", result.stderr, file=sys.stderr)
        raise RuntimeError(f"Edge print-to-pdf failed with exit code {result.returncode}")
    if not PDF_PATH.exists():
        raise RuntimeError("PDF file was not created")
    size_mb = PDF_PATH.stat().st_size / (1024 * 1024)
    print(f"✓ PDF created: {PDF_PATH} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    print_pdf()
