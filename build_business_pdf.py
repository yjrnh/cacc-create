"""创业计划书 markdown -> HTML -> PDF。复用 build_report_pdf.py 的本地 Edge 后端。

用法:
    "C:/Users/LENOVO/.workbuddy/binaries/python/envs/default/Scripts/python.exe" \\
        CACC_vecsearch/experiments/build_business_pdf.py

输出:
    .workbuddy/artifacts/CACC创业计划书_v1.pdf
    .workbuddy/artifacts/CACC创业计划书_v1.html
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

WORKSPACE = Path(__file__).resolve().parents[2]
ARTIFACTS = WORKSPACE / ".workbuddy" / "artifacts"
SRC_MD = ARTIFACTS / "CACC创业计划书_v1.md"
OUT_HTML = ARTIFACTS / "CACC创业计划书_v1.html"
OUT_PDF = ARTIFACTS / "CACC创业计划书_v1.pdf"

# 复用 build_report_pdf 里的样式与公式处理
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_report_pdf import _process_math, _latex_to_unicode  # noqa: E402


def md_to_html(md_text: str) -> str:
    md_text = _process_math(md_text)
    body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc", "sane_lists"],
    )
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<title>CACC 创业计划书 — BCAS 面向 RAG 应用的向量检索成本治理引擎</title>
<style>
  @page {{
    size: A4;
    margin: 2cm 2cm 2.2cm 2cm;
  }}
  body {{
    font-family: "Microsoft YaHei", "PingFang SC", "Source Han Sans CN", sans-serif;
    font-size: 10.5pt;
    line-height: 1.65;
    color: #111;
    max-width: 100%;
  }}
  h1, h2, h3, h4 {{ color: #0a3b30; }}
  h1 {{
    font-size: 22pt;
    border-bottom: 3px solid #0a3b30;
    padding-bottom: 6pt;
    margin-top: 0;
    page-break-before: always;
  }}
  h1:first-of-type {{ page-break-before: avoid; }}
  h2 {{ font-size: 15pt; border-bottom: 1px solid #ccc; padding-bottom: 4pt; margin-top: 18pt; }}
  h3 {{ font-size: 12.5pt; margin-top: 14pt; }}
  h4 {{ font-size: 11pt; margin-top: 12pt; }}
  p {{ margin: 6pt 0; text-align: justify; }}
  ul, ol {{ margin: 6pt 0; padding-left: 22pt; }}
  li {{ margin: 3pt 0; }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 10pt 0;
    font-size: 10pt;
  }}
  th, td {{
    border: 1px solid #888;
    padding: 5pt 7pt;
    text-align: left;
    vertical-align: top;
  }}
  th {{ background: #f3f6f4; font-weight: 600; }}
  tr:nth-child(even) td {{ background: #fafbfa; }}
  code {{
    background: #f4f4f4;
    padding: 1pt 4pt;
    border-radius: 3pt;
    font-family: Consolas, "Courier New", monospace;
    font-size: 9.5pt;
  }}
  pre {{
    background: #f4f4f4;
    padding: 8pt 10pt;
    border-radius: 4pt;
    font-size: 9.5pt;
    overflow-x: auto;
    line-height: 1.4;
  }}
  pre code {{ background: transparent; padding: 0; }}
  blockquote {{
    border-left: 3pt solid #0a3b30;
    margin: 8pt 0;
    padding: 4pt 12pt;
    color: #333;
    background: #f3f6f4;
  }}
  hr {{
    border: 0;
    border-top: 1px dashed #aaa;
    margin: 16pt 0;
  }}
  strong {{ color: #0a3b30; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def main() -> None:
    print(f"[1/3] 读取 {SRC_MD.name} …")
    md_text = SRC_MD.read_text(encoding="utf-8")
    html = md_to_html(md_text)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"      → HTML: {OUT_HTML}  ({OUT_HTML.stat().st_size/1024:.1f} KB)")

    edge = Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")
    if not edge.exists():
        edge = Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe")
    if not edge.exists():
        print("[!] Edge 未找到，无法转 PDF。请改用 wkhtmltopdf 或安装 Edge。")
        sys.exit(1)

    print(f"[2/3] 调用 Edge headless …")
    cmd = [
        str(edge),
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        f"--print-to-pdf={OUT_PDF}",
        str(OUT_HTML),
    ]
    subprocess.run(cmd, check=True, timeout=120)

    print(f"[3/3] PDF: {OUT_PDF}  ({OUT_PDF.stat().st_size/1e6:.2f} MB)")
    # 页数统计
    try:
        from pypdf import PdfReader
        n = len(PdfReader(str(OUT_PDF)).pages)
        print(f"      → 页数: {n}")
    except Exception as e:
        print(f"      → 页数统计跳过（pypdf 未装或解析失败: {e}）")


if __name__ == "__main__":
    main()
