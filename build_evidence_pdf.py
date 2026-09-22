"""佐证材料 markdown -> PDF。复用本地 Edge 后端。

用法:
    "C:/Users/LENOVO/.workbuddy/binaries/python/envs/default/Scripts/python.exe" \\
        CACC_vecsearch/experiments/build_evidence_pdf.py

输出:
    .workbuddy/artifacts/CACC佐证材料_v1.pdf
    .workbuddy/artifacts/CACC佐证材料_v1.html
"""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path

import markdown

WORKSPACE = Path(__file__).resolve().parents[2]
ARTIFACTS = WORKSPACE / ".workbuddy" / "artifacts"
SRC_MD = ARTIFACTS / "CACC佐证材料_v1.md"
OUT_HTML = ARTIFACTS / "CACC佐证材料_v1.html"
OUT_PDF = ARTIFACTS / "CACC佐证材料_v1.pdf"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_report_pdf import _process_math  # noqa: E402


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
<title>CACC 佐证材料</title>
<style>
  @page {{ size: A4; margin: 1.8cm 1.8cm 2cm 1.8cm; }}
  body {{
    font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
    font-size: 10pt;
    line-height: 1.55;
    color: #111;
  }}
  h1 {{ font-size: 20pt; border-bottom: 3px solid #0a3b30; padding-bottom: 6pt; color: #0a3b30; }}
  h2 {{ font-size: 14pt; border-bottom: 1px solid #ccc; padding-bottom: 3pt; color: #0a3b30; margin-top: 16pt; }}
  h3 {{ font-size: 12pt; color: #0a3b30; margin-top: 12pt; }}
  p {{ margin: 4pt 0; text-align: justify; }}
  ul, ol {{ margin: 4pt 0; padding-left: 22pt; }}
  li {{ margin: 2pt 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 9pt; }}
  th, td {{ border: 1px solid #888; padding: 4pt 6pt; text-align: left; vertical-align: top; }}
  th {{ background: #f3f6f4; font-weight: 600; }}
  tr:nth-child(even) td {{ background: #fafbfa; }}
  code {{ background: #f4f4f4; padding: 1pt 4pt; border-radius: 3pt; font-family: Consolas, monospace; font-size: 9pt; }}
  pre {{ background: #f4f4f4; padding: 6pt 8pt; border-radius: 4pt; font-size: 8.5pt; overflow-x: auto; line-height: 1.35; }}
  pre code {{ background: transparent; padding: 0; }}
  blockquote {{ border-left: 3pt solid #0a3b30; margin: 6pt 0; padding: 4pt 10pt; background: #f3f6f4; }}
  strong {{ color: #0a3b30; }}
  hr {{ border: 0; border-top: 1px dashed #aaa; margin: 12pt 0; }}
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
        print("[!] Edge 未找到")
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
    subprocess.run(cmd, check=True, timeout=180)
    print(f"[3/3] PDF: {OUT_PDF}  ({OUT_PDF.stat().st_size/1e6:.2f} MB)")
    try:
        from pypdf import PdfReader
        n = len(PdfReader(str(OUT_PDF)).pages)
        print(f"      → 页数: {n}")
    except Exception as e:
        print(f"      → 页数跳过: {e}")


if __name__ == "__main__":
    main()
