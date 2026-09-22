"""一次性 fix 嵌套引号问题。"""
import re
from pathlib import Path

p = Path(r'C:/Users/LENOVO/WorkBuddy/2026-09-14-16-24-50/CACC_vecsearch/experiments/build_pptx.py')
text = p.read_text(encoding='utf-8')

def repl(m):
    return m.group(1) + chr(0x300C) + m.group(2) + chr(0x300D) + m.group(3)

# 1) 前导是空格/=/>/- 等 ASCII 符号
new = re.sub(r'([\s=><+\-])"([^"\n]{1,40}?)"([\u4e00-\u9fff，])', repl, text)
# 2) 前导是中文/中文逗号
new = re.sub(r'([\u4e00-\u9fff，])"([^"\n]{1,40}?)"([，。\u4e00-\u9fff])',
             lambda m: m.group(1) + chr(0x300C) + m.group(2) + chr(0x300D) + m.group(3), new)

p.write_text(new, encoding='utf-8')
print('done. 「」 count:', new.count(chr(0x300C)))
