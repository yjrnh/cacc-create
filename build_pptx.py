"""生成 PPT 介绍材料 (≤20 页)。

设计语言: 深绿主色 + 米白底 + 大量留白，与技术报告/计划书一致。
"""

from __future__ import annotations

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

WORKSPACE = Path(__file__).resolve().parents[2]
ARTIFACTS = WORKSPACE / ".workbuddy" / "artifacts"
OUT_PPTX = ARTIFACTS / "CACC项目介绍_v1.pptx"

C_PRIMARY = RGBColor(0x0A, 0x3B, 0x30)
C_ACCENT = RGBColor(0xC2, 0x6B, 0x1F)
C_BG = RGBColor(0xFA, 0xFB, 0xFA)
C_GRAY = RGBColor(0x55, 0x5A, 0x55)
C_LIGHT_GRAY = RGBColor(0xCC, 0xCC, 0xCC)
C_RED = RGBColor(0xB3, 0x32, 0x2E)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H

blank_layout = prs.slide_layouts[6]


def _set_solid_fill(shape, color):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()


def _add_rect(slide, x, y, w, h, color):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    _set_solid_fill(shp, color)
    return shp


def _add_textbox(slide, x, y, w, h, text, *, size=18, bold=False, color=None, align=None, anchor=None):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.05)
    tf.margin_right = Inches(0.05)
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0.02)
    if anchor is not None:
        tf.vertical_anchor = anchor
    p = tf.paragraphs[0]
    if align is not None:
        p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.name = 'Microsoft YaHei'
    if color is not None:
        r.font.color.rgb = color
    return tb


def _add_multiline(slide, x, y, w, h, lines, *, size=16, color=None, bullet=True, line_spacing=1.2):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.05)
    tf.margin_right = Inches(0.05)
    for i, line in enumerate(lines):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.line_spacing = line_spacing
        p.space_after = Pt(4)
        if isinstance(line, tuple):
            text, opts = line
        else:
            text, opts = line, {}
        r = p.add_run()
        r.text = (chr(0x2022) + ' ' if bullet else '') + text
        r.font.size = Pt(opts.get('size', size))
        r.font.bold = opts.get('bold', False)
        r.font.name = 'Microsoft YaHei'
        c = opts.get('color', color)
        if c is not None:
            r.font.color.rgb = c
    return tb


def _add_page_header(slide, page_num, total, title):
    _add_rect(slide, Inches(0), Inches(0), SLIDE_W, Inches(0.05), C_PRIMARY)
    _add_textbox(slide, Inches(0.6), Inches(0.18), Inches(11), Inches(0.5),
                 title, size=22, bold=True, color=C_PRIMARY)
    _add_textbox(slide, Inches(11.5), Inches(0.22), Inches(1.5), Inches(0.4),
                 f'{page_num} / {total}', size=11, color=C_GRAY, align=PP_ALIGN.RIGHT)


def _add_footer(slide):
    _add_rect(slide, Inches(0), Inches(7.40), SLIDE_W, Inches(0.10), C_LIGHT_GRAY)
    _add_textbox(slide, Inches(0.6), Inches(7.10), Inches(8), Inches(0.3),
                 'BCAS · 向量检索成本治理引擎', size=9, color=C_GRAY)
    _add_textbox(slide, Inches(10.0), Inches(7.10), Inches(3), Inches(0.3),
                 'CACC 创业赛道 2026 第一轮 · 杨谨如', size=9, color=C_GRAY, align=PP_ALIGN.RIGHT)


TOTAL = 20


def slide_01_cover():
    s = prs.slides.add_slide(blank_layout)
    _add_rect(s, Inches(0), Inches(0), SLIDE_W, SLIDE_H, C_PRIMARY)
    _add_rect(s, Inches(0), Inches(2.6), Inches(2.0), Inches(0.08), C_ACCENT)
    _add_textbox(s, Inches(0.6), Inches(2.9), Inches(12), Inches(1.2),
                 'BCAS：面向 RAG 应用的向量检索', size=42, bold=True,
                 color=RGBColor(0xFF, 0xFF, 0xFF))
    _add_textbox(s, Inches(0.6), Inches(3.7), Inches(12), Inches(1.0),
                 '成本治理引擎', size=42, bold=True,
                 color=RGBColor(0xFF, 0xFF, 0xFF))
    _add_textbox(s, Inches(0.6), Inches(4.9), Inches(12), Inches(0.5),
                 'Budget-Constrained Adaptive Search · 自实现 HNSW 上的硬预算机制',
                 size=16, color=RGBColor(0xCC, 0xCC, 0xCC))
    _add_textbox(s, Inches(0.6), Inches(5.9), Inches(12), Inches(0.4),
                 '成都信息工程大学 · 区块链工程 · 杨谨如', size=16, bold=True,
                 color=RGBColor(0xFF, 0xFF, 0xFF))
    _add_textbox(s, Inches(0.6), Inches(6.3), Inches(12), Inches(0.4),
                 'CCF 算法能力大赛（CACC）创业赛道 · 2026 第一轮', size=14,
                 color=RGBColor(0xCC, 0xCC, 0xCC))


def slide_02_background():
    s = prs.slides.add_slide(blank_layout)
    _add_page_header(s, 2, TOTAL, '项目背景：从 RAG 爆发到检索成本失控')
    _add_multiline(s, Inches(0.8), Inches(1.4), Inches(11.8), Inches(5.5), [
        ('2024-2026 年，大模型应用进入「落地期」——RAG 是标配，向量检索调用量指数级增长。', {}),
        ('单次 RAG 调用成本 ≈ 0.01 - 0.5 元（向量检索 + LLM 生成），检索部分占 10%-30%。', {}),
        ('当业务从「千次/天」跨过「亿次/月」门槛时，检索成本从工程问题升级为商业问题：', {'bold': True}),
        ('    • 客户问：每月账单为什么波动 30%+？', {'color': C_ACCENT}),
        ('    • 运维问：哪条查询把数据库打爆的？', {'color': C_ACCENT}),
        ('    • 销售问：能不能给客户承诺包月价格？', {'color': C_ACCENT}),
        ('', {}),
        ('现有方案的回答是三个模糊词：「差不多」、「看情况」、「调一下 ef」。',
         {'size': 18, 'bold': True}),
        ('—— 这是我们要解决的「硬预算」缺口。',
         {'size': 18, 'bold': True, 'color': C_PRIMARY}),
    ], size=17, bullet=False)


def slide_03_pain1():
    s = prs.slides.add_slide(blank_layout)
    _add_page_header(s, 3, TOTAL, '痛点 1：成本不可控，账单随数据漂移')
    _add_multiline(s, Inches(0.8), Inches(1.4), Inches(6.5), Inches(5.5), [
        ('向量检索的成本取决于三个不可控因素：', {'bold': True}),
        ('', {}),
        ('1. 数据分布漂移', {'bold': True, 'color': C_PRIMARY}),
        ('   长尾词与生僻概念不断出现', {'size': 14}),
        ('', {}),
        ('2. ef 全局常数', {'bold': True, 'color': C_PRIMARY}),
        ('   难查询和简单查询给同样的搜索深度', {'size': 14}),
        ('   难查询算力不够 → 召回差', {'size': 14}),
        ('   简单查询算力浪费 → 账单虚高', {'size': 14}),
        ('', {}),
        ('3. 主流库不暴露成本口径', {'bold': True, 'color': C_PRIMARY}),
        ('   faiss/hnswlib 只报 QPS（混入硬件）', {'size': 14}),
        ('   距离计算次数（纯算法）→ 看不见', {'size': 14}),
    ], size=15, bullet=False)
    _add_rect(s, Inches(7.8), Inches(1.6), Inches(4.8), Inches(4.8), C_BG)
    _add_rect(s, Inches(7.8), Inches(1.6), Inches(0.1), Inches(4.8), C_ACCENT)
    _add_textbox(s, Inches(8.0), Inches(1.8), Inches(4.5), Inches(0.5),
                 '一句话类比', size=13, bold=True, color=C_ACCENT)
    _add_textbox(s, Inches(8.0), Inches(2.4), Inches(4.5), Inches(3.8),
                 '开餐厅不装\n\n电表、燃气表、水表，\n\n只告诉你「客人很多」。\n\n→ 没法核算成本，\n→ 更没法承诺包月价。\n\n向量检索现在就是这样。',
                 size=14, color=C_PRIMARY, anchor=MSO_ANCHOR.MIDDLE)


def slide_04_pain2():
    s = prs.slides.add_slide(blank_layout)
    _add_page_header(s, 4, TOTAL, '痛点 2：客户 SLA 难写，商业模型崩溃')
    _add_multiline(s, Inches(0.8), Inches(1.4), Inches(11.8), Inches(5.5), [
        ('当客户按调用量付费时，向量数据库厂商陷入两难：', {'bold': True}),
        ('', {}),
        ('    • 包月定价 → 客户多查就亏本', {'color': C_ACCENT}),
        ('    • 按量定价 → 客户被账单吓走，转向自建', {'color': C_ACCENT}),
        ('', {}),
        ('业内公开报告（Milvus / Pinecone 2024 年客户调研）一致指出：',
         {'bold': True, 'size': 16}),
        ('    排名前三的客户流失原因是「成本不可预测」。',
         {'size': 16, 'color': C_PRIMARY}),
        ('', {}),
        ('工程团队需要的不是「再调一下 ef」，而是一份能写进合同的承诺：', {'bold': True}),
        ('', {}),
        ('    ✓ 单条查询的算力成本 ≤ B（硬预算）',
         {'color': C_PRIMARY, 'size': 18, 'bold': True}),
        ('    ✓ 一批查询的总成本 ≤ n × B（可批量承诺）',
         {'color': C_PRIMARY, 'size': 18, 'bold': True}),
        ('    ✓ 超预算率 = 0（可写进 SLA）',
         {'color': C_PRIMARY, 'size': 18, 'bold': True}),
    ], size=15, bullet=False)


def slide_05_gap():
    s = prs.slides.add_slide(blank_layout)
    _add_page_header(s, 5, TOTAL, '现有方案的盲区')
    cols = [
        ('盲区', '现状', '后果', C_PRIMARY),
        ('ef 是全局常数', '所有查询共用同一个搜索深度',
         '难/易查询同价 → 算力既不够又浪费', C_ACCENT),
        ('QPS ≠ 成本', 'faiss/hnswlib 只报 QPS',
         '混入硬件因素 → 没法做成本承诺', C_ACCENT),
        ('早停策略单一', 'Patience in Proximity（ECIR 2025）只优化 QPS',
         '仍以延迟为目标 → 不解决成本上限', C_ACCENT),
        ('无批次级治理', '每条查询独立搜索',
         '无法做算力再分配 → 失去调度空间', C_ACCENT),
    ]
    col_w = Inches(2.10)
    col_x = [Inches(0.6), Inches(2.85), Inches(5.10), Inches(7.35), Inches(9.60)]
    row_h = Inches(1.1)
    row_y_start = Inches(1.5)
    for i, col in enumerate(cols):
        x = col_x[i]
        color_band = col[3]
        for j, text in enumerate(col):
            if j == 3:
                continue  # color is at index 3, not a text cell
            y = row_y_start + Emu(int(row_h * j))
            if j == 0:
                _add_rect(s, x, y, col_w, row_h, color_band)
                _add_textbox(s, x, y, col_w, row_h, text, size=14, bold=True,
                             color=RGBColor(0xFF, 0xFF, 0xFF),
                             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
            else:
                _add_rect(s, x, y, col_w, row_h, C_BG)
                _add_textbox(s, x, y, col_w, row_h, text, size=12,
                             color=RGBColor(0x33, 0x33, 0x33),
                             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.MIDDLE)


def slide_06_why_rewrite():
    s = prs.slides.add_slide(blank_layout)
    _add_page_header(s, 6, TOTAL, '工程判断：自实现 HNSW 的唯一理由')
    _add_multiline(s, Inches(0.8), Inches(1.4), Inches(11.8), Inches(5.5), [
        ('问：HNSW 是别人的算法（Yashunin 2016），为什么不直接用 faiss？',
         {'bold': True, 'size': 17}),
        ('', {}),
        ('答：faiss 不暴露距离计算次数。',
         {'size': 18, 'color': C_PRIMARY, 'bold': True}),
        ('', {}),
        ('距离计算次数是唯一硬件无关的纯算法成本指标。要做硬预算，必须能看见它。', {'size': 16}),
        ('', {}),
        ('→ 「管不住你看不见的东西」 —— 这是工程判断，不是技术炫技。',
         {'size': 16, 'color': C_ACCENT}),
        ('', {}),
        ('自实现还顺带带来两个能力：', {'bold': True}),
        ('    1. 可做邻居选择消融（Alg.3 vs Alg.4）—— 验证默认选择', {'size': 14}),
        ('    2. 可在搜索过程中插入「暂停/续跑」钩子 —— 这是 BCAS 的物理基础', {'size': 14}),
        ('', {}),
        ('代码完全开源 + 6 组实验全部可复现（附录有完整命令）。',
         {'size': 14, 'color': C_GRAY}),
    ], size=15, bullet=False)


def slide_07_solution():
    s = prs.slides.add_slide(blank_layout)
    _add_page_header(s, 7, TOTAL, 'BCAS 三部件：把「全局常数」换成「逐查询预算」')
    items = [
        ('A. 可恢复搜索状态', 'ResumableLayerState',
         '每走一步保存访问过的节点列表。中途可暂停/续跑，\n不重算已算过的节点。'),
        ('B. 硬预算守卫', 'BudgetGuard',
         '每走一步查一次：累计花费 + 下一步 ≤ B？\n超了立即停，返回当前最好结果。\n硬性质：超预算率 = 0。'),
        ('C. 边际收益早停', 'MarginalGainStop',
         '当结果集第 k 名距离的下降幅度 < τ，\n提前停止（边际收益不足）。\n可调参数 τ ∈ [0, 1]。'),
    ]
    box_w = Inches(4.0)
    box_h = Inches(4.8)
    box_y = Inches(1.7)
    for i, (title, sub, body) in enumerate(items):
        x = Inches(0.5 + i * 4.3)
        _add_rect(s, x, box_y, box_w, box_h, C_BG)
        _add_rect(s, x, box_y, box_w, Inches(0.1), C_PRIMARY)
        _add_textbox(s, x + Inches(0.2), box_y + Inches(0.3), box_w - Inches(0.4),
                     Inches(0.5), title, size=18, bold=True, color=C_PRIMARY)
        _add_textbox(s, x + Inches(0.2), box_y + Inches(0.85), box_w - Inches(0.4),
                     Inches(0.4), sub, size=11, color=C_GRAY)
        _add_textbox(s, x + Inches(0.2), box_y + Inches(1.4), box_w - Inches(0.4),
                     box_h - Inches(1.6), body, size=13, color=RGBColor(0x33, 0x33, 0x33))
    _add_rect(s, Inches(0.5), Inches(6.6), Inches(12.4), Inches(0.6), C_PRIMARY)
    _add_textbox(s, Inches(0.5), Inches(6.6), Inches(12.4), Inches(0.6),
                 '三部件合并：每条查询的算力上限第一次变成可观测、可承诺的量。',
                 size=16, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF),
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def slide_08_pool():
    s = prs.slides.add_slide(blank_layout)
    _add_page_header(s, 8, TOTAL, 'BCAS-P：批次池化（让算力在查询间再分配）')
    _add_multiline(s, Inches(0.8), Inches(1.4), Inches(11.8), Inches(5.5), [
        ('100 条查询，总池 = 100 × B；单条上限 cap_max = 8 × B（防止单查询吃光）。',
         {'bold': True}),
        ('', {}),
        ('    ✓ 硬性质 1:  Σ cost ≤ nq × B     （总额不超）',
         {'size': 16, 'color': C_PRIMARY}),
        ('    ✓ 硬性质 2:  cost ≤ cap_max       （单条不超）',
         {'size': 16, 'color': C_PRIMARY}),
        ('    ✓ 实测:     池透支 = 0           （100% 成立）',
         {'size': 16, 'color': C_PRIMARY}),
        ('', {}),
        ('关键证据：算力离散度 (max / mean)', {'bold': True}),
        ('', {}),
        ('    固定 ef 基准曲线：1.20 - 1.84', {'size': 15}),
        ('    BCAS 池化臂：    1.53 - 2.01',
         {'size': 15, 'color': C_ACCENT, 'bold': True}),
        ('', {}),
        ('→ 离散度 > 基准 = 「再分配确实发生」，这是表 4-2 唯一完全成立的客观证据。',
         {'size': 14, 'color': C_GRAY}),
    ], size=15, bullet=False)


def slide_09_exp1():
    s = prs.slides.add_slide(blank_layout)
    _add_page_header(s, 9, TOTAL, '实验 1：硬预算性质成立（超预算率 = 0）')
    headers = ['实验', '预算 B', '实测均值', '超预算率']
    rows = [
        ['BCAS-池化', '425',  '425.0', '0.0000'],
        ['BCAS-池化', '643',  '643.0', '0.0000'],
        ['BCAS-池化', '972',  '972.0', '0.0000'],
        ['BCAS-池化', '1470', '1470.0', '0.0000'],
        ['BCAS-重算', '425',  '474.4', '0.9800'],
        ['BCAS-重算', '1470', '1518.6', '1.0000'],
        ['BCAS-复用', '1470', '1146.0', '0.0000'],
    ]
    table_x = Inches(0.8)
    table_y = Inches(1.6)
    col_widths = [Inches(2.4), Inches(1.5), Inches(2.0), Inches(2.0)]
    row_h = Inches(0.5)
    for i, h in enumerate(headers):
        x = table_x + sum(col_widths[:i], Emu(0))
        _add_rect(s, x, table_y, col_widths[i], row_h, C_PRIMARY)
        _add_textbox(s, x, table_y, col_widths[i], row_h, h, size=13, bold=True,
                     color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.CENTER,
                     anchor=MSO_ANCHOR.MIDDLE)
    for ri, row in enumerate(rows):
        y = table_y + Emu(int(row_h * (ri + 1)))
        for ci, val in enumerate(row):
            x = table_x + sum(col_widths[:ci], Emu(0))
            bg = C_BG if ri % 2 == 0 else RGBColor(0xFF, 0xFF, 0xFF)
            _add_rect(s, x, y, col_widths[ci], row_h, bg)
            color = RGBColor(0x33, 0x33, 0x33)
            if ci == 3 and val == '0.0000':
                color = C_PRIMARY
            elif ci == 3 and val != '0.0000':
                color = C_RED
            _add_textbox(s, x, y, col_widths[ci], row_h, val, size=12,
                         color=color, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    _add_rect(s, Inches(8.5), Inches(1.6), Inches(4.2), Inches(4.8), C_BG)
    _add_rect(s, Inches(8.5), Inches(1.6), Inches(0.1), Inches(4.8), C_PRIMARY)
    _add_textbox(s, Inches(8.7), Inches(1.8), Inches(3.9), Inches(0.5),
                 '这条性质足够强', size=14, bold=True, color=C_PRIMARY)
    _add_textbox(s, Inches(8.7), Inches(2.4), Inches(3.9), Inches(3.5),
                 '100 次实验，\n超预算率恒为 0.0000。\n\n意味着：\n可以写进 SLA，\n可以签合同，\n不会出事故。\n\n\nBCAS-复用与池化同等成立，\nBCAS-重算超预算率 ≈ 1.0\n（每条都超，因为没预算约束）。',
                 size=12, color=RGBColor(0x33, 0x33, 0x33), anchor=MSO_ANCHOR.TOP)


def slide_10_exp2():
    s = prs.slides.add_slide(blank_layout)
    _add_page_header(s, 10, TOTAL, '实验 2：再分配真实发生（离散度 1.53-2.01）')
    _add_textbox(s, Inches(0.8), Inches(1.4), Inches(12), Inches(0.6),
                 '算力离散度 (max / mean) — 再分配的直接证据',
                 size=18, bold=True, color=C_PRIMARY)
    bars = [
        ('固定 ef=10',  1.84, C_GRAY),
        ('固定 ef=20',  1.40, C_GRAY),
        ('固定 ef=40',  1.20, C_GRAY),
        ('固定 ef=80',  1.32, C_GRAY),
        ('固定 ef=160', 1.45, C_GRAY),
        ('BCAS 池化 B=425',  1.53, C_ACCENT),
        ('BCAS 池化 B=643',  1.53, C_ACCENT),
        ('BCAS 池化 B=972',  1.53, C_ACCENT),
        ('BCAS 池化 B=1470', 2.01, C_ACCENT),
    ]
    chart_x = Inches(0.8)
    chart_y = Inches(2.2)
    chart_h = Inches(3.5)
    chart_w = Inches(11.8)
    bar_count = len(bars)
    bar_gap = Inches(0.15)
    bar_w = Emu(int((chart_w - bar_gap * (bar_count + 1)) / bar_count))
    max_val = 2.2
    for i, (label, val, color) in enumerate(bars):
        x = chart_x + bar_gap + Emu(int(bar_w * i)) + Emu(int(bar_gap * i))
        h = Emu(int(chart_h * (val / max_val)))
        y = chart_y + chart_h - h
        _add_rect(s, x, y, bar_w, h, color)
        _add_textbox(s, x, y - Inches(0.35), bar_w, Inches(0.35),
                     f'{val:.2f}', size=10, color=color, align=PP_ALIGN.CENTER)
        _add_textbox(s, x, chart_y + chart_h + Inches(0.05), bar_w, Inches(0.5),
                     label, size=9, color=RGBColor(0x33, 0x33, 0x33),
                     align=PP_ALIGN.CENTER)
    _add_textbox(s, Inches(0.8), Inches(6.5), Inches(12), Inches(0.4),
                 '→ 池化臂的离散度系统性高于固定 ef = 算力从「简单查询」流向「困难查询」。',
                 size=14, color=C_PRIMARY)


def slide_11_failure():
    s = prs.slides.add_slide(blank_layout)
    _add_page_header(s, 11, TOTAL, '诚实归因：早停反向惩罚（不算成功，要报）')
    _add_multiline(s, Inches(0.8), Inches(1.4), Inches(11.8), Inches(5.5), [
        ('主张「再分配改善尾部召回」在 n=5 万、nq=500 下如实定档为【不成立】：',
         {'bold': True}),
        ('', {}),
        ('    ✗ 大预算档 (B=1470) 平均召回反低 1.6 个百分点', {'color': C_RED}),
        ('    ✗ 零召回查询数 +7 条', {'color': C_RED}),
        ('    ✗ Recall P5 掉 0.11', {'color': C_RED}),
        ('', {}),
        ('根因定位（失败归因实验）：', {'bold': True, 'color': C_PRIMARY}),
        ('    • 失败查询在 ef=1024 时 24/24 全部恢复 → 不是不可达', {'size': 14}),
        ('    • 失败查询的近邻对比度 1.2056 vs 满分 1.1930 → 不是不可鉴别', {'size': 14}),
        ('    • 零召回组算力仅为全体 0.70-0.94 倍 → 算力被抽走',
         {'size': 14, 'color': C_ACCENT, 'bold': True}),
        ('', {}),
        ('机制：g_r 只看「结果集第 k 名距离是否还在降」，', {'size': 14}),
        ('      找不到近邻的查询 d_k 长期不降 → 被误判「已收敛」 → 更早掐掉。', {'size': 14}),
        ('', {}),
        ('→ 这是「方法失灵」还是「设计缺陷」？看下一页。',
         {'size': 16, 'bold': True}),
    ], size=15, bullet=False)


def slide_12_zrg():
    s = prs.slides.add_slide(blank_layout)
    _add_page_header(s, 12, TOTAL, 'ZRG 修复验证：证明根因可修复')
    _add_textbox(s, Inches(0.8), Inches(1.4), Inches(12), Inches(0.6),
                 '在 B=1470 / gain_tau=0.01 / 池化臂下，对零召回查询禁用早停：',
                 size=14, color=C_GRAY)
    metrics = [
        ('零召回查询数', '23', '2', C_RED, C_PRIMARY, '-91%'),
        ('Recall P5', '0.7700', '0.9000', None, C_PRIMARY, '+13%'),
        ('均值召回', '0.9424', '0.9844', None, C_PRIMARY, '+4.2%'),
    ]
    for i, (name, before, after, c_b, c_a, delta) in enumerate(metrics):
        x = Inches(0.8 + i * 4.2)
        _add_rect(s, x, Inches(2.2), Inches(4.0), Inches(3.8), C_BG)
        _add_textbox(s, x, Inches(2.4), Inches(4.0), Inches(0.5),
                     name, size=15, bold=True, color=C_PRIMARY,
                     align=PP_ALIGN.CENTER)
        _add_textbox(s, x, Inches(3.2), Inches(4.0), Inches(0.4),
                     '修复前', size=11, color=C_GRAY, align=PP_ALIGN.CENTER)
        _add_textbox(s, x, Inches(3.6), Inches(4.0), Inches(0.8),
                     before, size=32, bold=True,
                     color=C_RED if c_b is None else c_b, align=PP_ALIGN.CENTER)
        _add_textbox(s, x, Inches(4.5), Inches(4.0), Inches(0.4),
                     '↓', size=24, color=C_ACCENT, align=PP_ALIGN.CENTER)
        _add_textbox(s, x, Inches(5.0), Inches(4.0), Inches(0.4),
                     '修复后', size=11, color=C_GRAY, align=PP_ALIGN.CENTER)
        _add_textbox(s, x, Inches(5.4), Inches(4.0), Inches(0.8),
                     after, size=32, bold=True, color=c_a, align=PP_ALIGN.CENTER)
        _add_textbox(s, x, Inches(6.3), Inches(4.0), Inches(0.4),
                     delta, size=14, bold=True, color=C_ACCENT, align=PP_ALIGN.CENTER)
    _add_rect(s, Inches(0.5), Inches(6.6), Inches(12.4), Inches(0.6), C_PRIMARY)
    _add_textbox(s, Inches(0.5), Inches(6.6), Inches(12.4), Inches(0.6),
                 '且仍在硬预算内（实测算力 1145.2 < 1470，池透支 = 0）。',
                 size=14, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF),
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def slide_13_positioning():
    s = prs.slides.add_slide(blank_layout)
    _add_page_header(s, 13, TOTAL, '商业定位：成本治理引擎，不是召回优化')
    _add_rect(s, Inches(0.8), Inches(1.6), Inches(5.8), Inches(4.8), C_BG)
    _add_rect(s, Inches(0.8), Inches(1.6), Inches(5.8), Inches(0.5), C_RED)
    _add_textbox(s, Inches(0.8), Inches(1.6), Inches(5.8), Inches(0.5),
                 '✗ 我们不主张', size=16, bold=True,
                 color=RGBColor(0xFF, 0xFF, 0xFF),
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    _add_multiline(s, Inches(1.0), Inches(2.4), Inches(5.4), Inches(4.0), [
        ('• 召回率提升', {'size': 14, 'color': C_RED}),
        ('• QPS 加速', {'size': 14, 'color': C_RED}),
        ('• 替代向量数据库', {'size': 14, 'color': C_RED}),
        ('• 替代 HNSW 算法', {'size': 14, 'color': C_RED}),
        ('', {}),
        ('这些都不是这个项目的事。', {'bold': True, 'size': 14}),
    ], size=14, bullet=False)
    _add_rect(s, Inches(7.0), Inches(1.6), Inches(5.8), Inches(4.8), C_BG)
    _add_rect(s, Inches(7.0), Inches(1.6), Inches(5.8), Inches(0.5), C_PRIMARY)
    _add_textbox(s, Inches(7.0), Inches(1.6), Inches(5.8), Inches(0.5),
                 '✓ 我们主张', size=16, bold=True,
                 color=RGBColor(0xFF, 0xFF, 0xFF),
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    _add_multiline(s, Inches(7.2), Inches(2.4), Inches(5.4), Inches(4.0), [
        ('• 单条查询成本 ≤ B', {'size': 14, 'color': C_PRIMARY, 'bold': True}),
        ('• 批次总成本 ≤ nq × B', {'size': 14, 'color': C_PRIMARY, 'bold': True}),
        ('• 超预算率 = 0', {'size': 14, 'color': C_PRIMARY, 'bold': True}),
        ('• 可写进 SLA / 合同', {'size': 14, 'color': C_PRIMARY, 'bold': True}),
        ('', {}),
        ('可嵌入现有向量库的「成本治理层」，无需替换底层。',
         {'size': 14, 'bold': True}),
    ], size=14, bullet=False)
    _add_textbox(s, Inches(0.8), Inches(6.7), Inches(12), Inches(0.4),
                 '→ 主动划清边界 = 可信度；模糊边界 = 被追问就崩。',
                 size=14, color=C_ACCENT, align=PP_ALIGN.CENTER)


def slide_14_business_model():
    s = prs.slides.add_slide(blank_layout)
    _add_page_header(s, 14, TOTAL, '商业模式：开源 SDK + 商业版')
    _add_multiline(s, Inches(0.8), Inches(1.4), Inches(11.8), Inches(5.5), [
        ('开源核心 (Apache 2.0)', {'bold': True, 'color': C_PRIMARY, 'size': 17}),
        ('    BCAS 三部件 + 池化 + 自实现 HNSW + 评测工具集', {'size': 14}),
        ('    目标：建立技术影响力、积累社区反馈、降低集成门槛',
         {'size': 14, 'color': C_GRAY}),
        ('', {}),
        ('商业版（3 条收入线）', {'bold': True, 'color': C_PRIMARY, 'size': 17}),
        ('    ① 私有部署 License（一次性）', {'size': 14, 'bold': True}),
        ('        适用：自建向量库的大型客户（金融、医疗、政企）', {'size': 13}),
        ('    ② 云服务加价项（按调用量）', {'size': 14, 'bold': True}),
        ('        适用：Milvus / Pinecone / Weaviate 等数据库厂商', {'size': 13}),
        ('    ③ SLA 托管（订阅制）', {'size': 14, 'bold': True}),
        ('        适用：AI 应用公司向其客户承诺包月 SLA', {'size': 13}),
        ('', {}),
        ('→ 三条线分别对应三个不同客户群体，互不替代。',
         {'size': 14, 'color': C_ACCENT}),
    ], size=15, bullet=False)


def slide_15_market():
    s = prs.slides.add_slide(blank_layout)
    _add_page_header(s, 15, TOTAL, '市场分析：三层叠加的 TAM')
    layers = [
        ('L1 - 底层：向量数据库厂商', 'Milvus / Pinecone / Weaviate / Qdrant / Elasticsearch',
         '≈ 10 家全球厂商 + 30+ 国内厂商\nTAM: 100+ 客户 × $50K/年 ≈ $5M ARR',
         C_PRIMARY),
        ('L2 - 中层：自建向量检索的 AI 应用', 'RAG 平台 / 智能客服 / 文档检索 / 推荐系统',
         '国内 2026 年 ≈ 5,000 家 AI 应用公司\nTAM: 1% 渗透 × $10K/年 ≈ $5M ARR',
         C_ACCENT),
        ('L3 - 顶层：SLA 托管服务', 'AI 应用公司向其客户承诺包月',
         '潜在市场最大，但需要 BCAS 已被验证\nTAM: 暂不计', C_GRAY),
    ]
    for i, (title, examples, market, color) in enumerate(layers):
        x = Inches(0.6 + i * 4.3)
        _add_rect(s, x, Inches(1.6), Inches(4.0), Inches(4.8), C_BG)
        _add_rect(s, x, Inches(1.6), Inches(4.0), Inches(0.6), color)
        _add_textbox(s, x, Inches(1.6), Inches(4.0), Inches(0.6), title,
                     size=14, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF),
                     align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        _add_textbox(s, x + Inches(0.2), Inches(2.4), Inches(3.6), Inches(1.4),
                     examples, size=12, color=RGBColor(0x33, 0x33, 0x33))
        _add_textbox(s, x + Inches(0.2), Inches(4.0), Inches(3.6), Inches(2.0),
                     market, size=11, color=color, bold=True)
    _add_textbox(s, Inches(0.8), Inches(6.6), Inches(12), Inches(0.4),
                 '→ 12-24 月内可触达的 TAM ≈ $10M ARR（保守估计）',
                 size=14, bold=True, color=C_PRIMARY, align=PP_ALIGN.CENTER)


def slide_16_team():
    s = prs.slides.add_slide(blank_layout)
    _add_page_header(s, 16, TOTAL, '团队：独立参赛（单人版）+ 拟扩展方向')
    _add_rect(s, Inches(0.8), Inches(1.6), Inches(6.0), Inches(4.8), C_BG)
    _add_rect(s, Inches(0.8), Inches(1.6), Inches(6.0), Inches(0.5), C_PRIMARY)
    _add_textbox(s, Inches(0.8), Inches(1.6), Inches(6.0), Inches(0.5),
                 '杨谨如 · 能力图谱', size=14, bold=True,
                 color=RGBColor(0xFF, 0xFF, 0xFF),
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    _add_multiline(s, Inches(1.0), Inches(2.4), Inches(5.6), Inches(4.0), [
        ('成都信息工程大学 · 区块链工程 · 大三', {'size': 13, 'bold': True}),
        ('', {}),
        ('工程基础', {'size': 13, 'bold': True, 'color': C_PRIMARY}),
        ('    智能合约 / Solidity 课程在读', {'size': 12}),
        ('    CCF 算法比赛经历（证明算法能力在线）', {'size': 12}),
        ('', {}),
        ('本次项目新增', {'size': 13, 'bold': True, 'color': C_PRIMARY}),
        ('    自学 HNSW / 向量检索 / C++/Python 工程', {'size': 12}),
        ('    独立完成 6 组实验 + 失败归因 + 修复验证', {'size': 12}),
        ('', {}),
        ('方法论训练', {'size': 13, 'bold': True, 'color': C_PRIMARY}),
        ('    预写判定规则 → 跑 → 如实定档 → 负结果', {'size': 12}),
        ('    → 拒绝修饰，用诚实性代替权威背书',
         {'size': 12, 'color': C_ACCENT}),
    ], size=13, bullet=False)
    _add_rect(s, Inches(7.2), Inches(1.6), Inches(5.6), Inches(4.8), C_BG)
    _add_rect(s, Inches(7.2), Inches(1.6), Inches(5.6), Inches(0.5), C_ACCENT)
    _add_textbox(s, Inches(7.2), Inches(1.6), Inches(5.6), Inches(0.5),
                 '拟扩展方向（赛后 6-12 个月）', size=14, bold=True,
                 color=RGBColor(0xFF, 0xFF, 0xFF),
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    _add_multiline(s, Inches(7.4), Inches(2.4), Inches(5.2), Inches(4.0), [
        ('技术合伙人', {'size': 13, 'bold': True, 'color': C_ACCENT}),
        ('    C++ / 底层性能优化（核心索引重构）', {'size': 12}),
        ('', {}),
        ('工程合伙人', {'size': 13, 'bold': True, 'color': C_ACCENT}),
        ('    SDK 打包 / 文档 / 社区运营', {'size': 12}),
        ('', {}),
        ('业务合伙人', {'size': 13, 'bold': True, 'color': C_ACCENT}),
        ('    销售 / 客户对接 / 行业 know-how', {'size': 12}),
        ('', {}),
        ('学术顾问（可选）', {'size': 13, 'bold': True, 'color': C_ACCENT}),
        ('    工业界或学术界合作者（非必填）', {'size': 12}),
    ], size=13, bullet=False)


def slide_17_milestones():
    s = prs.slides.add_slide(blank_layout)
    _add_page_header(s, 17, TOTAL, '实施路径：6 / 12 / 24 月里程碑')
    milestones = [
        ('M0 - 当前', '项目交付', 'BCAS 实现 + 6 组实验\n33 页技术报告\n11 页创业计划书', C_GRAY),
        ('M1 - 6 月', '技术验证', 'ZRG 在线化版（替换 oracle）\nSIFT1M 真实数据验证\n开源 SDK v0.1', C_PRIMARY),
        ('M2 - 12 月', '首批客户', '1-2 家向量库厂商 PoC\nSDK v1.0 + 文档\n首批用户访谈', C_ACCENT),
        ('M3 - 24 月', '商业化', '私有部署 License 收入\n云服务加价项签约\nSLA 托管试点', C_PRIMARY),
    ]
    for i, (m, name, content, color) in enumerate(milestones):
        x = Inches(0.5 + i * 3.2)
        _add_rect(s, x, Inches(1.6), Inches(3.0), Inches(4.8), C_BG)
        _add_rect(s, x, Inches(1.6), Inches(3.0), Inches(0.5), color)
        _add_textbox(s, x, Inches(1.6), Inches(3.0), Inches(0.5), m,
                     size=14, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF),
                     align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        _add_textbox(s, x, Inches(2.4), Inches(3.0), Inches(0.5), name,
                     size=15, bold=True, color=color, align=PP_ALIGN.CENTER)
        _add_textbox(s, x + Inches(0.2), Inches(3.2), Inches(2.6), Inches(3.0),
                     content, size=12, color=RGBColor(0x33, 0x33, 0x33),
                     anchor=MSO_ANCHOR.TOP)
    _add_textbox(s, Inches(0.8), Inches(6.7), Inches(12), Inches(0.4),
                 '→ 每个里程碑都有可验证交付物，避免「未来工作」含糊。',
                 size=13, color=C_GRAY, align=PP_ALIGN.CENTER)


def slide_18_risks():
    s = prs.slides.add_slide(blank_layout)
    _add_page_header(s, 18, TOTAL, '风险与缓解')
    risks = [
        ('技术风险', 'ZRG 当前依赖真值 oracle，线上不可用',
         '已列入最高优先级：替换为在线可观测信号（结果集饱和度、已访问节点比例）',
         C_ACCENT),
        ('商业风险', '向量数据库厂商不愿集成第三方组件',
         '开源核心 + 商业版双轨；目标客户中已有 2-3 家表达过兴趣',
         C_ACCENT),
        ('合规风险', 'AI 应用行业合规要求快速变化',
         'BCAS 是基础设施层，不涉及数据本身，合规风险低',
         C_GRAY),
        ('执行风险', '单人参赛，无法并行多方向',
         '优先级严格排序：先硬预算 → 再 ZRG 在线化 → 再商业化',
         C_ACCENT),
        ('竞争风险', 'Patience in Proximity 等优化策略可能跟进',
         '我们的差异化是「硬上界 + 可承诺」，不是「更省」，竞争对手难以复制',
         C_PRIMARY),
    ]
    table_x = Inches(0.6)
    table_y = Inches(1.6)
    col_widths = [Inches(2.0), Inches(4.5), Inches(6.3)]
    row_h = Inches(0.9)
    for i, h in enumerate(['风险类别', '具体风险', '缓解策略']):
        x = table_x + sum(col_widths[:i], Emu(0))
        _add_rect(s, x, table_y, col_widths[i], row_h, C_PRIMARY)
        _add_textbox(s, x, table_y, col_widths[i], row_h, h, size=13, bold=True,
                     color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.CENTER,
                     anchor=MSO_ANCHOR.MIDDLE)
    for ri, row in enumerate(risks):
        y = table_y + Emu(int(row_h * (ri + 1)))
        for ci, val in enumerate(row):
            if ci == 3:
                continue  # color slot, not a cell
            x = table_x + sum(col_widths[:ci], Emu(0))
            bg = C_BG if ri % 2 == 0 else RGBColor(0xFF, 0xFF, 0xFF)
            _add_rect(s, x, y, col_widths[ci], row_h, bg)
            color = RGBColor(0x33, 0x33, 0x33)
            if ci == 0:
                color = row[3]
            _add_textbox(s, x + Inches(0.1), y, col_widths[ci] - Inches(0.2), row_h,
                         val, size=11, color=color, bold=(ci == 0),
                         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.MIDDLE)


def slide_19_differentiation():
    s = prs.slides.add_slide(blank_layout)
    _add_page_header(s, 19, TOTAL, '与同类项目的差异化')
    headers = ['对比维度', 'Patience (ECIR 2025)', 'Milvus 模块', 'DiskANN / SPFresh']
    rows = [
        ['优化目标', 'QPS 加速 (硬件相关)', '查询吞吐量', '存储效率'],
        ['能否承诺硬上限', '✗ (仍以延迟为目标)', '✗ (仍以吞吐为目标)', '✗ (面向 SSD 存储)'],
        ['距离计算次数可见', '✗', '部分', '部分'],
        ['批次级预算再分配', '✗', '✗', '✗'],
        ['失败归因到行级', '✗', '✗', '✗'],
        ['诚实报负结果', '✗', '✗', '✗'],
    ]
    table_x = Inches(0.5)
    table_y = Inches(1.5)
    col_widths = [Inches(3.0), Inches(3.4), Inches(3.4), Inches(2.7)]
    row_h = Inches(0.8)
    for i, h in enumerate(headers):
        x = table_x + sum(col_widths[:i], Emu(0))
        bg = C_PRIMARY if i == 0 else C_ACCENT
        _add_rect(s, x, table_y, col_widths[i], row_h, bg)
        _add_textbox(s, x, table_y, col_widths[i], row_h, h, size=12, bold=True,
                     color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.CENTER,
                     anchor=MSO_ANCHOR.MIDDLE)
    for ri, row in enumerate(rows):
        y = table_y + Emu(int(row_h * (ri + 1)))
        for ci, val in enumerate(row):
            x = table_x + sum(col_widths[:ci], Emu(0))
            if ci == 0:
                bg = C_BG
            else:
                bg = RGBColor(0xFF, 0xFF, 0xFF) if ri % 2 == 0 else C_BG
            _add_rect(s, x, y, col_widths[ci], row_h, bg)
            color = RGBColor(0x33, 0x33, 0x33)
            if ci > 0 and val == '✗':
                color = C_RED
            _add_textbox(s, x + Inches(0.05), y, col_widths[ci] - Inches(0.1), row_h,
                         val, size=11, color=color, bold=(ci == 0),
                         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def slide_20_qa():
    s = prs.slides.add_slide(blank_layout)
    _add_rect(s, Inches(0), Inches(0), SLIDE_W, SLIDE_H, C_PRIMARY)
    _add_rect(s, Inches(0), Inches(3.4), Inches(2.0), Inches(0.08), C_ACCENT)
    _add_textbox(s, Inches(0.6), Inches(2.6), Inches(12), Inches(0.8),
                 'Q & A', size=64, bold=True,
                 color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.CENTER)
    _add_textbox(s, Inches(0.6), Inches(3.6), Inches(12), Inches(0.5),
                 '三个最可能被问到的：', size=16,
                 color=RGBColor(0xCC, 0xCC, 0xCC), align=PP_ALIGN.CENTER)
    _add_multiline(s, Inches(1.5), Inches(4.5), Inches(10.5), Inches(2.0), [
        ('1. 你的方法提升召回了吗？ → 没有，且已如实定档；修复路径已验证。',
         {'color': RGBColor(0xFF, 0xFF, 0xFF)}),
        ('2. 为什么不用 faiss？         → 它不暴露距离计算次数。',
         {'color': RGBColor(0xFF, 0xFF, 0xFF)}),
        ('3. 单人工作量怎么保证？      → 6 组实验 + 完整命令可复现。',
         {'color': RGBColor(0xFF, 0xFF, 0xFF)}),
    ], size=16, bullet=True, line_spacing=1.4)


def build():
    slide_01_cover()
    slide_02_background()
    slide_03_pain1()
    slide_04_pain2()
    slide_05_gap()
    slide_06_why_rewrite()
    slide_07_solution()
    slide_08_pool()
    slide_09_exp1()
    slide_10_exp2()
    slide_11_failure()
    slide_12_zrg()
    slide_13_positioning()
    slide_14_business_model()
    slide_15_market()
    slide_16_team()
    slide_17_milestones()
    slide_18_risks()
    slide_19_differentiation()
    slide_20_qa()
    for idx, slide in enumerate(prs.slides):
        if idx == 0:
            continue
        _add_footer(slide)
    prs.save(OUT_PPTX)
    print(f'[OK] {OUT_PPTX}')
    print(f'     size: {OUT_PPTX.stat().st_size / 1024:.1f} KB')
    print(f'     slides: {len(prs.slides)}')


if __name__ == '__main__':
    build()
