"""构建第三周汇报 deck：YOLO26 分割头学习对比 + 条码码制分类实验。

设计要点
--------
1. **两册各自成册，首尾相接**：第一册只留分割头的学习对比（封面、
   差异①–④、实测、结论、过程①②），底座里的前置结构页与附录按需裁掉；
   第二册是条码部分自己的封面 + 内容 + 收尾。两部分互不引用。
2. 新页的视觉参数（字号、留白、表头色、页脚）与底座完全一致，
   靠复刻同一个调色板与 helper 实现。
3. 页码在重排后会错位，所以最后统一重刷右下角的页码文本框。
4. 条码部分的数据全部来自仓库内 docs/ 与 .tmp_tests/ 的实测产物，
   不写回忆值。

输出：slides/第三周汇报.pptx
运行：python slides/build_report_deck.py
"""

from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

# ---------------------------------------------------------------- 路径与配色
# 底座 deck 与它的配图都在 week3 工作区（本脚本只读，不修改它）
BASE_DIR = Path(r"D:\Users\admin\Desktop\代码\week3\ultralytics-main")
BASE_PPTX = BASE_DIR / "YOLO11-vs-YOLO26-分割头对比.pptx"
FIGS = BASE_DIR / "note-figs"

OUT_DIR = Path(__file__).resolve().parent
OUT_PPTX = OUT_DIR / "第三周汇报.pptx"

# 与底座 deck 完全相同的调色板，保证两部分的视觉是同一套
INK = RGBColor(0x1F, 0x29, 0x37)
GRAY = RGBColor(0x6B, 0x72, 0x80)
LIGHT = RGBColor(0xF3, 0xF5, 0xF9)
BLUE = RGBColor(0x2F, 0x5D, 0x8A)
ORANGE = RGBColor(0xB2, 0x5A, 0x1B)
GREEN = RGBColor(0x2F, 0x6F, 0x4F)
RULE = RGBColor(0xD5, 0xDB, 0xE3)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
CJK = "Microsoft YaHei"
MONO = "Consolas"

W = Inches(13.333)
H = Inches(7.5)
MARGIN = Inches(0.62)
CONTENT_W = W - 2 * MARGIN


# ---------------------------------------------------------------- 基础 helper
def blank_layout(prs):
    """取"空白"版式；不同模板下标可能不同，所以按名字找、再退回下标 6。"""
    for layout in prs.slide_layouts:
        if str(layout.name).strip().lower() == "blank":
            return layout
    return prs.slide_layouts[6]


def tb(slide, x, y, w, h, text, size=14, bold=False, color=INK, font=CJK,
       align=PP_ALIGN.LEFT, line_spacing=1.2, anchor=MSO_ANCHOR.TOP):
    """放一个无边框文本框，逐行写入（每行一个 paragraph）。"""
    box = slide.shapes.add_textbox(x, y, w, h)
    frame = box.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = anchor
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0
    for index, line in enumerate(str(text).split("\n")):
        paragraph = (frame.paragraphs[0] if index == 0
                     else frame.add_paragraph())
        paragraph.alignment = align
        paragraph.line_spacing = line_spacing
        run = paragraph.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
        run.font.name = font
    return box


def header(slide, tag, title, subtitle=None):
    """页眉：小标签 + 大标题 + 蓝色短横线 + 可选副标题。"""
    tb(slide, MARGIN, Inches(0.34), CONTENT_W, Inches(0.26), tag,
       size=10.5, color=GRAY)
    tb(slide, MARGIN, Inches(0.60), CONTENT_W, Inches(0.52), title,
       size=25, bold=True)
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, MARGIN, Inches(1.20),
                                 Inches(0.86), Pt(3.2))
    bar.fill.solid()
    bar.fill.fore_color.rgb = BLUE
    bar.line.fill.background()
    bar.shadow.inherit = False
    if subtitle:
        tb(slide, MARGIN, Inches(1.34), CONTENT_W, Inches(0.34), subtitle,
           size=12.5, color=GRAY)


def footer(slide, note):
    """页脚：左侧标注数据来源，右侧页码（页码稍后统一重刷）。"""
    tb(slide, MARGIN, H - Inches(0.50), Inches(10.4), Inches(0.26), note,
       size=9, color=GRAY)
    tb(slide, W - MARGIN - Inches(1.2), H - Inches(0.50), Inches(1.2),
       Inches(0.26), "0", size=9, color=GRAY, align=PP_ALIGN.RIGHT)


def estimate_lines(text, width_emu, size_pt):
    """估算一段文字在给定宽度下会折成几行。

    按「半角单位」估算：ASCII 记 1 个单位、中文字符记 2 个单位，
    一个字符宽度近似等于字号的一半，于是可用单位数 = 宽度(pt) / (字号/2)。
    只用于排布下一条目之前留白，不作为最终渲染依据。
    """
    units = sum(1 if ord(ch) < 128 else 2 for ch in text)
    capacity = max(1.0, (width_emu / 12700.0) / (size_pt / 2.0))
    return max(1, int(-(-units // capacity)))


def bullets(slide, x, y, w, items, size=13.5, gap=Inches(0.10), dot=BLUE):
    """圆点条目。items 元素可以是 str，也可以是 (加粗前缀, 正文)。

    每条的行距按估算行数自适应，避免长句折行后压到下一条上。
    """
    cursor = y
    for item in items:
        lead, body = item if isinstance(item, tuple) else (None, item)
        marker = slide.shapes.add_shape(MSO_SHAPE.OVAL, x,
                                        cursor + Inches(0.075), Pt(6.5),
                                        Pt(6.5))
        marker.fill.solid()
        marker.fill.fore_color.rgb = dot
        marker.line.fill.background()
        marker.shadow.inherit = False
        box = slide.shapes.add_textbox(x + Inches(0.24), cursor,
                                       w - Inches(0.24), Inches(0.5))
        frame = box.text_frame
        frame.word_wrap = True
        frame.margin_left = frame.margin_right = 0
        frame.margin_top = frame.margin_bottom = 0
        paragraph = frame.paragraphs[0]
        paragraph.line_spacing = 1.25
        if lead:
            run = paragraph.add_run()
            run.text = lead
            run.font.size = Pt(size)
            run.font.bold = True
            run.font.color.rgb = INK
            run.font.name = CJK
        run = paragraph.add_run()
        run.text = body
        run.font.size = Pt(size)
        run.font.color.rgb = GRAY if lead else INK
        run.font.name = CJK
        full = (lead or "") + body
        lines = estimate_lines(full, w - Inches(0.24), size)
        cursor += Inches(0.22 * lines) + Inches(0.16) + gap
    return cursor


def table(slide, data, x, y, col_widths, row_h=Inches(0.38),
          head_h=Inches(0.42), fs=11, align_right_cols=(), head_fill=BLUE):
    """带表头的浅色斑马纹表格。"""
    rows, cols = len(data), len(data[0])
    shape = slide.shapes.add_table(rows, cols, x, y, sum(col_widths),
                                   head_h + row_h * (rows - 1))
    tbl = shape.table
    tbl.first_row = True
    for j, width in enumerate(col_widths):
        tbl.columns[j].width = width
    tbl.rows[0].height = head_h
    for i in range(1, rows):
        tbl.rows[i].height = row_h
    for i, row in enumerate(data):
        for j, value in enumerate(row):
            cell = tbl.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.09)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = (head_fill if i == 0 else
                                        (WHITE if i % 2 else LIGHT))
            paragraph = cell.text_frame.paragraphs[0]
            paragraph.alignment = (PP_ALIGN.RIGHT if j in align_right_cols
                                   else PP_ALIGN.LEFT)
            run = paragraph.add_run()
            run.text = str(value)
            run.font.size = Pt(fs)
            run.font.name = (MONO if j > 0 and str(value)[:1].isdigit()
                             else CJK)
            run.font.bold = i == 0
            run.font.color.rgb = WHITE if i == 0 else INK
    return tbl


def callout(slide, x, y, w, h, text, fill=RGBColor(0xFD, 0xF3, 0xE6),
            edge=ORANGE, fs=12):
    """浅底描边的强调块。"""
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    box.fill.solid()
    box.fill.fore_color.rgb = fill
    box.line.color.rgb = edge
    box.line.width = Pt(1)
    box.shadow.inherit = False
    frame = box.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    frame.margin_left = frame.margin_right = Inches(0.14)
    paragraph = frame.paragraphs[0]
    run = paragraph.add_run()
    run.text = text
    run.font.size = Pt(fs)
    run.font.color.rgb = INK
    run.font.name = CJK
    return box


def finish_slide(slide, note):
    """新页统一的页脚标注。"""
    footer(slide, note)


# ---------------------------------------------------------------- 条码部分
def build_part2_cover(prs):
    """条码部分的封面：这一册自己的标题页。"""
    slide = prs.slides.add_slide(blank_layout(prs))
    band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.34), H)
    band.fill.solid()
    band.fill.fore_color.rgb = ORANGE
    band.line.fill.background()
    band.shadow.inherit = False

    tb(slide, Inches(1.35), Inches(1.45), Inches(10), Inches(0.3),
       "2026-11-05", size=12, color=GRAY)
    tb(slide, Inches(1.35), Inches(1.90), Inches(10.5), Inches(0.85),
       "条码码制分类实验", size=40, bold=True)
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.35),
                                  Inches(2.90), Inches(1.6), Pt(4))
    line.fill.solid()
    line.fill.fore_color.rgb = BLUE
    line.line.fill.background()
    line.shadow.inherit = False

    tb(slide, Inches(1.35), Inches(3.18), Inches(10.6), Inches(0.8),
       "数据集 6527 张（C128 : EAN13 ≈ 1 : 3.3）\n"
       "MobileNetV3-Large 从零训练 · CPU · 3 seed 同 seed 配对检验",
       size=15, color=GRAY, line_spacing=1.55)

    tb(slide, Inches(1.35), Inches(4.42), Inches(10.6), Inches(0.3),
       "内容", size=13, bold=True, color=BLUE)
    left = ["01  实验设定：数据、配方、判据",
            "02  唯一有效的改动 128 → 224",
            "03  五个负结果：都不显著",
            "04  划分反转复核：方向能复现吗",
            "05  镜像复核：三个干预重跑",
            "06  一维剖面：设计与预注册"]
    right = ["07  结果与两轮诊断",
             "08  三个把变量控制住的做法",
             "09  五次让结论作废的错误",
             "10  操作层面的错误与规则",
             "11  局限与下一步"]
    for column, items in ((Inches(1.35), left), (Inches(7.0), right)):
        y = Inches(4.78)
        for item in items:
            tb(slide, column, y, Inches(4.9), Inches(0.3), item, size=12.5,
               color=INK)
            y += Inches(0.34)

    tb(slide, Inches(1.35), H - Inches(0.62), Inches(10.6), Inches(0.26),
       "数据来源：docs/条码实验记录.md · docs/一维剖面实验.md · "
       ".tmp_tests/ 内实测产物", size=9, color=GRAY)
    return slide


# ---------------------------------------------------------------- 条码各页
BC_NOTE = "数据来源：docs/条码实验记录.md · docs/一维剖面实验.md · .tmp_tests/ 内实测产物"


def build_bc_setup(prs):
    """B1：数据集与协议。"""
    slide = prs.slides.add_slide(blank_layout(prs))
    header(slide, "条码 · 01", "实验设定：先钉死协议，再谈涨点",
           "模型、配方、判据都固定下来，后面每一页只改一个变量")

    tb(slide, MARGIN, Inches(1.72), Inches(5.9), Inches(0.3),
       "数据与任务", size=13, bold=True, color=BLUE)
    table(slide, [
        ["项", "值"],
        ["任务", "条码码制分类（C128 / EAN13）"],
        ["样本", "6527 张（train 4569 / val 979 / test 979）"],
        ["类别比", "≈ 1 : 3.3（少数类是难点）"],
        ["输入", "条码区域裁剪灰度图，宽度 31~2024 px"],
        ["评价", "Top-1、balanced、C128 召回、错误数"],
    ], MARGIN, Inches(2.08), [Inches(1.05), Inches(4.85)], fs=11)

    tb(slide, Inches(7.0), Inches(1.72), Inches(5.7), Inches(0.3),
       "训练配方（全程不动的部分）", size=13, bold=True, color=BLUE)
    table(slide, [
        ["项", "值"],
        ["模型", "MobileNetV3-Large，从零训练"],
        ["参数 / MACs", "4,204,306 / 229.9 M @224"],
        ["损失 / 优化", "CE / Adam 1e-3，batch 32"],
        ["轮数 / 选轮", "10 轮 / 验证损失最小"],
        ["增强 / 设备", "无 / CPU"],
        ["判据", "n=3 配对 t 检验，阈值 4.303"],
    ], Inches(7.0), Inches(2.08), [Inches(1.25), Inches(4.45)], fs=11)

    callout(slide, MARGIN, Inches(5.62), CONTENT_W, Inches(0.95),
            "两条纪律：① 一切结论用 3 个 seed 的同 seed 配对差，不看单次；"
            "② 先算最小可检测效应（MDE），低于它的差异一律写「不显著」。",
            fs=12.5, fill=RGBColor(0xEA, 0xF1, 0xF8), edge=BLUE)
    finish_slide(slide, BC_NOTE)
    return slide


def build_bc_win(prs):
    """B2：唯一有效的改动 128 → 224。"""
    slide = prs.slides.add_slide(blank_layout(prs))
    header(slide, "条码 · 02", "唯一站得住的改动：输入 128 → 224",
           "三个 seed 配对，+0.51 ± 0.10 pt，t = 8.83（显著）")

    col_w = [Inches(1.6), Inches(2.35), Inches(1.75),
             Inches(1.5), Inches(1.35)]
    table(slide, [
        ["输入", "Top-1（3 seed）", "C128 召回", "错误数", "MACs"],
        ["128 × 128", "98.88 ± 0.11", "96.31", "11.0", "≈75 M（折算）"],
        ["224 × 224", "99.39 ± 0.21", "98.23", "6.0", "229.9 M"],
        ["配对差", "+0.51 ± 0.10", "+1.92", "−5.0", "×3.06"],
    ], MARGIN, Inches(1.78), col_w, fs=11.5,
        align_right_cols=(1, 2, 3, 4))

    tb(slide, MARGIN, Inches(3.62), Inches(6.1), Inches(0.3),
       "为什么这条站得住", size=13, bold=True, color=BLUE)
    bullets(slide, MARGIN, Inches(3.98), Inches(6.0), [
        ("效应量大于噪声底：", "0.51 pt 对应约 5 个样本，t=8.83 远超临界值"),
        ("错误数 11 → 6：", "近腰斩，说明不是指标抖动而是真的判别边界移动"),
        ("代价明确：", "MACs 增加 3 倍，CPU 每轮 30 s → 2 min"),
    ], size=12)

    tb(slide, Inches(7.1), Inches(3.62), Inches(5.6), Inches(0.3),
       "为什么不继续加分辨率", size=13, bold=True, color=BLUE)
    bullets(slide, Inches(7.1), Inches(3.98), Inches(5.6), [
        ("S16 已验证：", "把末端特征从 7×7 提到 14×14，99.49%，t=0.63 不显著"),
        ("结论：", "瓶颈已从「网络分辨率」转移到「输入信息量」"),
    ], size=12, dot=ORANGE)

    callout(slide, MARGIN, Inches(5.62), CONTENT_W, Inches(0.95),
            "这一页是整条线里唯一被证实有效的改动，也是后面所有对比的基准："
            "224 基线 = 99.39 ± 0.21%（重新划分后为 97.00 ± 0.68%）。", fs=12.5)
    finish_slide(slide, BC_NOTE)
    return slide


def build_bc_negative(prs):
    """B3：负结果清单。"""
    slide = prs.slides.add_slide(blank_layout(prs))
    header(slide, "条码 · 03", "五个负结果：都试过，都不显著",
           "把它们列齐比只报成功更重要 —— 否则下一个人会重跑一遍")

    col_w = [Inches(2.05), Inches(2.55), Inches(2.3),
             Inches(1.55), Inches(3.1)]
    table(slide, [
        ["尝试", "做法", "结果", "配对检验", "判定"],
        ["Focal Loss", "γ=2，替换 CE", "99.08%", "−0.31 pt", "不显著"],
        ["Label Smoothing", "ε=0.1", "99.18 ± 0.37", "t = 1.52", "不显著"],
        ["降主干步长 S16", "末端 7×7 → 14×14", "99.49 ± 0.18", "t = 0.63",
         "不显著（MACs 1.62×）"],
        ["加轮数 10 → 20", "同配方翻倍", "99.39%（同 10 轮）", "—",
         "无效，第 20 轮 val 崩到 94.48%"],
        ["只改长宽比", "128×384 等", "≈ 0 提升", "—", "无效"],
    ], MARGIN, Inches(1.78), col_w, fs=11,
        row_h=Inches(0.52))

    tb(slide, MARGIN, Inches(5.25), Inches(6.1), Inches(0.3),
       "共同的原因", size=13, bold=True, color=BLUE)
    bullets(slide, MARGIN, Inches(5.6), Inches(6.0), [
        ("基线已经接近上限：", "错误只剩 6 个，任何改动的空间都很窄"),
        ("效应量小于 MDE：", "原划分 MDE ≈ 0.5 pt，这些改动只有 0.1~0.3 pt"),
    ], size=12)

    callout(slide, Inches(7.1), Inches(5.25), Inches(5.6), Inches(1.55),
            "教训：当基线错误数已是个位数量级时，"
            "「再调一个损失函数」几乎不可能被检出 —— "
            "要么换更大的训练集，要么换问题（见后文一维剖面）。",
            fs=12, fill=RGBColor(0xEA, 0xF1, 0xF8), edge=BLUE)
    finish_slide(slide, BC_NOTE)
    return slide


def build_bc_split(prs):
    """B4：划分反转复核。"""
    slide = prs.slides.add_slide(blank_layout(prs))
    header(slide, "条码 · 04", "把训练集和测试集对调，结论还成立吗",
           "训练 979 / 测试 4569 —— 用最小训练集换最大检验强度")

    col_w = [Inches(1.45), Inches(1.35), Inches(1.35),
             Inches(1.95), Inches(1.95), Inches(3.45)]
    table(slide, [
        ["指标", "128（原）", "224（原）", "128（swap）", "224（swap）", "swap 配对差"],
        ["Top-1", "98.88", "99.39", "96.21 ± 0.78", "97.00 ± 0.68",
         "+0.79 ± 1.03（t=1.32）"],
        ["C128 召回", "96.31", "98.23", "88.29", "93.89",
         "+5.60 ± 1.64（t=5.90 显著）"],
        ["错误数", "11.0", "6.0", "173.3", "137.3", "−36.0 ± 47.2"],
    ], MARGIN, Inches(1.80), col_w, fs=11,
        align_right_cols=(1, 2, 3, 4), row_h=Inches(0.46))

    tb(slide, MARGIN, Inches(3.95), Inches(6.0), Inches(0.3),
       "读数", size=13, bold=True, color=BLUE)
    bullets(slide, MARGIN, Inches(4.30), Inches(6.0), [
        ("方向全部复现：", "四个指标、六个 run，224 全面优于 128，无一反向"),
        ("少数类显著放大：", "C128 +5.60 pt，训练集缩到 1/4.7 后少数类只剩 226 张"),
        ("Top-1 检不出：", "MDE = 2.57 pt，而真实效应只有 0.8 pt"),
    ], size=12)

    tb(slide, Inches(7.1), Inches(3.95), Inches(5.6), Inches(0.3),
       "这一页真正的价值", size=13, bold=True, color=BLUE)
    callout(slide, Inches(7.1), Inches(4.30), Inches(5.6), Inches(1.55),
            "小训练集让功效崩掉：seed 间方差从 ±0.10 涨到 ±1.03。"
            "所以「不显著」只能读作「没检出」，不能读作「证明无效」。"
            "—— 这句话在下一部分的 1D 实验里会反噬我们自己。",
            fs=12, fill=RGBColor(0xFD, 0xF3, 0xE6), edge=ORANGE)
    finish_slide(slide, BC_NOTE)
    return slide


def build_bc_mirror(prs):
    """B5：三个负结果的镜像复核。"""
    slide = prs.slides.add_slide(blank_layout(prs))
    header(slide, "条码 · 05", "镜像复核：负结果复现了，但方向不稳",
           "把三个「无效」干预在 swap 划分上按同样协议重跑")

    col_w = [Inches(2.35), Inches(0.9), Inches(1.5),
             Inches(1.9), Inches(1.9), Inches(1.2),
             Inches(1.5)]
    table(slide, [
        ["干预", "seed", "Top-1", "配对 dTop-1", "配对 dC128", "t", "判定"],
        ["Focal vs 224", "2", "95.84", "−0.77", "+1.19", "−0.45", "不显著"],
        ["Label Smoothing", "3", "97.19", "+0.20", "−0.82", "0.32", "不显著"],
        ["S16", "3", "96.09", "−0.90", "−4.30", "−0.85", "不显著"],
        ["128×384 几何", "3", "95.56", "−0.64", "−0.35", "−2.75", "不显著"],
    ], MARGIN, Inches(1.80), col_w, fs=11.5,
        align_right_cols=(1, 2, 3, 4, 5), row_h=Inches(0.46))

    tb(slide, MARGIN, Inches(4.0), Inches(6.0), Inches(0.3),
       "两个发现", size=13, bold=True, color=BLUE)
    bullets(slide, MARGIN, Inches(4.35), Inches(6.0), [
        ("「都不显著」复现了：", "与原划分一致，没有任何一项值得采纳"),
        ("但点估计方向不稳：", "S16 从 +0.10 变 −0.90、LS 从 −0.21 变 +0.20，符号都翻了"),
    ], size=12)

    tb(slide, Inches(7.1), Inches(4.0), Inches(5.6), Inches(0.3),
       "所以 S16 的结论要改写", size=13, bold=True, color=BLUE)
    callout(slide, Inches(7.1), Inches(4.35), Inches(5.6), Inches(1.55),
            "从「无影响」改成「不确定有收益，且可能在少数类上有代价」"
            "（C128 召回 −4.30 pt，已接近 MDE 4.08）。"
            "效应量在噪声量级时，方向由划分决定 —— 不能拿某一次是正的当采纳理由。",
            fs=12, fill=RGBColor(0xEA, 0xF1, 0xF8), edge=BLUE)
    finish_slide(slide, BC_NOTE)
    return slide


def build_bc_1d_design(prs):
    """B6：一维剖面的动机与设计。"""
    slide = prs.slides.add_slide(blank_layout(prs))
    header(slide, "条码 · 06", "换问题：2D 布局到底贡献了多少精度",
           "把条码图压成一维剖面 (2, L)，只保留「横向条宽」这一种信息")

    tb(slide, MARGIN, Inches(1.72), Inches(5.9), Inches(0.3),
       "前端口径：三条实测决定的规则", size=13, bold=True, color=BLUE)
    bullets(slide, MARGIN, Inches(2.08), Inches(5.9), [
        ("98% 的图是竖直条：", "于是沿行平均取列投影（反向投影锐度只有 1/10.6）"),
        ("无倾斜：", "旋转增益中位 1.00，不做 deskew，少一个不稳定环节"),
        ("两路通道：", "平滑剖面 + 高通（增益 1.4 标定得来，实测截断 1.10%）"),
        ("长度归一化：", "固定 L=1024 / 128，不用逐样本标准化（会抹掉对比度）"),
    ], size=12)

    tb(slide, Inches(7.1), Inches(1.72), Inches(5.6), Inches(0.3),
       "主干：转换而不是另写", size=13, bold=True, color=BLUE)
    bullets(slide, Inches(7.1), Inches(2.08), Inches(5.6), [
        ("目的：", "唯一变量只能是输入张量，骨架必须同一份"),
        ("做法：", "把 2D MobileNetV3 逐层替换成 1D（Conv / BN / Pool / SE）"),
        ("代价：", "参数 4.204M → 4.134M，MACs 229.9M → 69.0M（0.30×）"),
    ], size=12, dot=ORANGE)

    col_w = [Inches(1.75), Inches(2.05), Inches(2.75),
             Inches(1.15)]
    table(slide, [
        ["维度", "2D 224×224", "1D 剖面 L=1024", "比值"],
        ["参数", "4,204,306", "4,133,506", "0.98×"],
        ["MACs", "229.9 M", "69.0 M", "0.30×"],
        ["分类头前特征", "7 × 7", "长度 32", "—"],
        ["预注册预测", "—", "≥ 96.21%（打赢 128×384）", "—"],
    ], MARGIN, Inches(4.55), col_w, fs=11.5,
        align_right_cols=(1, 2, 3), row_h=Inches(0.40))

    callout(slide, Inches(8.65), Inches(4.55), Inches(4.05), Inches(1.85),
            "预注册写在跑之前：\n"
            "① 1D ≥ 96.21%\n② 1D ≈ 97.00%\n③ C128 召回 ≥ 93.89%\n\n"
            "——三条全部被证伪。", fs=12)
    finish_slide(slide, BC_NOTE)
    return slide


def build_bc_1d_result(prs):
    """B7：1D 结果与两轮诊断。"""
    slide = prs.slides.add_slide(blank_layout(prs))
    header(slide, "条码 · 07", "1D 只到 93.06%，但差距不在信息上",
           "两轮诊断：固定分类器看表示，固定输入看训练")

    chart_data = CategoryChartData()
    chart_data.categories = ["1D L=1024", "1D L=128", "1D L=128+wd",
                             "2D 128×384", "2D 224"]
    chart_data.add_series("Top-1", (93.06, 94.14, 95.34, 96.21, 97.00))
    chart_data.add_series("C128 召回", (82.30, 85.19, 85.88, 88.29, 93.89))
    gf = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, MARGIN,
                                Inches(1.72), Inches(6.35), Inches(3.55),
                                chart_data)
    chart = gf.chart
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.BOTTOM
    chart.legend.include_in_layout = False
    chart.font.size = Pt(10)
    chart.font.name = CJK
    chart.value_axis.minimum_scale = 75
    chart.value_axis.maximum_scale = 100
    chart.value_axis.has_major_gridlines = False
    chart.has_title = True
    chart.chart_title.text_frame.text = "各配置测试精度（%，3 seed 均值）"
    chart.chart_title.text_frame.paragraphs[0].font.size = Pt(11)

    tb(slide, Inches(7.3), Inches(1.72), Inches(5.4), Inches(0.3),
       "诊断一：固定分类器，只换表示", size=12.5, bold=True, color=BLUE)
    bullets(slide, Inches(7.3), Inches(2.06), Inches(5.4), [
        ("同一岭回归：", "2D 像素 96.67% vs 1D 剖面 95.84% → 表示只差 0.83 pt"),
        ("少数类重灾：", "C128 召回 94.78 vs 88.41，差 5~6 pt"),
        ("关键反例：", "同特征上岭回归 96.04% > 1D CNN 94.14%"),
    ], size=11.5, gap=Inches(0.05))

    tb(slide, Inches(7.3), Inches(3.85), Inches(5.4), Inches(0.3),
       "诊断二：过拟，不是欠拟合", size=12.5, bold=True, color=BLUE)
    bullets(slide, Inches(7.3), Inches(4.19), Inches(5.4), [
        ("证据：", "train loss 0.40 → 0.048，val 卡 92%"),
        ("降分辨率：", "L=1024 → 128，+1.1 pt"),
        ("加正则：", "再 +1.2 pt，到 95.34 ± 0.46"),
        ("最终差：", "vs 128×384 −0.87（全不显著）；vs 224 C128 召回 −8.01（显著）"),
    ], size=11.5, gap=Inches(0.05), dot=ORANGE)
    finish_slide(slide, BC_NOTE)
    return slide


def build_bc_lessons(prs):
    """B8：方法学教训。"""
    slide = prs.slides.add_slide(blank_layout(prs))
    header(slide, "条码 · 08", "三个把变量控制住的做法",
           "三条都是本轮实验里实际采用的做法")

    items = [
        ("比较「表示」→ 固定分类器",
         "用同一个线性模型分别喂 2D 像素与 1D 剖面，才敢说「表示的差距只有 0.8 pt」。"
         "若直接拿两个 CNN 比，0.8 pt 的信息差会被 3 pt 的训练差淹没。"),
        ("比较「架构」→ 固定输入",
         "1D 主干由 2D 逐层替换而来，参数量 4.204M → 4.134M，"
         "这样「1D 比 2D 差」才只剩输入这一个解释变量。"),
        ("先量噪声底，再谈涨点",
         "n=3 配对检验 + MDE：原划分 0.5 pt、swap 划分 2.57 pt。"
         "低于 MDE 的差异只能写「没检出」，不能写「有效」或「无效」。"),
    ]
    y = Inches(1.80)
    for index, (head, body) in enumerate(items, 1):
        num = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, MARGIN, y,
                                     Inches(0.52), Inches(0.52))
        num.fill.solid()
        num.fill.fore_color.rgb = BLUE
        num.line.fill.background()
        num.shadow.inherit = False
        frame = num.text_frame
        frame.margin_left = frame.margin_right = 0
        frame.paragraphs[0].alignment = PP_ALIGN.CENTER
        run = frame.paragraphs[0].add_run()
        run.text = str(index)
        run.font.size = Pt(15)
        run.font.bold = True
        run.font.color.rgb = WHITE
        run.font.name = MONO
        tb(slide, MARGIN + Inches(0.75), y, Inches(5.1), Inches(0.3), head,
           size=14, bold=True, color=INK)
        tb(slide, MARGIN + Inches(0.75), y + Inches(0.34), Inches(11.1),
           Inches(0.7), body, size=12, color=GRAY, line_spacing=1.35)
        y += Inches(1.30)

    callout(slide, MARGIN, Inches(5.85), CONTENT_W, Inches(0.85),
            "还有一个必须警惕的巧合：2D 像素缩到 16×32 的线性模型就有 96.76%，"
            "与训练 310 步的 MobileNetV3（97.00%）几乎持平 —— 深度模型的价值"
            "必须靠更强的基线来证明。", fs=12,
            fill=RGBColor(0xEA, 0xF1, 0xF8), edge=BLUE)
    finish_slide(slide, BC_NOTE)
    return slide


def build_part2_next(prs):
    """条码部分的收尾页：只讲这部分的局限与下一步。"""
    slide = prs.slides.add_slide(blank_layout(prs))
    header(slide, "条码 · 11", "局限与下一步", "都只针对本部分的实验")

    tb(slide, MARGIN, Inches(1.78), Inches(6.0), Inches(0.3),
       "局限", size=13, bold=True, color=BLUE)
    bullets(slide, MARGIN, Inches(2.14), Inches(5.9), [
        ("训练预算：", "10 轮；swap 划分下训练样本只有 979 张"),
        ("选轮依据：", "验证集 979 张，按验证损失选最佳轮，自身有噪声"),
        ("前端假设：", "条码竖直、无倾斜（200 张探针支持，不等于全部样本）"),
        ("缺对照：", "未做 2D 长训练（30~60 轮）对照，所以 2D 上限没有直接证据"),
    ], size=12)

    tb(slide, Inches(7.1), Inches(1.78), Inches(5.6), Inches(0.3),
       "下一步", size=13, bold=True, color=ORANGE)
    bullets(slide, Inches(7.1), Inches(2.14), Inches(5.6), [
        ("原始划分重跑 1D：", "用 4569 张训练，检验「差距来自 979 样本过拟」这个归因"),
        ("1D 侧强增广：", "条宽抖动、保比例的随机裁剪、弹性形变"),
        ("预训练权重灰度适配：", "从零训练只用 4569 张，这是最可能有增益的方向"),
        ("多尺度增强：", "尺寸跨度 31~2024 px，固定缩放会同时牺牲小图与大图"),
    ], size=12, dot=ORANGE)

    callout(slide, MARGIN, Inches(5.55), CONTENT_W, Inches(0.85),
            "本部分所有数字都能在 docs/ 与 .tmp_tests/ 里查到出处；"
            "复现命令见 docs/一维剖面实验.md §10。", fs=12,
            fill=RGBColor(0xEA, 0xF1, 0xF8), edge=BLUE)
    finish_slide(slide, BC_NOTE)
    return slide


# ---------------------------------------------------------------- 分割头过程页
def build_p1_process_a(prs):
    """分割头 过程①：每题一行 —— 原题、错在哪、要点。"""
    slide = prs.slides.add_slide(blank_layout(prs))
    header(slide, "过程 ①", "分割头自检：题目 · 错在哪 · 要点",
           "第一轮 2.0 / 5 · 26-seg 预习 4 / 10 · 第二轮 6 / 12；"
           "答不出就写「不会」，编答案比留空更糟")

    col_w = [Inches(1.10), Inches(3.60), Inches(2.95), Inches(4.05)]
    table(slide, [
        ["题 · 得分", "题目", "错在哪", "要点"],
        ["Q1 · 0/5",
         "写出 mask = σ(?) 的公式，c 与 P 的形状、参数量",
         "空着没答",
         "mask = σ(Σ c_i·P_i)；c (N,32) 来自 cv4，"
         "P (32,160,160) 来自 Proto；一个实例一次 32 维点积"],
        ["Q2 · 2.5/5",
         "proto 为什么是 160×160，GT 掩码怎么和它对上",
         "说成「proto 在预测框内匹配 GT」；interpolate 方向也说反",
         "proto 是全图的，裁框发生在损失/后处理；mask_ratio=4 让 GT "
         "天然是 160，兜底是把 proto 上采样到 GT"],
        ["Q5 · 0/5",
         "32 个 mask 系数在 NMS 前后怎么用，会被丢掉吗",
         "空着没答",
         "NMS 把 mask 一起 cat 进输出，仍是 (N,38)；列序 [4·框|conf|cls|"
         "32 系数]，后处理取 [:,6:]，且必须先 NMS 再合成"],
        ["预习① · 2.5/5",
         "Proto26 上采样为什么用 nearest、为什么用 scale_factor",
         "说成「实现硬边界」",
         "硬边界是 >0 二值化的事；真因是常数 scale_factor 让 resize "
         "成为形状无关的静态算子，动态 shape 导出才不写死输入尺寸"],
        ["预习② · 1.5/5",
         "sem_masks 从哪来、是什么类型的一张图",
         "说成「与分类得分相乘」",
         "乘的是类别 id：值域 0..nc-1，0 同时表示背景/未覆盖；"
         "所以不需要额外语义标注"],
        ["二轮① · 1.5/2",
         "12 个实例合成 mask 要做多少次 32 维点积",
         "只给了 MACs 与形状，没答次数",
         "12 × 160 × 160 = 307,200 次；proto 全图只生成 1 次、"
         "被每个实例各用一次"],
        ["二轮④ · 0.5/2",
         "sem_masks 里 0 与 7 各是什么；重叠像素取谁",
         "说重叠取「分类分数高」的 —— GT 里没有分数",
         "0 = 未覆盖/背景，7 = 第 7 类；overlap_mask 时面积小者胜，"
         "与实例掩码的取法同源（utils.py 按面积降序写入）"],
    ], MARGIN, Inches(1.80), col_w, fs=10, row_h=Inches(0.48))

    callout(slide, MARGIN, Inches(5.72), CONTENT_W, Inches(0.66),
            "错的共同点：把「在哪发生」记混了 —— 裁框是损失侧的事、"
            "二值化是后处理的事、静态化是导出的事。", fs=12)
    finish_slide(slide, "ultralytics 8.4.184 · torch 2.4.1+cpu · "
                        "出处：YOLO11→YOLO26-代码精读笔记.md §11")
    return slide


def build_p1_process_b(prs):
    """分割头 过程②：错误能被找回来的方法，以及一条 proto 对齐链。"""
    slide = prs.slides.add_slide(blank_layout(prs))
    header(slide, "过程 ②", "把错找回来的三步",
           "自测 → 落到行号 → 实测对账")

    tb(slide, MARGIN, Inches(1.74), Inches(6.0), Inches(0.3),
       "做法", size=13, bold=True, color=BLUE)
    bullets(slide, MARGIN, Inches(2.10), Inches(5.9), [
        ("先自测：", "答不出就写「不会」—— 编答案比留空更糟，"
         "错的答案会变成错误的记忆"),
        ("每条更正落到行号：", "head.py:341-346、loss.py:546、nms.py:119 —— "
         "落不到行号的就不算更正"),
        ("实测对账：", "参数量 160,224（cv4）与 92,544（proto）、形状 "
         "(1,32,160,160)、ONNX 节点 476 → 560，都要与源码推算一致"),
    ], size=12)

    tb(slide, Inches(7.1), Inches(1.74), Inches(5.6), Inches(0.3),
       "一条完整的链", size=13, bold=True, color=ORANGE)
    callout(slide, Inches(7.1), Inches(2.10), Inches(5.6), Inches(2.15),
            "为什么 proto 是 160×160：\n"
            "GT 掩码在 Format.apply_instances 里按 mask_ratio = 4 下采样到 "
            "160×160（augment.py:2459）；\n"
            "proto 因为 ConvTranspose2d 2× 也正好是 160 —— 两者默认就一致；\n"
            "F.interpolate 只是兜底，且方向是把 proto 上采样到 GT 尺寸。",
            fs=11.5)

    tb(slide, MARGIN, Inches(4.55), Inches(6.0), Inches(0.3),
       "所以「更正」不止是改结论", size=13, bold=True, color=BLUE)
    bullets(slide, MARGIN, Inches(4.91), Inches(5.9), [
        ("改的是判据：", "从「看起来合理」改成「能对上行号与实测量」"),
        ("改的是顺序：", "先 NMS 再合成 mask；8400 行全合成会慢 20 倍以上"),
        ("留下的是方法：", "下一轮换模型或换任务，这套对账流程可以照搬"),
    ], size=12)
    finish_slide(slide, "ultralytics 8.4.184 · torch 2.4.1+cpu · "
                        "出处：YOLO11→YOLO26-代码精读笔记.md §11")
    return slide


def build_bc_process_a(prs):
    """条码 09：让结论作废的四次错误（正确性 bug + 统计纪律）。"""
    slide = prs.slides.add_slide(blank_layout(prs))
    header(slide, "条码 · 09", "五次让结论作废的错误",
           "共同点：不是跑不起来，而是跑起来了但数字是假的")

    col_w = [Inches(2.25), Inches(3.05), Inches(4.05), Inches(2.7)]
    table(slide, [
        ["错误", "症状", "根因", "修正"],
        ["BN 动量沿用 TF 配方", "验证/测试恒为 77.02%，正好等于全猜多数类",
         "momentum=0.01，143 步/轮残留约 24% 初始化值，46 层串联放大",
         "统一 0.1（5dfb1fe），历史数字作废"],
        ["DML 检查点不可加载", "10 轮跑完，加载最佳权重时 UnpicklingError，结果全丢",
         "DirectML 张量序列化，weights_only=True 不认", "保存前搬 CPU（13acbd7）"],
        ["错误架构能静默加载", "S16 与原版参数形状相同，load_state_dict 不报错",
         "审计脚本硬编码架构名", "架构名从实验日志读"],
        ["单 seed 当结论", "LS 的 seed0 = 99.59%，补 2 个 seed 后 99.18±0.37，t=1.52",
         "只看一个 seed", "一律 3 seed 同 seed 配对"],
        ["用「都错的样本」估上限", "S16 一上来就修好 3 个「不可约」样本中的 2 个",
         "错误集合随架构与 seed 漂移", "不做上限估计"],
    ], MARGIN, Inches(1.74), col_w, fs=10.5, row_h=Inches(0.62))

    callout(slide, MARGIN, Inches(5.44), CONTENT_W, Inches(0.9),
            "A1 最典型：损失正常下降、训练准确率正常、验证准确率一字不动。"
            "所以训练框架里加了「连续两轮验证准确率完全相同就告警」的钩子。",
            fs=12.5)
    finish_slide(slide, BC_NOTE)
    return slide


def build_bc_process_b(prs):
    """条码 10：操作层面的错误与后来定下的规则。"""
    slide = prs.slides.add_slide(blank_layout(prs))
    header(slide, "条码 · 10", "操作层面的错误与现在的规则",
           "前三条来自 docs/排查问题清单.md 的 E 段，后两条是这一轮新增的")

    col_w = [Inches(4.0), Inches(4.3), Inches(3.7)]
    table(slide, [
        ["错误", "后果", "现在的规则"],
        ["改文件时把函数体/文档字符串删掉",
         "先后弄坏 _first_conv_path、main()、一处 save_model 行",
         "改完立刻读回或跑测试"],
        ["heredoc 里塞复杂 Python",
         "引号被 PowerShell 吃掉，报错还指向错误的行号", "写文件，不用 heredoc"],
        ["自检表整列相同没起疑",
         "stride 列全是 1，累计倍数算成 2 倍", "整列相同先怀疑采集方式"],
        ["模块名与标准库重名",
         "src/profile.py 遮蔽标准库 profile，torch 导入崩溃，4 个 run 白跑",
         "新模块先查是否重名"],
        ["checkpoint 只写不删",
         "单个 run 堆 10 个权重，测试目录一度 4.3 GB", "存新最佳时删旧文件"],
    ], MARGIN, Inches(1.74), col_w, fs=10.5, row_h=Inches(0.60))

    callout(slide, MARGIN, Inches(5.30), CONTENT_W, Inches(0.9),
            "记下来的目的只有一个：同样的错误不在第二轮重复。"
            "这份清单现在有 35 条，按「会让结论作废的 bug / 数据审查 / "
            "工具链陷阱 / 统计纪律 / 我自己的操作错误」五类归档。", fs=12.5)
    finish_slide(slide, BC_NOTE)
    return slide


# ---------------------------------------------------------------- 重排与页码
def refresh_page_numbers(prs):
    """重排后页码会错位，按当前顺序重刷右下角的页码文本框。

    识别规则：文本框内容全是数字，且位置贴着页面右下角。
    """
    for page, slide in enumerate(prs.slides, start=1):
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            text = shape.text_frame.text.strip()
            if not text.isdigit():
                continue
            if shape.top is None or shape.left is None:
                continue
            near_bottom = abs(int(shape.top) - int(H - Inches(0.62))) < Emu(
                Inches(0.25))
            near_right = int(shape.left) > int(W - Inches(2.2))
            if near_bottom and near_right:
                shape.text_frame.paragraphs[0].runs[0].text = str(page)


def drop_slides(prs, indexes):
    """按 0 基下标删除幻灯片，并把它的 part 关系一并摘掉。

    只从 ``sldIdLst`` 移除是不够的 —— slide part 会留在包里成为孤儿，
    表现为文件体积不减。这里先 drop_rel 再移除元素。
    """
    sld_ids = prs.slides._sldIdLst
    rel_ns = ("http://schemas.openxmlformats.org/officeDocument/2006/"
              "relationships")
    for index in sorted(indexes, reverse=True):
        element = sld_ids[index]
        rel_id = element.get("{%s}id" % rel_ns)
        prs.part.drop_rel(rel_id)
        sld_ids.remove(element)


def prune_base_deck(prs):
    """把底座 deck 裁成「分割头学习对比」这一段。

    底座原顺序（14 页）：
      0 封面 | 1-3 结构①②③ | 4-7 差异①–④ | 8 实测 | 9 结论 | 10-13 附录
    保留 0 与 4-9，删掉 1-3 与 10-13。
    """
    drop_slides(prs, [10, 11, 12, 13])   # 先删靠后的附录，下标不失效
    drop_slides(prs, [1, 2, 3])          # 再删前置的结构页


def fix_cover_reference(prs):
    """封面原本写「脚本见附录」，附录已删，把这个指向去掉。"""
    for shape in prs.slides[0].shapes:
        if not shape.has_text_frame:
            continue
        for paragraph in shape.text_frame.paragraphs:
            for run in paragraph.runs:
                if "脚本见附录" in run.text:
                    run.text = run.text.replace("，脚本见附录", "")


def renumber_slide_parts(prs):
    """把 slide 部件名重排成 slide1..slideN。

    python-pptx 新建页时用 ``slide(len(sldIdLst)+1).xml`` 命名，
    所以只要先删过页，新页的名字就会和残存的 slide8/9/10 撞车，
    保存出来的 zip 里会出现重名条目。这里按当前页序统一重排。
    """
    r_ids = [sld_id.rId for sld_id in prs.slides._sldIdLst]
    prs.part.rename_slide_parts(r_ids)


def main():
    if not BASE_PPTX.is_file():
        print("找不到底座 deck: %s" % BASE_PPTX)
        return 1
    prs = Presentation(str(BASE_PPTX))
    print("  底座 deck: %d 页" % len(prs.slides))
    prune_base_deck(prs)
    fix_cover_reference(prs)
    print("  裁剪后（分割头学习对比）: %d 页" % len(prs.slides))

    # 分割头的两页「过程」直接接在结论之后
    build_p1_process_a(prs)
    build_p1_process_b(prs)

    # 第二册（条码）接着排
    build_part2_cover(prs)
    build_bc_setup(prs)
    build_bc_win(prs)
    build_bc_negative(prs)
    build_bc_split(prs)
    build_bc_mirror(prs)
    build_bc_1d_design(prs)
    build_bc_1d_result(prs)
    build_bc_lessons(prs)
    build_bc_process_a(prs)
    build_bc_process_b(prs)
    build_part2_next(prs)

    total = len(prs.slides)
    renumber_slide_parts(prs)   # 先重排部件名，再刷页码
    refresh_page_numbers(prs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT_PPTX))
    print("  总计 %d 页 -> %s" % (total, OUT_PPTX))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
