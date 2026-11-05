"""构建第三周汇报 deck：YOLO26 分割头学习对比 + 条码码制分类实验。

设计要点
--------
1. **以已有的分割头 deck 为底座**（13.33x7.5 英寸、Microsoft YaHei、
   蓝/橙双色），不重建那 14 页 —— 它们包含手工绘制的结构图与实测配图，
   重建等于丢内容。做法是 ``Presentation(源文件)`` 打开后追加新页，
   最后重排 ``sldIdLst`` 得到目标顺序。
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


def stat_blocks(slide, y, stats, size_big=22, x0=None):
    """一行数值卡片：(大数字, 小标签) 列表。

    x0 可指定起始左边界，卡片宽度按剩余可用宽度均分，
    这样封面（左边距比正文更大）与正文页都能用同一个函数。
    """
    n = len(stats)
    start = MARGIN if x0 is None else x0
    gap = Inches(0.30)
    usable = W - start - MARGIN
    width = int((usable - gap * (n - 1)) / n)
    x = start
    for big, label in stats:
        tb(slide, x, y, width, Inches(0.42), big, size=size_big, bold=True,
           color=BLUE)
        tb(slide, x, y + Inches(0.44), width, Inches(0.26), label, size=11,
           color=GRAY)
        x += width + gap


def finish_slide(slide, note):
    """新页统一的页脚标注。"""
    footer(slide, note)


# ---------------------------------------------------------------- 新页：总纲
def build_cover(prs):
    """封面：标题 + 并列陈述两件事 + 三个关键数字。"""
    slide = prs.slides.add_slide(blank_layout(prs))
    band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.34), H)
    band.fill.solid()
    band.fill.fore_color.rgb = BLUE
    band.line.fill.background()
    band.shadow.inherit = False

    tb(slide, Inches(1.35), Inches(1.50), Inches(10), Inches(0.3),
       "2026-11-05", size=12, color=GRAY)
    tb(slide, Inches(1.35), Inches(1.95), Inches(10.5), Inches(0.85),
       "第三周汇报", size=40, bold=True)
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.35),
                                  Inches(2.95), Inches(1.6), Pt(4))
    line.fill.solid()
    line.fill.fore_color.rgb = ORANGE
    line.line.fill.background()
    line.shadow.inherit = False
    # 两件事分别陈述，不做归纳：各自写清做了什么
    tb(slide, Inches(1.35), Inches(3.25), Inches(10.6), Inches(1.0),
       "一、YOLO11 → YOLO26 实例分割头：结构差异与 CPU 实测\n"
       "二、条码码制分类：基线、负结果、划分反转与一维剖面",
       size=17, color=INK, line_spacing=1.6)

    stat_blocks(slide, Inches(5.75), [
        ("26 页", "分割头 10 · 条码 12 · 附录 4"),
        ("99.39%", "条码 2D 基线 Top-1"),
        ("95.34%", "一维剖面最佳 Top-1"),
    ], size_big=20, x0=Inches(1.35))
    tb(slide, Inches(1.35), H - Inches(0.62), Inches(9), Inches(0.26),
       "ultralytics 8.4.184 · torch 2.4.1+cpu · 全部数字来自本机实测与仓库内产物",
       size=9, color=GRAY)
    return slide


def build_agenda(prs):
    """目录：两部分各自的子问题，逐条陈述。"""
    slide = prs.slides.add_slide(blank_layout(prs))
    header(slide, "目录", "两部分的内容")

    tb(slide, MARGIN, Inches(1.75), Inches(6.0), Inches(0.3),
       "第一部分  ·  YOLO26 分割头", size=13, bold=True, color=BLUE)
    bullets(slide, MARGIN, Inches(2.15), Inches(5.7), [
        "三条支路：检测支 / 系数支 / 原型支",
        "一次矩阵乘：系数 × 原型 → mask",
        "11 与 26 的四处差异与不变的接口",
        "实测：参数量、延迟、ONNX 节点",
    ], size=13)

    tb(slide, Inches(7.0), Inches(1.75), Inches(5.7), Inches(0.3),
       "第二部分  ·  条码码制分类", size=13, bold=True, color=BLUE)
    bullets(slide, Inches(7.0), Inches(2.15), Inches(5.7), [
        "基线：128 → 224，唯一有效的改动",
        "负结果：Focal / LS / S16 / 20 轮 / 几何",
        "划分反转与镜像复核：方向能复现吗",
        "一维剖面：差距到底来自信息还是训练",
    ], size=13, dot=ORANGE)

    # 只交代页码与内容范围，不做跨部分的归纳
    callout(slide, MARGIN, Inches(5.2), CONTENT_W, Inches(1.05),
            "第一部分（分割头）P3–P12，资料与附录 P23–P26；"
            "第二部分（条码）P13–P22。两部分内容相互独立，"
            "各自给出问题、方法、数据与结论。", fs=13)
    finish_slide(slide, "第三周汇报 · 目录")
    return slide


def build_part_divider(prs):
    """第二部分的分隔页，插在条码实验各页之前。"""
    slide = prs.slides.add_slide(blank_layout(prs))
    band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.34), H)
    band.fill.solid()
    band.fill.fore_color.rgb = ORANGE
    band.line.fill.background()
    band.shadow.inherit = False

    tb(slide, Inches(1.35), Inches(2.55), Inches(10), Inches(0.34),
       "第二部分", size=13, color=GRAY)
    tb(slide, Inches(1.35), Inches(3.0), Inches(10.5), Inches(1.2),
       "条码码制分类实验", size=36, bold=True)
    tb(slide, Inches(1.35), Inches(4.15), Inches(10.5), Inches(0.9),
       "基线 · 负结果 · 划分反转复核 · 一维剖面\n"
       "数据集 6527 张（C128 : EAN13 ≈ 1 : 3.3）· MobileNetV3-Large 从零训练 · CPU",
       size=14, color=GRAY, line_spacing=1.5)
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
            "要么换更大的训练集，要么换问题（见第二部分）。",
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


def build_summary(prs):
    """收尾页：产出 + 下一步。"""
    slide = prs.slides.add_slide(blank_layout(prs))
    header(slide, "总结", "本周产出与下一步",
           "代码、测试、文档、实验产物都在仓库里，可复现")

    tb(slide, MARGIN, Inches(1.75), Inches(6.0), Inches(0.3),
       "产出", size=13, bold=True, color=BLUE)
    table(slide, [
        ["类别", "内容"],
        ["新增模块", "profile1d / models_1d / datasets.profile / train_profile"],
        ["测试", "3 个文件 31 项，flake8 无告警"],
        ["文档", "一维剖面实验.md + 条码实验记录 §7.1"],
        ["实验", "2D 32 个 run + 1D 33 个 run，全部落盘"],
        ["提交", "本轮 9 个 commit（含 2 处 bug 修复）"],
    ], MARGIN, Inches(2.10), [Inches(1.3), Inches(4.6)], fs=11)

    tb(slide, Inches(7.1), Inches(1.75), Inches(5.6), Inches(0.3),
       "下一步（按价值排序）", size=13, bold=True, color=BLUE)
    bullets(slide, Inches(7.1), Inches(2.10), Inches(5.6), [
        ("原始划分重跑 1D：", "直接用 4569 张训练，检验「差距源于 979 样本过拟」这个归因"),
        ("1D 侧强增广：", "条宽抖动、保比例的随机裁剪、弹性形变"),
        ("条码侧下一个高地：", "预训练权重灰度适配；多尺度增强（尺寸跨度 31~2024 px）"),
    ], size=12, dot=ORANGE)

    stats = [("9", "新增 commit"), ("31", "新增测试项"),
             ("65", "实验 run"), ("0", "flake8 告警")]
    x = MARGIN
    width = Inches(2.0)
    for big, label in stats:
        tb(slide, x, Inches(5.35), width, Inches(0.55), big, size=30,
           bold=True, color=BLUE)
        tb(slide, x, Inches(5.95), width, Inches(0.26), label, size=11,
           color=GRAY)
        x += width

    callout(slide, Inches(9.6), Inches(5.28), Inches(3.1), Inches(1.0),
            "全部结论可复现：\n每页数字都能在 docs/ 或 .tmp_tests/ 里查到出处。",
            fs=11.5, fill=RGBColor(0xF6, 0xFF, 0xF6), edge=GREEN)
    finish_slide(slide, BC_NOTE)
    return slide


# ---------------------------------------------------------------- 重排与页码
def reorder_slides(prs, order):
    """按 order（0 基下标列表）重排幻灯片顺序。

    只调整 ``sldIdLst`` 的元素次序，不复制/删除任何 part，
    因此图片、图表、备注都原样保留。
    """
    sld_ids = prs.slides._sldIdLst
    elements = list(sld_ids)
    if len(order) != len(elements):
        raise ValueError("重排下标的数量 (%d) 与页数 (%d) 不一致"
                         % (len(order), len(elements)))
    for element in elements:
        sld_ids.remove(element)
    for index in order:
        sld_ids.append(elements[index])


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


def add_part_tag(prs, existing_count, tag):
    """在底座 deck 的封面（第 1 页）右上角加一个分部标签。"""
    slide = prs.slides[0]
    tb(slide, W - MARGIN - Inches(3.0), Inches(0.42), Inches(3.0),
       Inches(0.3), tag, size=11, color=GRAY, align=PP_ALIGN.RIGHT)


def main():
    if not BASE_PPTX.is_file():
        print("找不到底座 deck: %s" % BASE_PPTX)
        return 1
    prs = Presentation(str(BASE_PPTX))
    existing = len(prs.slides)
    print("  底座 deck: %d 页" % existing)

    # 新页按追加顺序生成，最后统一重排
    build_cover(prs)          # new[0]
    build_agenda(prs)         # new[1]
    build_part_divider(prs)   # new[2]
    build_bc_setup(prs)       # new[3]
    build_bc_win(prs)         # new[4]
    build_bc_negative(prs)    # new[5]
    build_bc_split(prs)       # new[6]
    build_bc_mirror(prs)      # new[7]
    build_bc_1d_design(prs)   # new[8]
    build_bc_1d_result(prs)   # new[9]
    build_bc_lessons(prs)     # new[10]
    build_summary(prs)        # new[11]

    total = len(prs.slides)
    new_count = total - existing
    base_idx = list(range(existing))
    new_idx = list(range(existing, total))

    # 目标顺序：
    #   封面 / 目录 / [分割头 10 页] / 第二部分分隔页 / 条码 8 页 /
    #   总结 / [附录 4 页]
    order = ([new_idx[0], new_idx[1]] + base_idx[:10] + [new_idx[2]] +
             new_idx[3:11] + [new_idx[11]] + base_idx[10:])
    # 分部标签加在底座 deck 的封面上，必须在重排之前加，
    # 否则 prs.slides[0] 已经是新做的总封面了
    add_part_tag(prs, existing, "第一部分  ·  分割头源码对比")
    reorder_slides(prs, order)
    refresh_page_numbers(prs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT_PPTX))
    print("  新页 %d 页，总计 %d 页 -> %s" % (new_count, total, OUT_PPTX))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
