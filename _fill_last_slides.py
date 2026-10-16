# -*- coding: utf-8 -*-
"""填充 docs/第一周汇报.pptx 末页：目前的问题 + 下周计划。

策略：
  - 不改动已有 24 页的任何内容（表格/公式图/图片都不碰）
  - 复用原 deck 的表格样式（从第 23 页 deepcopy tableStyleId）
  - 第 24 页用已有的「目前的问题，下周计划」页，内容区填入问题 + 计划 + 一句话结论
  - 输出到新文件，保留原文件不动
"""
import copy
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn

SRC = r"D:\Users\admin\Desktop\代码\study\docs\第一周汇报.pptx"
DST = r"D:\Users\admin\Desktop\代码\study\docs\第一周汇报.pptx"

FONT = "微软雅黑"
NAVY = RGBColor(0x1F, 0x38, 0x64)
BODY = RGBColor(0x26, 0x26, 0x26)
RED = RGBColor(0xC0, 0x00, 0x00)
GREEN = RGBColor(0x1F, 0x7A, 0x3D)


def set_run_font(run, size, bold=False, color=BODY):
    """设置字体，含东亚字体（name 只影响 latin，中文需单独设 ea）。"""
    f = run.font
    f.size = Pt(size)
    f.bold = bold
    f.name = FONT
    f.color.rgb = color
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:latin", "a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = rPr.makeelement(qn(tag), {})
            rPr.append(el)
        el.set("typeface", FONT)


def add_textbox(slide, left, top, width, height):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    return tb, tf


def fill_para(para, text, size, bold=False, color=BODY, space_after=0, align=PP_ALIGN.LEFT):
    """把文本写进 paragraph：首行复用已有空 run 的就清空。"""
    para.alignment = align
    if space_after:
        para.space_after = Pt(space_after)
    if para.runs:
        run = para.runs[0]
        run.text = text
    else:
        run = para.add_run()
        run.text = text
    set_run_font(run, size, bold, color)
    return run


def new_table(slide, rows, cols, left, top, width, height, style_src):
    gf = slide.shapes.add_table(rows, cols, Inches(left), Inches(top), Inches(width), Inches(height))
    tbl = gf.table
    # 复用原 deck 的表格配色/边框
    src_pr = style_src._tbl.find(qn("a:tblPr"))
    dst_pr = tbl._tbl.find(qn("a:tblPr"))
    if src_pr is not None and dst_pr is not None:
        sid = src_pr.find(qn("a:tableStyleId"))
        if sid is not None:
            dst_sid = dst_pr.find(qn("a:tableStyleId"))
            if dst_sid is None:
                dst_pr.append(copy.deepcopy(sid))
            else:
                dst_sid.text = sid.text
    return tbl


def put_cell(cell, lines, size, bold=False, color=BODY, align=PP_ALIGN.LEFT):
    """lines 为 str 或多行 list；首行用 cell 自带段落避免多出空行。"""
    if isinstance(lines, str):
        lines = [lines]
    tf = cell.text_frame
    tf.word_wrap = True
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    cell.margin_left = Inches(0.10)
    cell.margin_right = Inches(0.08)
    cell.margin_top = Inches(0.04)
    cell.margin_bottom = Inches(0.04)
    # 清掉多余段落
    for p in list(tf.paragraphs[1:]):
        p._p.getparent().remove(p._p)
    for i, line in enumerate(lines):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        fill_para(para, line, size, bold, color)


def main():
    prs = Presentation(SRC)

    # ---------- 参考样式来源：第 23 页的表格 ----------
    style_src = None
    for sh in prs.slides[22].shapes:
        if sh.has_table:
            style_src = sh.table
            break
    assert style_src is not None, "找不到参考表格"

    # ---------- 目标页 = 第 24 页 ----------
    slide = prs.slides[23]

    # 副标题
    for sh in slide.shapes:
        if sh.name == "TextBox 3" and sh.has_text_frame:
            fill_para(sh.text_frame.paragraphs[0], "代码仓库规范差距 · 下一步重构与学习路线",
                      13.5, False, RGBColor(0x59, 0x59, 0x59))
            break

    # ============ 表 1：目前的问题 ============
    tbl1 = new_table(slide, rows=6, cols=3, left=0.55, top=1.68, width=12.25, height=2.50,
                     style_src=style_src)
    tbl1.columns[0].width = Inches(1.20)
    tbl1.columns[1].width = Inches(6.45)
    tbl1.columns[2].width = Inches(4.60)
    tbl1.rows[0].height = Inches(0.38)
    for r in range(1, 6):
        tbl1.rows[r].height = Inches(0.42)

    hdr = ["类别", "具体表现（已量化）", "后果"]
    for c, txt in enumerate(hdr):
        put_cell(tbl1.cell(0, c), txt, 12, True, RGBColor(0xFF, 0xFF, 0xFF))

    rows1 = [
        ("模块化", "仅 1 个 trainer 类（401 行 / 22 函数）承载全部功能，"
                   "未达到模块化清晰、对象化程度高", "无分层，难复用、难单测"),
        ("PEP8", "超长行 56 处（>120 列 8 处）；= 两侧缺空格约 140 行；"
                 "未使用导入 6 个；类前缺空行", "可读性不一致"),
        ("注释", "docstring 0 个（21 个方法全无）；注释共 9 行且全在 train.py",
         "读代码靠猜意图"),
        ("重点学习内容", "模型 / 激活 / 损失 / 卷积全部直接调用 torchvision 预设，"
                       "无一行自研实现", "逐行注释无实现可落笔"),
        ("实验进度", "基准对比与 CE vs LS 消融已跑通（第 17–21 页）",
         "dev 版可复用，作为重构基线"),
    ]
    for i, (a, b, c) in enumerate(rows1, start=1):
        put_cell(tbl1.cell(i, 0), a, 12, True, NAVY)
        put_cell(tbl1.cell(i, 1), b, 11.5, False, BODY)
        put_cell(tbl1.cell(i, 2), c, 11.5, False, GREEN if i == 5 else RED)

    # ============ 表 2：下周计划 ============
    tbl2 = new_table(slide, rows=3, cols=2, left=0.55, top=4.32, width=12.25, height=1.62,
                     style_src=style_src)
    tbl2.columns[0].width = Inches(1.20)
    tbl2.columns[1].width = Inches(11.05)
    tbl2.rows[0].height = Inches(0.34)
    tbl2.rows[1].height = Inches(0.64)
    tbl2.rows[2].height = Inches(0.64)

    put_cell(tbl2.cell(0, 0), "层面", 12, True, RGBColor(0xFF, 0xFF, 0xFF))
    put_cell(tbl2.cell(0, 1), "下周计划", 12, True, RGBColor(0xFF, 0xFF, 0xFF))
    put_cell(tbl2.cell(1, 0), "代码", 12, True, NAVY)
    put_cell(tbl2.cell(1, 1), [
        "① 先搭框架：直接仿照论文结构写骨架，只写类 / 函数接口与实现思路，不做具体实现",
        "② 按模块依次学习 → 逐个补齐实现，边实现边写逐行注释并 debug",
        "③ 全部模块完成后统一跑实验，与原 dev 版本结果对照",
    ], 11.5)
    put_cell(tbl2.cell(2, 0), "学习", 12, True, NAVY)
    put_cell(tbl2.cell(2, 1), [
        "① 先了解总体架构，再依次深入各模块的具体实现",
        "② 用 AI 出题、自己作答的方式巩固知识点",
    ], 11.5)

    # ============ 一句话归因 ============
    tb, tf = add_textbox(slide, 0.55, 6.06, 12.25, 1.05)
    fill_para(tf.paragraphs[0],
              "归因：初期仿照自己此前用 AI 跑通的结构，把功能全部塞进单个 trainer；"
              "模型 / 激活函数 / 损失函数 / 卷积均直接调用 torch 预设架构，"
              "既没有分层，也没有可供注释的实现。",
              11, False, RGBColor(0x59, 0x59, 0x59), align=PP_ALIGN.LEFT)
    p2 = tf.add_paragraph()
    fill_para(p2,
              "下一步：重构一份可对比的实现，边学边注释，把「学习重点」落到代码里。",
              11, True, NAVY)

    prs.save(DST)
    print("saved:", DST)


if __name__ == "__main__":
    main()
