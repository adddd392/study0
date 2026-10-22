"""
绘图工具：把若干条曲线画进一张图，保存为 PNG。

本模块只依赖 matplotlib，不引用项目内任何其它模块，
所以训练、推理、分析脚本都能直接复用。

设计要点：
- 所有绘制走显式 Axes 对象（ax.*），不碰 pyplot 的全局状态；
- figure 用完立刻关闭，批量画图不会持续占用内存；
- 输入校验前置，报错信息指明是哪条曲线出问题。
"""

import os

import matplotlib.pyplot as plt


def save_curve(path, title, ylabel, series):
    """把若干条曲线画进一张图，保存为 PNG。

    参数:
        path:   输出图片的完整路径。父目录不存在时自动创建；
                也可以只给文件名，此时存到当前工作目录。
        title:  图标题。
        ylabel: y 轴标签；x 轴固定为 Epochs。
        series: 曲线列表，格式为 [(values, label), ...]。
                至少一条，且所有 values 的长度必须一致。

    返回:
        传入的 path，方便调用方记录日志或做断言。

    异常:
        ValueError: series 为空，或各序列长度不一致。
    """
    # 先校验再画图：拿到明确的中文报错，而不是 matplotlib 那句
    # "x and y must have same first dimension"
    _validate_series(series)

    # x 轴长度取自数据本身，而不是外部的 num_epochs。
    # 这样无论传入多少个点都能画，调用方不必保证两者一致。
    x_axis = range(1, len(series[0][0]) + 1)

    # 拿到显式的 Figure / Axes。
    # plt.subplots() 一次创建"一个画布 + 一个坐标系"，
    # 之后所有绘制都用 ax.*，不再用 plt.plot 那种"画到当前 figure"的写法。
    fig, ax = plt.subplots()

    for values, label in series:
        ax.plot(x_axis, values, label=label)

    ax.set_title(title)
    ax.set_xlabel("Epochs")
    ax.set_ylabel(ylabel)
    ax.legend()

    # 父目录可能是空字符串：只传文件名时 os.path.dirname 返回 ""，
    # 直接 os.makedirs("") 会抛 FileNotFoundError，所以要先判断。
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    fig.savefig(path)

    # pyplot 会一直持有所有未关闭 figure 的引用，
    # 绘图循环里不关会持续吃内存，必须显式关闭。
    plt.close(fig)

    return path


def _validate_series(series):
    """校验 series 非空且各序列长度一致。

    抛出的报错要能直接定位问题曲线，所以带上 label 和实际长度。
    """
    if not series:
        raise ValueError("series 不能为空，至少要有一条曲线")

    first_label, first_len = series[0][1], len(series[0][0])
    for values, label in series[1:]:
        length = len(values)
        if length != first_len:
            raise ValueError(
                f"曲线 {label!r} 长度 {length} 与 "
                f"{first_label!r} 长度 {first_len} 不一致"
            )
