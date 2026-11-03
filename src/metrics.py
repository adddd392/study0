"""分类评估指标：混淆矩阵、每类召回/精确率/F1、balanced accuracy。

为什么需要这个模块
------------------
训练脚本原本只报 Top-1 / Top-5。两个问题：

1. **2 类任务上 Top-5 恒为 100%**。torch.topk 的 k 被压到类别数后，
   Top-5 就是"把两类都猜一遍"，必然全中，没有任何信息量。
2. **总体准确率看不见小类**。条码数据集 C128 : EAN13 约 1 : 3.3，
   如果模型把所有 C128 都判成 EAN13，总体准确率仍有 75% 左右，
   但 C128 的召回是 0。这种"整体看着还行、某一类完全失效"的情况，
   只有混淆矩阵和每类召回能看出来。

设计原则
--------
1. **零重依赖**：只接收"真实标签序列 + 预测标签序列 + 类别数"，
   不 import torch / numpy。好处是可以在任何环境、任何设备上单测，
   也不需要为了测一个除法而构造张量。
2. **"没有样本"和"全错"必须区分**。某类在测试集里一个样本都没有时，
   它的召回率是**未定义**，返回 None 而不是 0.0。若返回 0.0，
   balanced accuracy 会被凭空拉低；若返回 1.0，则会被凭空抬高。
3. **比率一律用 0~1 的浮点表示**，百分号只在报告文本里格式化。
   这样调用方要做阈值判断时不必先除 100。

指标口径
--------
    召回率 recall    = 该类被正确预测数 / 该类真实样本数
    精确率 precision = 该类被正确预测数 / 被判为该类的样本数
    F1              = 2 * P * R / (P + R)
    balanced acc    = 各类召回率的算术平均（跳过未定义的类）
    macro F1        = 各类 F1 的算术平均（跳过未定义的类）

"跳过未定义"只针对**测试集里没有样本**的类别（它不属于本次评估）。
反过来，类别有样本却一次都没被预测到时，精确率与 F1 判 0.0 而不是
未定义——否则"完全不预测小类"会让 macro 指标反而变高。

balanced accuracy 与总体准确率的区别：前者对每个类别一视同仁，
后者按样本数加权。类别不均衡时以 balanced accuracy 为准。
"""

import unicodedata
from typing import Dict, List, Optional, Sequence

# 一个"比率"要么是 [0, 1] 的浮点，要么是 None（分母为 0，未定义）
Ratio = Optional[float]


def _validate(targets, predictions, num_classes):
    """检查输入合法性。

    指标算错的原因里，最常见的是"标签越界"和"两边长度不一致"，
    这两种错误如果不当场报错，会静默算出一个看似合理的数字
    （比如越界下标直接写进混淆矩阵的负索引位置）。

    :raises ValueError: 长度不等、类别数非正、标签非整数或越界
    """
    if num_classes <= 0:
        raise ValueError(f"num_classes 必须为正整数，收到 {num_classes}")
    if len(targets) != len(predictions):
        raise ValueError(
            f"真实标签与预测标签数量不一致: "
            f"{len(targets)} != {len(predictions)}"
        )
    for name, labels in (("真实标签", targets), ("预测标签", predictions)):
        for label in labels:
            # bool 是 int 的子类，显式排除，避免 True/False 被当成 1/0
            if isinstance(label, bool) or not isinstance(label, int):
                raise ValueError(f"{name}必须是整数下标，收到 {label!r}")
            if not 0 <= label < num_classes:
                raise ValueError(
                    f"{name} {label} 越界，应在 [0, {num_classes}) 内"
                )


def confusion_matrix(targets, predictions, num_classes) -> List[List[int]]:
    """统计混淆矩阵，返回 num_classes × num_classes 的嵌套列表。

    约定 **行 = 真实标签，列 = 预测标签**（sklearn 同此约定）。
    矩阵第 i 行所有元素之和 = 第 i 类的真实样本数；
    第 j 列所有元素之和 = 被预测为第 j 类的样本数。
    """
    _validate(targets, predictions, num_classes)
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for true_label, predicted_label in zip(targets, predictions):
        matrix[true_label][predicted_label] += 1
    return matrix


def per_class_support(matrix) -> List[int]:
    """每类真实样本数（行和）。"""
    return [sum(row) for row in matrix]


def per_class_prediction_count(matrix) -> List[int]:
    """每类被预测了多少次（列和）。它和 support 一般不等，差值就是误判。"""
    num_classes = len(matrix)
    return [
        sum(matrix[row][column] for row in range(num_classes))
        for column in range(num_classes)
    ]


def _safe_ratio(numerator, denominator) -> Ratio:
    """除法，分母为 0 时返回 None（未定义），而不是抛异常或给 0。"""
    if denominator == 0:
        return None
    return numerator / denominator


def per_class_recall(matrix) -> List[Ratio]:
    """每类召回率 = 对应对角线 / 该类样本数。"""
    support = per_class_support(matrix)
    return [
        _safe_ratio(matrix[index][index], support[index])
        for index in range(len(matrix))
    ]


def per_class_precision(matrix) -> List[Ratio]:
    """每类精确率 = 对应对角线 / 被判为该类的样本数。

    分母为 0 时分两种情况，这个区别很关键：

    - 该类在测试集里**根本没有样本**（support == 0）→ 返回 None。
      它不属于本次评估，不该进入 macro 平均。
    - 该类**有样本但一次都没被预测到**（support > 0）→ 返回 0.0。
      数学上精确率确实未定义，但如果返回 None 让它从 macro 平均里消失，
      "干脆完全不预测小类"反而会让 macro F1 变高——这是反向激励，
      也正好掩盖了本模块要暴露的那种失效。所以这里判 0.0。
    """
    support = per_class_support(matrix)
    predicted = per_class_prediction_count(matrix)
    precisions = []
    for index in range(len(matrix)):
        if predicted[index] == 0:
            precisions.append(0.0 if support[index] > 0 else None)
        else:
            precisions.append(matrix[index][index] / predicted[index])
    return precisions


def per_class_f1(matrix) -> List[Ratio]:
    """每类 F1 = 2PR / (P + R)。

    - P 或 R 为 None（该类无样本，不属于本次评估）→ None
    - P 和 R 都是 0（有样本但全判错 / 从没预测到）→ 0.0，
      这是有意义的"完全失效"，必须进入 macro 平均
    """
    recalls = per_class_recall(matrix)
    precisions = per_class_precision(matrix)
    scores = []
    for recall, precision in zip(recalls, precisions):
        if recall is None or precision is None:
            scores.append(None)
        elif recall + precision == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return scores


def overall_accuracy(matrix) -> Ratio:
    """总体准确率 = 对角线之和 / 样本总数。"""
    total = sum(per_class_support(matrix))
    correct = sum(matrix[index][index] for index in range(len(matrix)))
    return _safe_ratio(correct, total)


def mean_of_defined(values: Sequence[Ratio]) -> Ratio:
    """只对"有定义"（非 None）的值求平均；全是 None 时返回 None。"""
    defined = [value for value in values if value is not None]
    if not defined:
        return None
    return sum(defined) / len(defined)


def balanced_accuracy(matrix) -> Ratio:
    """各类召回率的算术平均。类别不均衡时比总体准确率可靠得多。"""
    return mean_of_defined(per_class_recall(matrix))


def macro_f1(matrix) -> Ratio:
    """各类 F1 的算术平均。"""
    return mean_of_defined(per_class_f1(matrix))


def classification_report(
    targets,
    predictions,
    num_classes,
    class_names=None,
) -> Dict:
    """把一次评估的所有指标算成一个字典。

    字典的键固定，方便调用方按需取值，也方便直接写成 JSON 存进实验记录：

        num_samples / num_classes / class_names
        matrix / support / prediction_count
        recall / precision / f1
        accuracy / balanced_accuracy / macro_f1

    :param class_names: 类别下标 → 名称，长度必须等于 num_classes。
                        传 None 时用 "0"、"1" 这样的下标字符串占位。
    """
    if class_names is not None and len(class_names) != num_classes:
        raise ValueError(
            f"class_names 长度 {len(class_names)} 与类别数 "
            f"{num_classes} 不一致"
        )
    matrix = confusion_matrix(targets, predictions, num_classes)
    if class_names is None:
        class_names = [str(index) for index in range(num_classes)]
    return {
        "num_samples": len(targets),
        "num_classes": num_classes,
        "class_names": list(class_names),
        "matrix": matrix,
        "support": per_class_support(matrix),
        "prediction_count": per_class_prediction_count(matrix),
        "recall": per_class_recall(matrix),
        "precision": per_class_precision(matrix),
        "f1": per_class_f1(matrix),
        "accuracy": overall_accuracy(matrix),
        "balanced_accuracy": balanced_accuracy(matrix),
        "macro_f1": macro_f1(matrix),
    }


def _display_width(text):
    """计算字符串在等宽终端里的显示宽度。

    中日韩字符（east_asian_width 为 W/F）占两列。用 len() 对齐会让
    含中文的类别名整体错位，所以这里按显示宽度补空格。
    """
    width = 0
    for char in text:
        width += 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
    return width


def _pad_right(text, width):
    """右侧补空格到指定显示宽度。"""
    return text + " " * max(0, width - _display_width(text))


def _pad_left(text, width):
    """左侧补空格到指定显示宽度（数字右对齐用）。"""
    return " " * max(0, width - _display_width(text)) + text


def _percent(value: Ratio) -> str:
    """把 [0,1] 的比率格式化成 "92.34%"，未定义时显示 "—"。"""
    if value is None:
        return "—"
    return f"{value * 100:.2f}%"


def format_report(report, title="分类报告") -> str:
    """把 classification_report 的字典渲染成可写入文本文件的多行字符串。

    之所以单独一个函数：results.txt 是给人看的，一个 2 类任务光看
    "Top-1: 92%" 无法判断哪一类失效；打印成表格后一眼能看出
    "C128 召回 91%、EAN13 召回 92%" 还是 "C128 召回 0%"。
    """
    names = report["class_names"]
    num_classes = report["num_classes"]
    # 类别名列宽度：至少 8，取最长的类别名，再加 2 列间隔
    name_width = max(8, max(_display_width(name) for name in names) + 2)
    cell_width = 10

    lines = []
    lines.append("=" * 50)
    lines.append(f"{title}")
    lines.append("=" * 50)
    lines.append(
        f"样本数: {report['num_samples']}    "
        f"类别数: {num_classes}"
    )
    lines.append(f"总体准确率: {_percent(report['accuracy'])}")
    lines.append(
        f"balanced accuracy: {_percent(report['balanced_accuracy'])}"
        "  （各类召回的平均，看类别不均衡时以它为准）"
    )
    lines.append(f"macro F1: {_percent(report['macro_f1'])}")
    lines.append("")

    lines.append("混淆矩阵（行 = 真实标签，列 = 预测标签）")
    header = _pad_right("真实 \\ 预测", name_width)
    header += "".join(
        _pad_left(name, cell_width) for name in names
    )
    lines.append(header)
    for row_index in range(num_classes):
        row = _pad_right(names[row_index], name_width)
        row += "".join(
            _pad_left(str(value), cell_width)
            for value in report["matrix"][row_index]
        )
        lines.append(row)
    lines.append("")

    lines.append("每类指标")
    lines.append(
        _pad_right("类别", name_width)
        + _pad_left("支持数", cell_width)
        + _pad_left("召回率", cell_width)
        + _pad_left("精确率", cell_width)
        + _pad_left("F1", cell_width)
    )
    for index in range(num_classes):
        lines.append(
            _pad_right(names[index], name_width)
            + _pad_left(str(report["support"][index]), cell_width)
            + _pad_left(_percent(report["recall"][index]), cell_width)
            + _pad_left(_percent(report["precision"][index]), cell_width)
            + _pad_left(_percent(report["f1"][index]), cell_width)
        )
    return "\n".join(lines)
