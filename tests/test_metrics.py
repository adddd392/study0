"""metrics 模块的验收测试。

契约::

    confusion_matrix(targets, predictions, num_classes) -> list[list[int]]
        行 = 真实标签，列 = 预测标签

    per_class_recall / per_class_precision / per_class_f1(matrix)
        -> list[float | None]
        分母为 0（该类没有样本 / 没被预测）时返回 None，不是 0.0

    balanced_accuracy(matrix) -> float | None
        = 各类召回率（跳过 None）的算术平均

    classification_report(targets, predictions, num_classes, class_names=None)
        -> dict，键固定，见模块文档

    format_report(report, title=...) -> str
        多行文本，可直接写入 results.txt

本文件刻意只用整数和几个手算的小例子：指标的错法只有几种（行列写反、
分母取错、None 与 0 混淆），用真实的 979 条预测去测反而看不清。

运行::

    python tests/test_metrics.py
"""

import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import metrics  # noqa: E402


def case_confusion_matrix_layout() -> None:
    """混淆矩阵必须是"行=真实、列=预测"，且每格计数正确。

    设计一个 3 类的小样本，把每一格都填上可区分的数字：

        真实 0 -> 预测 0: 3 次，预测 1: 1 次
        真实 1 -> 预测 1: 2 次，预测 2: 1 次
        真实 2 -> 预测 2: 4 次
    """
    targets = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
    predictions = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2]
    matrix = metrics.confusion_matrix(targets, predictions, 3)
    expected = [
        [3, 1, 0],
        [0, 2, 1],
        [0, 0, 4],
    ]
    assert matrix == expected, f"混淆矩阵错误: {matrix} != {expected}"

    # 行和 = 每类真实样本数，列和 = 每类被预测次数
    assert metrics.per_class_support(matrix) == [4, 3, 4], "行和错误"
    assert metrics.per_class_prediction_count(matrix) == [3, 3, 5], "列和错误"
    print("混淆矩阵方向与计数: OK")


def case_recall_precision_f1() -> None:
    """在已知矩阵上逐项核对召回/精确率/F1（手算值）。"""
    matrix = [
        [3, 1, 0],
        [0, 2, 1],
        [0, 0, 4],
    ]
    recalls = metrics.per_class_recall(matrix)
    precisions = metrics.per_class_precision(matrix)
    f1s = metrics.per_class_f1(matrix)

    # 召回 = 对角线 / 行和
    expected_recall = [3 / 4, 2 / 3, 4 / 4]
    # 精确率 = 对角线 / 列和
    expected_precision = [3 / 3, 2 / 3, 4 / 5]
    for index, expected in enumerate(expected_recall):
        assert abs(recalls[index] - expected) < 1e-12, (
            f"第 {index} 类召回期望 {expected}，实际 {recalls[index]}"
        )
    for index, expected in enumerate(expected_precision):
        assert abs(precisions[index] - expected) < 1e-12, (
            f"第 {index} 类精确率期望 {expected}，实际 {precisions[index]}"
        )

    # F1 = 2PR / (P + R)
    for index in range(3):
        precision = expected_precision[index]
        recall = expected_recall[index]
        expected = 2 * precision * recall / (precision + recall)
        assert abs(f1s[index] - expected) < 1e-12, (
            f"第 {index} 类 F1 期望 {expected}，实际 {f1s[index]}"
        )

    # 总体准确率 = (3+2+4) / 11
    assert abs(metrics.overall_accuracy(matrix) - 9 / 11) < 1e-12, (
        "总体准确率错误"
    )
    print("每类召回/精确率/F1 与总体准确率: OK")


def case_undefined_is_none_not_zero() -> None:
    """没有样本的类别，指标必须是 None，不能是 0.0。

    这是本模块最容易被写错的地方。若把"无样本"当成 0 召回，
    balanced accuracy 会被冤杀；当成 1.0 又会虚高。
    """
    # 第 2 类在真实标签里一次都没出现，也从没被预测过
    matrix = metrics.confusion_matrix([0, 0, 1, 1], [0, 0, 1, 1], 3)
    support = metrics.per_class_support(matrix)
    assert support == [2, 2, 0], f"支持数错误: {support}"

    recalls = metrics.per_class_recall(matrix)
    assert recalls[0] == 1.0, "第 0 类全对，召回应为 1.0"
    assert recalls[1] == 1.0, "第 1 类全对，召回应为 1.0"
    assert recalls[2] is None, (
        f"第 2 类没有样本，召回应为 None，实际 {recalls[2]!r}"
    )

    # balanced accuracy 只对有定义的类求平均 -> (1.0 + 1.0) / 2
    assert metrics.balanced_accuracy(matrix) == 1.0, (
        "balanced accuracy 应跳过 None 后求平均"
    )
    print("无样本类别返回 None（而非 0/1）: OK")


def case_class_with_no_prediction() -> None:
    """某类有样本但一次都没被预测：召回 0、精确率 0、F1 为 0。

    注意精确率是 0.0 而不是 None。若判成 None，macro 平均会跳过它，
    "干脆不预测小类"就能让 macro F1 变高——那是反向激励。
    """
    # 真实标签有 0 和 1，但预测全是 1
    matrix = metrics.confusion_matrix([0, 0, 1, 1], [1, 1, 1, 1], 2)
    recalls = metrics.per_class_recall(matrix)
    precisions = metrics.per_class_precision(matrix)
    f1s = metrics.per_class_f1(matrix)

    assert recalls[0] == 0.0, "第 0 类全判错，召回必须是 0.0（不是 None）"
    assert precisions[0] == 0.0, (
        f"第 0 类有样本却没被预测到，精确率应为 0.0，实际 {precisions[0]!r}"
    )
    assert f1s[0] == 0.0, f"第 0 类的 F1 应为 0.0，实际 {f1s[0]!r}"
    assert recalls[1] == 1.0 and abs(precisions[1] - 0.5) < 1e-12, (
        "第 1 类召回 1.0、精确率 2/4"
    )
    # 总体准确率 50%，balanced accuracy 也是 50%
    assert abs(metrics.overall_accuracy(matrix) - 0.5) < 1e-12, (
        "总体准确率应为 50%"
    )
    assert abs(metrics.balanced_accuracy(matrix) - 0.5) < 1e-12, (
        "balanced accuracy 应为 (0.0 + 1.0) / 2 = 0.5"
    )
    # macro F1 必须被小类的 0 拉低，而不是跳过它
    # 第 1 类 F1 = 2*0.5*1/(0.5+1) = 0.6667
    assert abs(metrics.macro_f1(matrix) - 0.3333) < 1e-3, (
        f"macro F1 应约为 0.3333，实际 {metrics.macro_f1(matrix)}"
    )
    print("有样本但零预测的类别（精确率记 0 而非 None）: OK")


def case_imbalanced_sees_minority_failure() -> None:
    """核心动机：不均衡二分类里"小类全错"，总体准确率骗人、balanced 不骗。

    模拟条码数据集的比例（C128 少、EAN13 多）：
    100 个样本里 20 个 C128、80 个 EAN13，模型全判 EAN13。
    - 总体准确率 = 80%（看着"可用"）
    - balanced accuracy = 50%（一眼看出小类完全失效）
    """
    targets = [0] * 20 + [1] * 80
    predictions = [1] * 100
    report = metrics.classification_report(
        targets, predictions, 2, class_names=["C128", "EAN13"]
    )

    assert abs(report["accuracy"] - 0.80) < 1e-12, (
        f"总体准确率应为 80%，实际 {report['accuracy']}"
    )
    assert abs(report["balanced_accuracy"] - 0.50) < 1e-12, (
        "小类召回为 0，balanced accuracy 必须是 50%"
    )
    assert report["recall"] == [0.0, 1.0], f"召回错误: {report['recall']}"
    assert report["support"] == [20, 80], f"支持数错误: {report['support']}"
    # macro F1 必须被小类的 0 拉低：(0.0 + 0.8889) / 2 ≈ 44.44%
    assert report["macro_f1"] < 0.45, (
        f"小类完全失效时 macro F1 不该好看，实际 {report['macro_f1']}"
    )
    print("不均衡场景下 balanced accuracy 暴露小类失效: OK")


def case_macro_average_skips_none() -> None:
    """macro 平均只对已定义的类别求平均。"""
    matrix = metrics.confusion_matrix([0, 0, 1, 1], [0, 0, 1, 1], 3)
    recalls = metrics.per_class_recall(matrix)      # [1.0, 1.0, None]
    assert metrics.mean_of_defined(recalls) == 1.0, "应跳过 None"
    assert metrics.macro_f1(matrix) is not None, "macro F1 不应为 None"

    # 一个样本都没有时，所有指标都是 None，不应抛 ZeroDivisionError
    empty = metrics.confusion_matrix([], [], 2)
    assert metrics.overall_accuracy(empty) is None, "空输入的准确率应为 None"
    assert metrics.balanced_accuracy(empty) is None, "空输入的 balanced 应为 None"
    assert metrics.macro_f1(empty) is None, "空输入的 macro F1 应为 None"
    print("空数据集不抛异常、返回 None: OK")


def case_input_validation() -> None:
    """长度不等 / 标签越界 / 非整数必须报错，而不是算出一个假指标。"""
    def expect_error(function, description):
        try:
            function()
        except ValueError:
            return
        raise AssertionError(f"应当报错却没有报错: {description}")

    expect_error(
        lambda: metrics.confusion_matrix([0, 1], [0], 2),
        "两侧长度不一致",
    )
    expect_error(
        lambda: metrics.confusion_matrix([0, 2], [0, 1], 2),
        "真实标签越界",
    )
    expect_error(
        lambda: metrics.confusion_matrix([0, 1], [0, -1], 2),
        "预测标签越界（负数）",
    )
    expect_error(
        lambda: metrics.confusion_matrix([0, 1], [0, 1], 0),
        "类别数非正",
    )
    expect_error(
        lambda: metrics.classification_report([0], [0], 2, ["a"]),
        "class_names 长度不匹配",
    )
    print("输入校验: OK")


def case_format_report_text() -> None:
    """报告文本必须包含所有关键行，未定义指标显示为 "—"。"""
    targets = [0] * 20 + [1] * 80
    predictions = [1] * 100
    report = metrics.classification_report(
        targets, predictions, 2, ["C128", "EAN13"]
    )
    text = metrics.format_report(report, title="分类报告（测试集）")

    for expected in ("分类报告（测试集）", "样本数: 100", "总体准确率: 80.00%",
                     "balanced accuracy: 50.00%", "混淆矩阵", "每类指标",
                     "C128", "EAN13", "召回率", "精确率"):
        assert expected in text, f"报告缺少 {expected!r}:\n{text}"

    # C128 有样本但没被预测过 -> 精确率 0.00%，F1 0.00%
    assert "0.00%" in text, f"未预测类别的精确率应显示 0.00%:\n{text}"

    # 矩阵第一行第二格应为 20（C128 全被猜成 EAN13）
    matrix_line = [line for line in text.splitlines() if "C128" in line]
    assert any("20" in line for line in matrix_line), (
        f"混淆矩阵缺少误判计数 20:\n{text}"
    )

    # 真有类别在测试集里没有样本时，指标显示为中划线
    absent = metrics.classification_report([0, 1], [0, 1], 3)
    absent_text = metrics.format_report(absent)
    assert "—" in absent_text, (
        f"无样本类别应显示为中划线:\n{absent_text}"
    )
    print("报告文本渲染: OK")


def main() -> int:
    case_confusion_matrix_layout()
    case_recall_precision_f1()
    case_undefined_is_none_not_zero()
    case_class_with_no_prediction()
    case_imbalanced_sees_minority_failure()
    case_macro_average_skips_none()
    case_input_validation()
    case_format_report_text()
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
