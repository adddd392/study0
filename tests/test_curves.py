"""draw_loss_acc_curve 的验收测试。

契约::

    trainer.draw_loss_acc_curve(
        training_losses, training_Top1_accuracies, training_Top5_accuracies,
        validation_losses=None, validation_Top1_accuracies=None,
        validation_Top5_accuracies=None,
    )

    行为:
    - 在 <args.save_path>/curves/ 下生成两张 PNG：
        loss_curve.png
        accuracy_curve.png
    - x 轴为 1..N，N 取自数据序列长度（不得依赖 args.num_epochs）
    - validation_* 为 None 时跳过对应曲线，不报错
    - 调用结束后不残留打开的 figure（plt.get_fignums() 为空）

关注点：本测试只断言【外部可观察行为】（文件是否生成、是否合法 PNG、
序列长度与 x 轴是否自洽），不断言内部是否拆了私有方法。

运行::

    python tests/test_curves.py
"""

import argparse
import contextlib
import os
import shutil
import sys
import uuid

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# 必须在 import pyplot 之前选定后端，否则无显示环境会失败
import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

_TMP_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".tmp_tests")
)


@contextlib.contextmanager
def temp_dir():
    """在工作区内创建临时目录，退出时清理。"""
    os.makedirs(_TMP_ROOT, exist_ok=True)
    path = os.path.join(_TMP_ROOT, uuid.uuid4().hex)
    os.makedirs(path, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def make_trainer(save_path, num_epochs):
    """构造一个只够画图用的 trainer（不建模型、不读数据）。"""
    from train import trainer

    args = argparse.Namespace(
        device="cpu", save_path=save_path, num_epochs=num_epochs
    )
    return trainer(args)


def assert_png(path):
    """断言文件存在且是合法 PNG。"""
    assert os.path.isfile(path), f"未生成 {path}"
    size = os.path.getsize(path)
    assert size > 0, f"{path} 是空文件"
    with open(path, "rb") as handle:
        head = handle.read(8)
    assert head == PNG_MAGIC, f"{path} 不是合法 PNG，头部字节 {head!r}"


def case_basic() -> None:
    """训练 + 验证共 6 条曲线，两张图都要生成。"""
    with temp_dir() as tmp:
        save_path = os.path.join(tmp, "run1")
        os.makedirs(save_path, exist_ok=True)
        obj = make_trainer(save_path, num_epochs=5)

        obj.draw_loss_acc_curve(
            [2.5, 2.0, 1.5, 1.2, 1.0],
            [20.0, 35.0, 50.0, 60.0, 65.0],
            [50.0, 65.0, 80.0, 88.0, 92.0],
            [2.7, 2.2, 1.8, 1.5, 1.3],
            [18.0, 30.0, 45.0, 55.0, 60.0],
            [48.0, 60.0, 75.0, 85.0, 90.0],
        )

        curves = os.path.join(save_path, "curves")
        assert_png(os.path.join(curves, "loss_curve.png"))
        assert_png(os.path.join(curves, "accuracy_curve.png"))
    print("draw_loss_acc_curve (训练+验证): OK")


def case_training_only() -> None:
    """validation_* 全为 None 时，只画训练曲线，仍要出两张图。"""
    with temp_dir() as tmp:
        save_path = os.path.join(tmp, "run1")
        os.makedirs(save_path, exist_ok=True)
        obj = make_trainer(save_path, num_epochs=3)

        obj.draw_loss_acc_curve(
            [2.5, 2.0, 1.5],
            [20.0, 35.0, 50.0],
            [50.0, 65.0, 80.0],
        )

        curves = os.path.join(save_path, "curves")
        assert_png(os.path.join(curves, "loss_curve.png"))
        assert_png(os.path.join(curves, "accuracy_curve.png"))
    print("draw_loss_acc_curve (仅训练集): OK")


def case_length_from_data() -> None:
    """x 轴长度必须取自数据，而不是 args.num_epochs。

    args.num_epochs 故意设成 999，而数据只有 5 个点。
    如果实现仍按 num_epochs 生成 x 轴，matplotlib 会抛
    ValueError: x and y must have same first dimension。
    """
    with temp_dir() as tmp:
        save_path = os.path.join(tmp, "run1")
        os.makedirs(save_path, exist_ok=True)
        obj = make_trainer(save_path, num_epochs=999)

        obj.draw_loss_acc_curve(
            [2.5, 2.0, 1.5, 1.2, 1.0],
            [20.0, 35.0, 50.0, 60.0, 65.0],
            [50.0, 65.0, 80.0, 88.0, 92.0],
        )

        curves = os.path.join(save_path, "curves")
        assert_png(os.path.join(curves, "loss_curve.png"))
        assert_png(os.path.join(curves, "accuracy_curve.png"))
    print("x 轴长度取自数据而非 args.num_epochs: OK")


def case_no_figure_leak() -> None:
    """调用结束后不能残留打开的 figure。"""
    with temp_dir() as tmp:
        save_path = os.path.join(tmp, "run1")
        os.makedirs(save_path, exist_ok=True)
        obj = make_trainer(save_path, num_epochs=3)

        before = len(plt.get_fignums())
        obj.draw_loss_acc_curve(
            [1.0, 0.9, 0.8],
            [10.0, 20.0, 30.0],
            [30.0, 40.0, 50.0],
        )
        after = len(plt.get_fignums())

        assert after == before, (
            f"figure 泄漏：调用前 {before} 个，调用后 {after} 个"
        )
    print("figure 无泄漏: OK")


def main() -> int:
    case_basic()
    case_training_only()
    case_length_from_data()
    case_no_figure_leak()
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
