"""plotting 模块的验收测试。

契约::

    save_curve(path, title, ylabel, series) -> str
        path   : 输出 PNG 的完整路径，父目录不存在时自动创建
        title  : 图标题
        ylabel : y 轴标签
        series : [(values, label), ...]，至少一条；各序列长度必须一致

    行为:
    - x 轴统一为 1..len(values)
    - 用显式 Axes 绘制，调用结束后不残留 figure
    - 返回传入的 path

    错误:
    - series 为空            -> ValueError
    - 各序列长度不一致       -> ValueError

运行::

    python tests/test_plotting.py

注意：本文件里图标题/轴标签一律用英文。
matplotlib 默认字体（DejaVu Sans）不含中文字形，传中文会渲染成方框，
并触发 UserWarning: Glyph xxxxx missing from font(s) DejaVu Sans。
确需中文标题，需先设置 rcParams["font.sans-serif"]，例如 SimHei。
"""

import contextlib
import os
import shutil
import sys
import uuid

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

import plotting  # noqa: E402

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


def assert_png(path):
    """断言文件存在且是合法 PNG。"""
    assert os.path.isfile(path), f"未生成 {path}"
    assert os.path.getsize(path) > 0, f"{path} 是空文件"
    with open(path, "rb") as handle:
        head = handle.read(8)
    assert head == PNG_MAGIC, f"{path} 不是合法 PNG，头部字节 {head!r}"


def case_single_series_creates_parent_dir() -> None:
    """只有一条曲线；父目录不存在时要自动创建。"""
    with temp_dir() as tmp:
        # 故意多套两层不存在的目录
        path = os.path.join(tmp, "a", "b", "c", "single.png")
        assert not os.path.isdir(os.path.dirname(path))

        returned = plotting.save_curve(
            path, "Single", "value", [([1.0, 2.0, 3.0], "train")]
        )

        assert_png(path)
        assert os.path.abspath(returned) == os.path.abspath(path), (
            f"返回值应为 {path}，实际 {returned}"
        )
    print("save_curve (单曲线 + 自动建目录 + 返回路径): OK")


def case_multi_series() -> None:
    """多条曲线连同验证集一起画。"""
    with temp_dir() as tmp:
        path = os.path.join(tmp, "multi.png")
        plotting.save_curve(
            path,
            "Multi",
            "Accuracy (%)",
            [
                ([10.0, 20.0, 30.0], "train top1"),
                ([30.0, 40.0, 50.0], "train top5"),
                ([8.0, 18.0, 28.0], "val top1"),
            ],
        )
        assert_png(path)
    print("save_curve (多曲线): OK")


def case_empty_series_rejected() -> None:
    """series 为空必须报 ValueError，不能默默画出空图。"""
    with temp_dir() as tmp:
        path = os.path.join(tmp, "empty.png")
        try:
            plotting.save_curve(path, "Empty", "value", [])
        except ValueError:
            pass
        else:
            raise AssertionError("series 为空时应抛 ValueError")
    print("save_curve (空 series 报错): OK")


def case_length_mismatch_rejected() -> None:
    """各序列长度不一致必须报 ValueError。"""
    with temp_dir() as tmp:
        path = os.path.join(tmp, "mismatch.png")
        try:
            plotting.save_curve(
                path,
                "Mismatch",
                "value",
                [([1.0, 2.0, 3.0], "a"), ([1.0, 2.0], "b")],
            )
        except ValueError:
            pass
        else:
            raise AssertionError("序列长度不一致时应抛 ValueError")
    print("save_curve (长度不一致报错): OK")


def case_no_figure_leak() -> None:
    """调用结束后不能残留打开的 figure。"""
    with temp_dir() as tmp:
        before = len(plt.get_fignums())
        plotting.save_curve(
            os.path.join(tmp, "leak.png"), "Leak", "value", [([1.0, 2.0], "a")]
        )
        after = len(plt.get_fignums())
        assert after == before, (
            f"figure 泄漏：调用前 {before} 个，调用后 {after} 个"
        )
    print("figure 无泄漏: OK")


def main() -> int:
    case_single_series_creates_parent_dir()
    case_multi_series()
    case_empty_series_rejected()
    case_length_mismatch_rejected()
    case_no_figure_leak()
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
