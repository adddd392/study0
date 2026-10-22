"""switch_device 的行为检查。

覆盖三条路径：
    1. 正常路径：有模型时切换设备，模型跟着迁移
    2. 边界路径：模型还没创建（None）时切换设备，不应崩溃
    3. 错误路径：设备名非法时抛 ValueError，且对象状态完全不变

运行::

    python tests/test_switch_device.py
"""

import argparse
import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch.nn as nn  # noqa: E402

from train import trainer  # noqa: E402


def make_trainer(device: str = "cpu") -> trainer:
    """构造一个只初始化了设备的最小 trainer。"""
    return trainer(argparse.Namespace(device=device))


def case_normal() -> None:
    """正常路径：有模型时切设备，模型应迁移到新设备。"""
    t = make_trainer("cpu")
    t.model = nn.Linear(4, 2)

    t.switch_device("cpu")

    assert t.args.device == "cpu", f"args.device 应为 cpu，实际 {t.args.device}"
    assert str(t.device) == "cpu", f"device 应为 cpu，实际 {t.device}"
    print("正常路径: OK")


def case_model_is_none() -> None:
    """边界路径：model 为 None 时切设备不应崩溃。"""
    t = make_trainer("cpu")
    assert t.model is None, "初始 model 应为 None"

    try:
        t.switch_device("cpu")
    except AttributeError as exc:
        raise AssertionError(
            f"model 为 None 时 switch_device 崩溃了: {exc}"
        ) from None

    print("边界路径（model=None）: OK")


def case_bad_device() -> None:
    """错误路径：设备名非法时应抛 ValueError，且状态不变。"""
    t = make_trainer("cpu")
    t.model = nn.Linear(4, 2)

    try:
        t.switch_device("tpu")
    except ValueError:
        pass
    else:
        raise AssertionError("非法设备名 'tpu' 没有抛 ValueError")

    # 强异常保证：解析失败时，对象状态必须与调用前一致
    assert t.args.device == "cpu", (
        f"解析失败后 args.device 被改成了 {t.args.device!r}，状态已撕裂"
    )
    assert str(t.device) == "cpu", f"解析失败后 device 被改成了 {t.device}"
    print("错误路径（非法设备名）: OK")


def main() -> int:
    case_normal()
    case_model_is_none()
    case_bad_device()
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
