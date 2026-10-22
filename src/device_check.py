"""device_utils.resolve_device 的行为检查。

覆盖两类路径：
    1. 正常路径：合法设备名返回 torch.device
    2. 错误路径：非法设备名必须抛 ValueError

注意不测 "dml"：当前环境为 CPU 版 torch，torch_directml 未必安装。
"""

import torch

from device_utils import resolve_device


def main() -> int:
    # 1) 正常路径
    dev = resolve_device("cpu")
    assert isinstance(dev, torch.device), f"期望 torch.device，得到 {type(dev)}"
    assert str(dev) == "cpu", f"期望 cpu，得到 {dev}"
    print(f"cpu -> {dev}  OK")

    # 2) 错误路径：非法设备名必须报错
    for bad in ("tpu", "cpuu", ""):
        try:
            resolve_device(bad)
        except ValueError as exc:
            print(f"{bad!r} -> 正确报错: {exc}")
        else:
            raise AssertionError(f"{bad!r} 没有报错，非法设备被静默接受了！")

    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
