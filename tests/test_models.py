"""models.py 的黄金测试。

锁定两件事：
    1. 分类头输出类数正确
    2. 主干网络总参数量不变（结构指纹）

参数量若变化，说明网络结构被改动过，必须停下来查。

运行::

    python tests/test_models.py
"""

import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from models import available_models, get_model  # noqa: E402

# 黄金数据：num_classes=10 下测得，改动 models.py 后必须完全一致
EXPECTED_PARAMS = {
    "Resnet-50": 23528522,
    "MobileNetV3-Large": 4214842,
    "ConvNeXt-Tiny": 27827818,
}
NUM_CLASSES = 10


def main() -> int:
    for name in available_models():
        model = get_model(name, NUM_CLASSES)
        header = model.fc if hasattr(model, "fc") else model.classifier[-1]

        assert header.out_features == NUM_CLASSES, (
            f"{name} 分类头不是 {NUM_CLASSES} 类"
        )

        total = sum(p.numel() for p in model.parameters())
        expected = EXPECTED_PARAMS[name]
        assert total == expected, (
            f"{name} 参数量变了: {total} != {expected}"
        )

        print(f"{name}: OK  params={total:,}  head={header.out_features}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
