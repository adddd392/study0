"""checkpoint 模块的行为检查。

覆盖三条路径：
    1. 路径约定：文件名与子目录结构符合训练端的写入约定
    2. 往返一致：保存后再加载，张量逐元素相同
    3. 安全性：weights_only=True 生效，含非张量对象时加载被拒绝

临时目录建在工作区内（.tmp_tests/），因为系统临时目录在本机沙箱下
创建/清理会触发 WinError 5 权限错误。

运行::

    python tests/test_checkpoint.py
"""

import contextlib
import os
import shutil
import sys
import uuid

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import checkpoint  # noqa: E402

# 工作区内的临时目录根，已在 .gitignore 中排除
_TMP_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".tmp_tests")
)


@contextlib.contextmanager
def temp_dir():
    """在工作区内创建临时目录，退出时清理。

    不使用 tempfile.mkdtemp：它建出的目录权限过严（0o700），
    本机沙箱下无法在其中再创建子目录（WinError 5）。
    改用普通 mkdir，权限按默认继承。
    """
    os.makedirs(_TMP_ROOT, exist_ok=True)
    path = os.path.join(_TMP_ROOT, uuid.uuid4().hex)
    os.makedirs(path, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class ArbitraryPayload:
    """仅用于验证 weights_only=True 会拒绝非张量对象。"""

    def __init__(self, value):
        self.value = value


def case_path_convention() -> None:
    """路径约定：<save_dir>/models/model_epoch_<epoch>.pth"""
    with temp_dir() as tmp:
        path = checkpoint.checkpoint_path(tmp, 7)
        expected = os.path.join(tmp, "models", "model_epoch_7.pth")
        assert path == expected, f"路径约定不符: {path} != {expected}"
    print("路径约定: OK")


def case_roundtrip() -> None:
    """保存后再加载，权重必须逐元素相同。"""
    with temp_dir() as tmp:
        model = nn.Linear(3, 2)
        with torch.no_grad():
            model.weight.fill_(1.5)
            model.bias.fill_(-2.0)

        path = checkpoint.save_checkpoint(model, tmp, 3)
        assert os.path.isfile(path), f"权重文件未生成: {path}"

        state = checkpoint.load_checkpoint(path)
        assert isinstance(state, dict), f"期望 dict，得到 {type(state)}"
        assert set(state) == {"weight", "bias"}, f"键不符: {sorted(state)}"
        assert torch.equal(state["weight"], model.weight), "weight 往返不一致"
        assert torch.equal(state["bias"], model.bias), "bias 往返不一致"
    print("保存/加载往返: OK")


def case_weights_only() -> None:
    """含非张量对象时必须被拒绝，证明 weights_only=True 真的生效。"""
    with temp_dir() as tmp:
        path = os.path.join(tmp, "with_object.pth")
        torch.save(
            {"weight": torch.zeros(2, 2), "payload": ArbitraryPayload(0.1)},
            path,
        )

        try:
            state = checkpoint.load_checkpoint(path)
        except Exception as exc:  # noqa: BLE001 - 具体异常类型随 torch 版本变化
            print(f"weights_only 拒绝非张量: OK  ({type(exc).__name__})")
            return

        raise AssertionError(
            f"weights_only=True 未生效，加载出了: {sorted(state)}"
        )


def main() -> int:
    case_path_convention()
    case_roundtrip()
    case_weights_only()
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
