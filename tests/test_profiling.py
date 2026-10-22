"""profiling 模块的验收测试。

口径约定：本项目只报告 MACs（乘加次数），不报告 FLOPs。
thop.profile 返回的就是 MACs；按惯例 1 MAC ≈ 2 FLOPs，需要时由使用方换算。

契约::

    get_model_macs_and_parameters(model, input_shape) -> (macs, params)
        model       : 待统计的 nn.Module
        input_shape : 完整输入形状，如 (1, 3, 224, 224)
        设备        : 从 model 的参数推断（next(model.parameters()).device）
        返回        : (macs, params)，两者都是 float

    save_macs_and_parameters(macs, params, save_path) -> str
        把两行报告写入 save_path 的【父目录】下的 macs_and_parameters.txt，
        内容格式严格为（各占一行，末尾有换行）::

            MACs = <macs/1000**3:.4f> G
            Params = <params/1000**2:.4f> M

        返回写入的文件完整路径

命名规范：函数名一律小写加下划线，缩写也不例外。
    macs  -> 正确
    MACs  -> 违反 PEP8（函数名必须全小写）

参考数值（thop 实测，已手算核对）：
    Linear(10, 5)      输入 (1, 10)        -> macs=50.0    params=55.0
    Conv2d(3, 4, 3)    输入 (1, 3, 8, 8)   -> macs=3888.0  params=112.0
        （输出 6x6x4=144 个元素，每个元素 3x3x3=27 次乘加）

运行::

    python tests/test_profiling.py
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

import torch.nn as nn  # noqa: E402

# 契约要求实现的模块
import profiling  # noqa: E402

# 工作区内的临时目录根，已在 .gitignore 中排除（.tmp_*/）
_TMP_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".tmp_tests")
)


@contextlib.contextmanager
def temp_dir():
    """在工作区内创建临时目录，退出时清理。

    不用 tempfile.mkdtemp：它建出的目录权限过严（0o700），
    本机沙箱下无法在其中再创建子目录（WinError 5）。
    """
    os.makedirs(_TMP_ROOT, exist_ok=True)
    path = os.path.join(_TMP_ROOT, uuid.uuid4().hex)
    os.makedirs(path, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def close_to(actual, expected, tol=1e-6):
    return abs(float(actual) - float(expected)) <= tol


def case_measure_linear() -> None:
    """最小可验证模型：Linear(10, 5) 在 (1, 10) 输入下。"""
    macs, params = profiling.get_model_macs_and_parameters(
        nn.Linear(10, 5), (1, 10)
    )
    assert close_to(macs, 50.0), f"macs 期望 50.0，实际 {macs}"
    assert close_to(params, 55.0), f"params 期望 55.0，实际 {params}"
    print("get_model_macs_and_parameters (Linear): OK")


def case_measure_conv() -> None:
    """卷积模型：Conv2d(3, 4, 3) 在 (1, 3, 8, 8) 输入下。"""
    model = nn.Conv2d(3, 4, 3)
    macs, params = profiling.get_model_macs_and_parameters(
        model, (1, 3, 8, 8)
    )
    assert close_to(macs, 3888.0), f"macs 期望 3888.0，实际 {macs}"
    assert close_to(params, 112.0), f"params 期望 112.0，实际 {params}"
    # 与 torch 自身统计交叉核对：thop 的 params 应等于所有参数元素个数
    assert close_to(params, sum(p.numel() for p in model.parameters()))
    print("get_model_macs_and_parameters (Conv2d): OK")


def case_save_report() -> None:
    """报告内容与路径必须精确匹配。"""
    with temp_dir() as tmp:
        save_path = os.path.join(tmp, "run1")
        os.makedirs(save_path, exist_ok=True)

        # 取整数值，让格式化结果无歧义
        macs = 1000 ** 3  # -> 1.0000 G
        params = 1000 ** 2  # -> 1.0000 M

        returned = profiling.save_macs_and_parameters(macs, params, save_path)

        expected_path = os.path.join(tmp, "macs_and_parameters.txt")
        assert os.path.isfile(expected_path), (
            f"报告未写在 {expected_path}，实际返回 {returned}"
            f"；目录内容: {sorted(os.listdir(tmp))}"
        )
        assert isinstance(returned, str), (
            f"save_macs_and_parameters 必须返回路径字符串，实际 {returned!r}"
        )
        assert os.path.abspath(returned) == os.path.abspath(expected_path), (
            f"返回值应为 {expected_path}，实际 {returned}"
        )

        with open(expected_path, "r", encoding="utf-8") as handle:
            text = handle.read()
        expected_text = "MACs = 1.0000 G\nParams = 1.0000 M\n"
        assert text == expected_text, (
            f"报告内容不符\n期望: {expected_text!r}\n实际: {text!r}"
        )
        assert "FLOPs" not in text, "口径错误：报告里不允许出现 FLOPs"
    print("save_macs_and_parameters (路径 + 内容 + 口径): OK")


def case_trainer_wiring() -> None:
    """trainer.get_model_macs_and_parameters() 必须端到端接通。

    run_exp.py 里有 3 处无参调用，所以方法签名不能改；
    原实现返回 (macs, params)，这个返回值也必须保留。
    """
    from train import trainer

    with temp_dir() as tmp:
        save_path = os.path.join(tmp, "run1")
        os.makedirs(save_path, exist_ok=True)

        args = argparse.Namespace(
            device="cpu", input_size=(8, 8), save_path=save_path
        )
        obj = trainer(args)
        obj.model = nn.Sequential(nn.Conv2d(3, 4, 3))

        result = obj.get_model_macs_and_parameters()

        assert isinstance(result, tuple) and len(result) == 2, (
            f"应返回 (macs, params)，实际 {result!r}"
        )
        macs, params = result
        assert macs > 0 and params > 0, (
            f"统计值应大于 0，实际 {(macs, params)}"
        )

        expected_path = os.path.join(tmp, "macs_and_parameters.txt")
        assert os.path.isfile(expected_path), (
            f"trainer 未生成报告 {expected_path}；目录内容: {sorted(os.listdir(tmp))}"
        )
        with open(expected_path, "r", encoding="utf-8") as handle:
            assert "MACs" in handle.read(), "报告口径不是 MACs"
    print("trainer 接线: OK")


def main() -> int:
    case_measure_linear()
    case_measure_conv()
    case_save_report()
    case_trainer_wiring()
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
