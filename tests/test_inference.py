"""inference.py 的验收测试。

契约::

    run(args) -> (预测下标列表, 概率列表)

    核心不变量:
    1. 【预处理同源】推理用的预处理必须与训练用的 BarcodeDataset 逐位一致。
       两套预处理只要有一处不同，预测结果就不可解释。
    2. 【映射同源】标签映射来自 fixed_classes / train.csv，与训练同一来源。
    3. 【通道数从权重推断】灰度模型（conv1 是 1 通道）能被正确识别，
       不需要在配置里再写一遍 in_channels。
    4. 【类别数从权重推断】2 类任务的 Top-K 会被压到 2 类，不能越界。

运行::

    python tests/test_inference.py
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

import checkpoint  # noqa: E402
import inference  # noqa: E402
from dataset_factory import (  # noqa: E402
    build_eval_preprocessor,
    build_label_map,
)
from datasets import BarcodeDataset  # noqa: E402
from models import get_model  # noqa: E402

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
BARCODE_DIR = os.path.join(PROJECT_ROOT, "dataset")
TRAIN_CSV = os.path.join(BARCODE_DIR, "barcode_train.csv")
VAL_CSV = os.path.join(BARCODE_DIR, "barcode_val.csv")

# 工作区内的临时目录根，已在 .gitignore 中排除
_TMP_ROOT = os.path.join(PROJECT_ROOT, ".tmp_tests")


@contextlib.contextmanager
def temp_dir():
    """在工作区内创建临时目录，退出时清理（系统 temp 在本机沙箱下不可写）。"""
    os.makedirs(_TMP_ROOT, exist_ok=True)
    path = os.path.join(_TMP_ROOT, uuid.uuid4().hex)
    os.makedirs(path, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def barcode_config(**overrides):
    """推理侧配置：只填 run() 会读到的字段。"""
    base = {
        "dataset": "barcode",
        "model_name": "Resnet-50",
        "checkpoint": None,
        "image": None,
        "train_csv": TRAIN_CSV,
        "val_csv": VAL_CSV,
        "keep_grayscale": True,
        "input_size": (224, 224),
        "device": "cpu",
        "topk": 5,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def case_preprocessor_matches_dataset() -> None:
    """推理预处理必须与数据集取出的张量逐位相同。"""
    dataset = BarcodeDataset(csv_file=VAL_CSV, target_size=(224, 224),
                             verbose=False)
    image_path = dataset._image_paths[0]
    from_dataset, _ = dataset[0]

    preprocess = build_eval_preprocessor("barcode", barcode_config())
    from_inference = preprocess(image_path)

    assert tuple(from_inference.shape) == tuple(from_dataset.shape), (
        f"形状不一致: {tuple(from_inference.shape)} vs "
        f"{tuple(from_dataset.shape)}"
    )
    max_diff = float((from_inference - from_dataset).abs().max())
    assert max_diff == 0.0, (
        f"推理预处理与训练预处理不一致，最大差 {max_diff:.3e}"
    )
    print("推理预处理与数据集同源: OK")


def case_label_map_from_fixed_classes() -> None:
    """条码集的标签映射必须来自 fixed_classes，而不是某个 CSV。"""
    mapping = build_label_map("barcode", barcode_config())
    assert mapping == {"C128": 0, "EAN13": 1}, f"映射错: {mapping}"

    # 小写也要能归一化
    mapping_lower = build_label_map(
        "barcode", barcode_config(fixed_classes=("c128", "ean13"))
    )
    assert mapping_lower == {"C128": 0, "EAN13": 1}, (
        f"未归一化: {mapping_lower}"
    )
    print("条码标签映射: OK")


def case_end_to_end_grayscale_checkpoint() -> None:
    """灰度权重能被识别：通道数、类别数都从 checkpoint 推断出来。"""
    with temp_dir() as tmp:
        model = get_model("Resnet-50", 2, in_channels=1)
        model_path = checkpoint.save_checkpoint(model, tmp, 0)
        assert os.path.isfile(model_path), f"权重未写出: {model_path}"

        args = barcode_config(checkpoint=model_path)
        out = []
        with contextlib.redirect_stdout(_Collector(out)):
            indices, probs = inference.run(args)

        # Top-K 被压到 2 类
        assert len(indices) == 2, f"2 类任务应输出 2 个候选，实际 {len(indices)}"
        assert abs(sum(probs) - 1.0) < 1e-5, f"概率之和应为 1，实际 {sum(probs)}"

        text = "".join(out)
        assert "类别数 2" in text, f"未从权重推断出类别数: {text}"
        assert "输入通道 1" in text, f"未从权重推断出通道数: {text}"
        # 类别名应来自 fixed_classes
        assert "C128" in text or "EAN13" in text, f"未输出类别名: {text}"
    print("端到端（灰度权重推理）: OK")


def case_topk_clamped() -> None:
    """topk 请求超过类别数时不能越界。"""
    with temp_dir() as tmp:
        model = get_model("Resnet-50", 2)
        model_path = checkpoint.save_checkpoint(model, tmp, 0)
        args = barcode_config(
            checkpoint=model_path, keep_grayscale=False, topk=10
        )
        with contextlib.redirect_stdout(_Collector([])):
            indices, _ = inference.run(args)
        assert len(indices) == 2, f"应被压到 2 个候选，实际 {len(indices)}"
    print("topk 超类别数不越界: OK")


class _Collector:
    """把 print 内容收集到列表里，便于断言。"""

    def __init__(self, bucket):
        self.bucket = bucket

    def write(self, text):
        self.bucket.append(text)

    def flush(self):
        pass


def main() -> int:
    case_preprocessor_matches_dataset()
    case_label_map_from_fixed_classes()
    case_end_to_end_grayscale_checkpoint()
    case_topk_clamped()
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
