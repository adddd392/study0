"""Mini-ImageNet 单张图像推理测试脚本。

输出图片的 Top-5 预测,每行同时给出**原始 logit**与 softmax 概率。

一键运行(所有参数都有默认值,在仓库根目录 study0 下执行即可):
    python src\\inference.py

也可按需覆盖任意一项:
    python src\\inference.py --model_name ConvNeXt-Tiny
    python src\\inference.py --image ..\\data\\Mini-ImageNet-Dataset\\val\\n01532829\\n0153282900000018.jpg
    python src\\inference.py --checkpoint ..\\saved_models\\Resnet-50\\1791900751\\models\\model_epoch_8.pth
"""

import argparse
import os
import sys

# albumentations 会在 import 时联网检查版本更新,断网时会一直阻塞到超时,这里关掉
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import matplotlib

matplotlib.use("Agg")  # 无显示环境也能运行
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import checkpoint
from datasets import build_eval_transform, build_label_to_index_map
from models import get_model

# Windows 控制台默认 GBK、VSCode 终端是 UTF-8,不固定编码中文会乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------- 默认路径 ----------------
# 以本文件所在位置(src/)为基准解析,因此从任何目录启动都能找到数据,不依赖 cd
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ...\study0
DATA_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "data", "Mini-ImageNet-Dataset"))
SAVE_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "saved_models"))
DEFAULT_MODEL = "Resnet-50"
DEFAULT_IMAGE = os.path.join(DATA_ROOT, "val", "n03544143", "n0354414300000023.jpg")

# 各架构分类头(最后一个 Linear)的权重名,用于从 checkpoint 推断类别数
CLASSIFIER_WEIGHT = {
    "Resnet-50": "fc.weight",
    "MobileNetV3-Large": "classifier.3.weight",
    "ConvNeXt-Tiny": "classifier.2.weight",
}


def find_default_checkpoint(model_name):
    """自动定位默认权重。

    在 saved_models/<model_name>/ 下取最近一次训练目录,读取 results.txt 里记录的
    "Best Validation Epoch",返回对应的 .pth;取不到再退化为目录里任意一个权重。
    """
    run_root = os.path.join(SAVE_ROOT, model_name)
    if not os.path.isdir(run_root):
        raise FileNotFoundError(f"找不到训练结果目录: {os.path.abspath(run_root)}")

    # 训练目录名是时间戳,按数值排序取最近一次
    runs = [d for d in os.listdir(run_root) if os.path.isdir(os.path.join(run_root, d))]
    if not runs:
        raise FileNotFoundError(f"{os.path.abspath(run_root)} 下没有任何训练记录")
    latest = os.path.join(run_root, sorted(runs, key=lambda d: int(d) if d.isdigit() else 0)[-1])

    # results.txt 里记录了验证损失最小的轮次
    best_epoch = None
    results_path = os.path.join(latest, "results.txt")
    if os.path.isfile(results_path):
        with open(results_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("Best Validation Epoch:"):
                    best_epoch = line.split(":")[-1].strip()
                    break

    if best_epoch:
        candidate = checkpoint.checkpoint_path(latest, best_epoch)
        if os.path.isfile(candidate):
            return candidate

    # 兜底:该目录下任意一个 .pth
    model_dir = os.path.join(latest, "models")
    checkpoints = (
        [os.path.join(model_dir, f) for f in sorted(os.listdir(model_dir)) if f.endswith(".pth")]
        if os.path.isdir(model_dir)
        else []
    )
    if not checkpoints:
        raise FileNotFoundError(f"{os.path.abspath(model_dir)} 下没有 .pth 权重")
    return checkpoints[-1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mini-ImageNet 单张图像推理测试(不带参数即用默认路径一键运行)"
    )
    parser.add_argument("--model_name", default=DEFAULT_MODEL, choices=list(CLASSIFIER_WEIGHT))
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="默认自动选 saved_models/<model_name> 下最近一次训练的最优权重",
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="待推理的单张图片路径")
    parser.add_argument(
        "--train_csv",
        default=os.path.join(DATA_ROOT, "train.csv"),
        help="训练集 CSV,用于重建 label->index 映射(必须用 train.csv)",
    )
    parser.add_argument("--input_size", type=int, nargs=2, default=[224, 224], metavar=("H", "W"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--topk", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()

    # 0) 没给权重就自动定位
    if args.checkpoint is None:
        args.checkpoint = find_default_checkpoint(args.model_name)

    # 1) 标签映射:用 train.csv 重建,保证下标语义与训练时完全一致
    label_to_index = build_label_to_index_map(args.train_csv)
    index_to_label = {index: label for label, index in label_to_index.items()}

    # 2) 设备
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    # 3) 类别数从 checkpoint 的分类头形状推断,再严格加载权重
    state_dict = checkpoint.load_checkpoint(args.checkpoint)
    num_classes = int(state_dict[CLASSIFIER_WEIGHT[args.model_name]].shape[0])
    model = get_model(args.model_name, num_classes)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    # 4) 预处理与训练完全一致:plt.imread 读成 0~1 float32 -> Resize -> Normalize -> ToTensor
    tensor = build_eval_transform(tuple(args.input_size))(image=plt.imread(args.image))["image"]

    # 5) 前向,按 logit 取 Top-K(logit 与 softmax 概率单调等价,排序结果相同)
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0).to(device)).float()[0].cpu()
    topk_logit, topk_index = torch.topk(logits, min(args.topk, num_classes))
    topk_prob = F.softmax(logits, dim=0)[topk_index]

    print(f"模型 {args.model_name} | 设备 {device} | 类别数 {num_classes}")
    print(f"权重 {os.path.abspath(args.checkpoint)}")
    print(f"图片 {os.path.abspath(args.image)}")
    for rank, (logit, prob, index) in enumerate(
        zip(topk_logit.tolist(), topk_prob.tolist(), topk_index.tolist()), start=1
    ):
        bar = "#" * int(round(prob * 40))
        print(
            f"  {rank}. {index_to_label[index]:<12} "
            f"logit {logit:8.4f}  prob {prob * 100:6.2f}%  {bar}"
        )

    # 6) 若图片父目录名是已知标签,顺带报出是否命中
    truth = os.path.basename(os.path.dirname(os.path.abspath(args.image)))
    if truth in label_to_index:
        hits = [
            rank
            for rank, index in enumerate(topk_index.tolist(), start=1)
            if index == label_to_index[truth]
        ]
        print(f"\n真值 {truth} -> " + (f"Top-{hits[0]} 命中" if hits else "Top-K 未命中"))


if __name__ == "__main__":
    main()
