"""单张图像推理测试脚本,支持 Mini-ImageNet 与条码数据集。

输出图片的 Top-K 预测,每行同时给出**原始 logit**与 softmax 概率。

关键原则:推理的**预处理**和**标签映射**必须与训练同源,否则预测结果
不可解释。所以这里不自己拼预处理,而是通过 dataset_factory 按数据集分派:

    build_eval_preprocessor(dataset_name, cfg)  -> image_path -> tensor
    build_label_map(dataset_name, cfg)          -> {类别名: 下标}

提示:条码任务的 Top-K 会被压到类别数(2 类),此时看 Top-1 即可。

一键运行(所有参数都有默认值,在仓库根目录 study 下执行即可):
    python src\\inference.py
    python src\\inference.py --dataset barcode --image <条码图路径>
"""

import argparse
import csv
import os
import sys

# albumentations 会在 import 时联网检查版本更新,断网时会一直阻塞到超时,这里关掉
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
# 无显示环境也能读图(matplotlib 的 imread 不需要交互后端)
os.environ.setdefault("MPLBACKEND", "Agg")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import checkpoint  # noqa: E402
from dataset_factory import (  # noqa: E402
    available_datasets,
    build_eval_preprocessor,
    build_label_map,
)
from models import first_conv_weight_key, get_model  # noqa: E402

# Windows 控制台默认 GBK、VSCode 终端是 UTF-8,不固定编码中文会乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------- 默认路径 ----------------
# 以本文件所在位置(src/)为基准解析,因此从任何目录启动都能找到数据,不依赖 cd
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.abspath(
    os.path.join(PROJECT_ROOT, "..", "data", "Mini-ImageNet-Dataset")
)
BARCODE_DIR = os.path.join(PROJECT_ROOT, "dataset")
SAVE_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "saved_models"))
DEFAULT_MODEL = "Resnet-50"
DEFAULT_DATASET = "mini_imagenet"
DEFAULT_IMAGE = os.path.join(
    DATA_ROOT, "val", "n03544143", "n0354414300000023.jpg"
)

# 各架构分类头(最后一个 Linear)的权重名,用于从 checkpoint 推断类别数
CLASSIFIER_WEIGHT = {
    "Resnet-50": "fc.weight",
    "MobileNetV3-Large": "classifier.3.weight",
    "ConvNeXt-Tiny": "classifier.2.weight",
}


def _latest_checkpoint(run_root):
    """在某个训练结果目录下取最近一次的权重。

    目录名是时间戳,按数值排序取最近一次;再读 results.txt 里记录的
    "Best Validation Epoch" 定位最优权重,取不到就退化为任意一个 .pth。
    """
    # 目录名是时间戳,按数值排序取最近一次
    runs = [
        d for d in os.listdir(run_root)
        if os.path.isdir(os.path.join(run_root, d))
    ]
    if not runs:
        raise FileNotFoundError(f"{os.path.abspath(run_root)} 下没有任何训练记录")
    latest = os.path.join(
        run_root, sorted(runs, key=lambda d: int(d) if d.isdigit() else 0)[-1]
    )

    # results.txt 里记录了验证损失最小的轮次
    best_epoch = None
    results_path = os.path.join(latest, "results.txt")
    if os.path.isfile(results_path):
        with open(results_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("Best Validation Epoch:"):
                    best_epoch = line.split(":")[-1].strip()
                    break

    if best_epoch and best_epoch != "None":
        candidate = checkpoint.checkpoint_path(latest, best_epoch)
        if os.path.isfile(candidate):
            return candidate

    # 兜底:该目录下任意一个 .pth
    model_dir = os.path.join(latest, "models")
    checkpoints = (
        [
            os.path.join(model_dir, name)
            for name in sorted(os.listdir(model_dir))
            if name.endswith(".pth")
        ]
        if os.path.isdir(model_dir)
        else []
    )
    if not checkpoints:
        raise FileNotFoundError(f"{os.path.abspath(model_dir)} 下没有 .pth 权重")
    return checkpoints[-1]


def find_default_checkpoint(dataset_name, model_name):
    """自动定位默认权重,优先新布局 saved_models/<dataset>/<model>/。"""
    candidates = [
        # 新布局:实验产物按数据集分目录,避免条码模型和 ImageNet 模型混在一起
        os.path.join(SAVE_ROOT, dataset_name, model_name),
        # 旧布局:兼容改造前训练出的结果
        os.path.join(SAVE_ROOT, model_name),
    ]
    for run_root in candidates:
        if os.path.isdir(run_root):
            return _latest_checkpoint(run_root)
    raise FileNotFoundError(
        f"找不到训练结果目录,尝试过: {[os.path.abspath(c) for c in candidates]}"
    )


def first_image_of_csv(csv_path):
    """取 CSV 第一行的 image_path,用于条码集的默认示例图。"""
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("image_path"):
                return row["image_path"]
    raise FileNotFoundError(f"CSV 里没有 image_path 列或没有数据: {csv_path}")


def resolve_paths(args):
    """按数据集补齐默认的 CSV 与示例图路径。"""
    if args.dataset == "barcode":
        args.train_csv = args.train_csv or os.path.join(
            BARCODE_DIR, "barcode_train.csv"
        )
        args.val_csv = args.val_csv or os.path.join(
            BARCODE_DIR, "barcode_val.csv"
        )
        if args.image is None:
            args.image = first_image_of_csv(args.val_csv)
    else:
        args.train_csv = args.train_csv or os.path.join(DATA_ROOT, "train.csv")
        args.val_csv = args.val_csv or os.path.join(DATA_ROOT, "val.csv")
        if args.image is None:
            args.image = DEFAULT_IMAGE
    return args


def parse_args(argv=None):
    """命令行参数。默认值全部指向可直接运行的状态。"""
    parser = argparse.ArgumentParser(
        description="单张图像推理(不带参数即用 Mini-ImageNet 默认路径一键运行)"
    )
    parser.add_argument(
        "--model_name", default=DEFAULT_MODEL, choices=list(CLASSIFIER_WEIGHT)
    )
    parser.add_argument(
        "--dataset", default=DEFAULT_DATASET, choices=available_datasets(),
        help="数据集名,决定用哪套预处理与标签映射",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="默认自动选 saved_models/<dataset>/<model_name> 下最近一次的权重",
    )
    parser.add_argument("--image", default=None, help="待推理的单张图片路径")
    parser.add_argument(
        "--train_csv", default=None,
        help="训练集 CSV,用于重建 label->index 映射(必须用 train.csv)",
    )
    parser.add_argument(
        "--val_csv", default=None,
        help="验证集 CSV,仅用于条码集挑默认示例图",
    )
    parser.add_argument(
        "--keep_grayscale", type=bool, default=True,
        help="条码集是否按灰度处理,必须与训练时一致",
    )
    parser.add_argument(
        "--input_size", type=int, nargs=2, default=[224, 224],
        metavar=("H", "W"), help="必须与训练时一致",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"]
    )
    parser.add_argument("--topk", type=int, default=5)
    return parser.parse_args(argv)


def run(args):
    """执行一次单张推理,返回 (预测下标列表, 概率列表)。"""
    args = resolve_paths(args)

    # 0) 没给权重就自动定位
    if args.checkpoint is None:
        args.checkpoint = find_default_checkpoint(
            args.dataset, args.model_name
        )

    # 1) 标签映射:按数据集分派,必须与训练同一来源
    label_to_index = build_label_map(args.dataset, args)
    index_to_label = {i: name for name, i in label_to_index.items()}

    # 2) 设备
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    # 3) 类别数与输入通道数都从 checkpoint 推断,不靠配置猜
    state_dict = checkpoint.load_checkpoint(args.checkpoint)
    num_classes = int(state_dict[CLASSIFIER_WEIGHT[args.model_name]].shape[0])
    in_channels = int(
        state_dict[first_conv_weight_key(args.model_name)].shape[1]
    )
    model = get_model(args.model_name, num_classes, in_channels=in_channels)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    # 4) 预处理:按数据集分派,与训练完全同源
    preprocess = build_eval_preprocessor(args.dataset, args)
    tensor = preprocess(args.image)

    # 5) 前向,按 logit 取 Top-K(logit 与 softmax 单调等价,排序结果相同)
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0).to(device)).float()[0].cpu()
    k = min(args.topk, num_classes)
    topk_logit, topk_index = torch.topk(logits, k)
    topk_prob = F.softmax(logits, dim=0)[topk_index]

    print(
        f"数据集 {args.dataset} | 模型 {args.model_name} | "
        f"设备 {device} | 类别数 {num_classes} | 输入通道 {in_channels}"
    )
    print(f"权重 {os.path.abspath(args.checkpoint)}")
    print(f"图片 {os.path.abspath(args.image)}")
    for rank, (logit, prob, index) in enumerate(
        zip(topk_logit.tolist(), topk_prob.tolist(), topk_index.tolist()),
        start=1,
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
        print(
            f"\n真值 {truth} -> "
            + (f"Top-{hits[0]} 命中" if hits else "Top-K 未命中")
        )
    return topk_index.tolist(), topk_prob.tolist()


def main():
    """命令行入口。"""
    run(parse_args())


if __name__ == "__main__":
    main()
