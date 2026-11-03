import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_factory import available_datasets, build_split
from configs import (
    DEFAULT_CONFIG,
    SAVED_ROOT,
    available_configs,
    build_config,
)
from models import get_model
import profiling
import checkpoint
import plotting
import seeding
import device_utils
import os
import argparse
from tqdm import tqdm

import time
import gc

import numpy as np


class Trainer:
    def __init__(self, args):
        self.args = args
        # 数据集名决定走哪个 builder；老配置没有这个字段时退回 Mini-ImageNet
        self.dataset_name = getattr(args, "dataset", "mini_imagenet")
        # 输入通道数由训练集声明(dataset_init 里覆盖)，默认 3 兼容 Mini-ImageNet
        self.in_channels = 3
        # 用户显式给了产物目录就尊重它；否则每次 run() 自动生成带时间戳的新目录
        self._explicit_save_path = getattr(args, "save_path", None) is not None
        if not self._explicit_save_path:
            args.save_path = self.default_save_path()
        self.device = device_utils.resolve_device(args.device)
        self.model = None
        # self.model_init()
        # self.criterion_init()
        # self.optimizer_init()
        # self.dataset_init()

    def default_save_path(self):
        """实验产物目录: saved_models/<数据集>/<模型>/<时间戳>（绝对路径）。

        按数据集分层是必须的:条码模型和 ImageNet 模型混在同一个
        saved_models/<模型>/ 下时，推理端的"自动找最近一次权重"会拿到
        另一个数据集的权重，类别数都不一样，还不会报错。

        用 SAVED_ROOT（绝对路径）而不是 r"..\\saved_models"：后者相对当前
        工作目录，从 src/ 启动落到 study\\saved_models，从仓库根启动落到
        代码\\saved_models，同一份配置换个目录启动就写到两个地方。
        """
        return os.path.join(
            SAVED_ROOT,
            self.dataset_name,
            getattr(self.args, "model_name", "unknown"),
            str(int(time.time())),
        )

    def switch_model(self, new_model_name):
        if self.device == "dml":
            self.model = None
            gc.collect()
            import torch_directml
            torch_directml.empty_cache()
        self.args.model_name = new_model_name
        self.args.save_path = self.default_save_path()

    def switch_criterion(self, new_criterion):
        self.args.criterion = new_criterion
        self.criterion_init()
        self.args.save_path = self.default_save_path()

    def _release_device_resources(self):
        """切换设备前释放旧设备占用（仅当旧设备是 DirectML）。"""
        if self.args.device != "dml":
            return
        self.model = None
        gc.collect()
        import torch_directml
        if hasattr(torch_directml, "empty_cache"):
            torch_directml.empty_cache()

    def switch_device(self, new_device_name):
        """切换设备。

        先解析新设备名，解析失败时对象状态保持不变（强异常保证）。
        """
        new_device = device_utils.resolve_device(new_device_name)
        self._release_device_resources()
        self.args.device = new_device_name
        self.device = new_device
        if self.model is not None:
            self.model.to(self.device)

    def _build_split(self, split, label_to_index_map, shuffle):
        """按当前数据集构建一份 split 的 dataset 与 DataLoader。

        数据集的具体类型由 dataset_factory 的注册表决定，这里只负责
        "建 DataLoader" 这一件事，不再出现任何具体数据集类的名字。

        label_to_index_map 传 None 时由该 split 自己建映射（仅限训练集）；
        val/test 必须传入训练集的映射，否则类别下标会与训练错位。
        """
        bundle = build_split(
            self.dataset_name, split, self.args, label_to_index_map
        )
        loader = DataLoader(
            bundle.dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            num_workers=getattr(self.args, "num_workers", 8),
            pin_memory=True,
            persistent_workers=getattr(self.args, "num_workers", 8) > 0,
            # 数据集自己声明的话就挂上（BarcodeDataset 用它关掉 cv2 线程池，
            # 否则每个 worker 里 OpenCV 会和 PyTorch 抢 CPU 核）
            worker_init_fn=getattr(bundle.dataset, "worker_init_fn", None),
        )
        return bundle, loader

    def dataset_init(self):
        """建立 train/val/test 的数据集与 DataLoader，返回类别数。

        三份数据共用同一个类别映射（训练集产生），保证 val/test 的下标语义
        与训练一致。输入通道数也由数据集的 builder 声明，写进 self.in_channels，
        后面 model_init 按它建模型。
        """
        if not self.args.use_train_csv and (
            self.args.use_val_csv or self.args.use_test_csv
        ):
            raise ValueError(
                "必须先加载训练集（use_train_csv=True）才能确定类别映射，"
                "否则 val/test 会自动用各自 csv 建映射，类别下标与训练错位"
            )
        label_to_index_map = None
        if self.args.use_train_csv:
            bundle, self.train_loader = self._build_split(
                split="train",
                label_to_index_map=label_to_index_map,
                shuffle=True
            )
            self.train_dataset = bundle.dataset
            label_to_index_map = bundle.label_to_index_map
            # 通道数跟着数据走：灰度数据集就是 1，模型第一层据此构建
            self.in_channels = bundle.in_channels
        if self.args.use_val_csv:
            bundle, self.val_loader = self._build_split(
                split="val",
                label_to_index_map=label_to_index_map,
                shuffle=False
            )
            self.val_dataset = bundle.dataset
        if self.args.use_test_csv:
            bundle, self.test_loader = self._build_split(
                split="test",
                label_to_index_map=label_to_index_map,
                shuffle=False
            )
            self.test_dataset = bundle.dataset
        return len(label_to_index_map)

    def optimizer_init(self):
        if self.args.optimizer == "Adam":
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.args.learning_rate
            )
        else:
            raise ValueError("Wrong optimizer name")

    def model_init(self, num_classes):
        self.model = get_model(
            self.args.model_name, num_classes, in_channels=self.in_channels
        )
        self.model.to(self.device)
        self.optimizer_init()
        self.criterion_init()
        # 注意：这里不再改写 save_path。原实现每次 model_init 都把它覆盖成
        # 硬编码的 ..\saved_models\<model>\<时间戳>，导致显式指定的 --save_path
        # 在训练开始前被悄悄丢掉。产物目录统一由 run() / switch_* 决定。

    def criterion_init(self):
        if self.args.criterion == "CrossEntropyLoss":
            self.criterion = nn.CrossEntropyLoss()
        elif self.args.criterion == "label_smoothing":
            self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        else:
            raise ValueError("Wrong criterion name")

    def _run_epoch(self, loader, is_train, label):
        """跑一个 epoch，返回 (平均损失, Top-1 正确率, Top-5 正确率)。

        is_train 同时决定三件事：模型处于 train 还是 eval 模式、梯度是否
        开启、是否执行 backward + optimizer.step。评估时在循环外统一关闭
        梯度，避免逐批创建 no_grad 上下文。
        """
        self.model.train(is_train)
        epoch_loss = 0.0
        Top1_correct = 0
        Top5_correct = 0
        total = 0
        with torch.set_grad_enabled(is_train):
            for images, labels in tqdm(loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                if is_train:
                    self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                if is_train:
                    loss.backward()
                    self.optimizer.step()
                epoch_loss += loss.item() * images.size(0)
                _, Top1_predicted = torch.max(outputs, 1)
                total += labels.size(0)
                Top1_correct += (Top1_predicted == labels).sum().item()

                # Top-K 的 K 不能超过类别数：条码任务只有 2 类，
                # torch.topk(outputs, 5) 会直接抛 "selected index k out of
                # range"。这里取 min(5, 类别数)。
                # 注意 K 被压到类别数时，Top-K 恒等于 100%，此时它不是
                # 有效指标，解读 2 类任务请以 Top-1 为准。
                topk_k = min(5, outputs.size(1))
                _, Top5_predicted = torch.topk(outputs, topk_k, dim=1)
                Top5_hit = Top5_predicted.eq(labels.view(-1, 1))
                Top5_correct += Top5_hit.sum().item()
        epoch_Top1_acc = 100. * Top1_correct / total
        epoch_Top5_acc = 100. * Top5_correct / total
        epoch_loss /= total
        print(f"{label} Loss: {epoch_loss:.4f}, "
              f"Top-1 Accuracy: {epoch_Top1_acc:.2f}%")
        print(f"Top-5 Accuracy: {epoch_Top5_acc:.2f}%")
        return epoch_loss, epoch_Top1_acc, epoch_Top5_acc

    def train_one_epoch(self):
        """训练一个 epoch，返回 (训练损失, Top-1 正确率, Top-5 正确率)。"""
        return self._run_epoch(
            self.train_loader, is_train=True, label="Training")

    def validate(self):
        """在验证集上评估，返回 (验证损失, Top-1 正确率, Top-5 正确率)。"""
        return self._run_epoch(
            self.val_loader, is_train=False, label="Validation")

    def test(self):
        """在测试集上评估，返回 (测试损失, Top-1 正确率, Top-5 正确率)。"""
        return self._run_epoch(
            self.test_loader, is_train=False, label="Test")

    def run_train(self):
        if not self.args.use_train_csv:
            print("Train CSV not provided.")
            return None, None, None, None, None, None, None, None
        best_val_loss = float('inf')
        best_val_epoch = None
        training_losses = []
        training_Top1_accuracies = []
        training_Top5_accuracies = []
        validation_losses = []
        validation_Top1_accuracies = []
        validation_Top5_accuracies = []

        seeding.set_seed(self.args.seed)
        self.model_init(self.num_classes)
        for epoch in range(self.args.num_epochs):
            print(f"Epoch{epoch+1}/{self.args.num_epochs}")
            train_loss, train_Top1_acc, train_Top5_acc = self.train_one_epoch()
            training_losses.append(train_loss)
            training_Top1_accuracies.append(train_Top1_acc)
            training_Top5_accuracies.append(train_Top5_acc)
            if self.args.use_val_csv:
                val_loss, val_Top1_acc, val_Top5_acc = self.validate()
                validation_losses.append(val_loss)
                validation_Top1_accuracies.append(val_Top1_acc)
                validation_Top5_accuracies.append(val_Top5_acc)
            if (
                self.args.save_model
                and self.args.use_val_csv
                and val_loss < best_val_loss
            ):
                best_val_loss = val_loss
                best_val_epoch = epoch + 1
                self.best_model_path = self.save_model(epoch + 1)
        return (
            training_losses,
            training_Top1_accuracies,
            training_Top5_accuracies,
            validation_losses,
            validation_Top1_accuracies,
            validation_Top5_accuracies,
            best_val_epoch,
            best_val_loss,
        )

    def run_test(self):
        if not self.args.use_test_csv:
            print("Test CSV not provided.")
            return None, None, None
        test_loss, test_Top1_acc, test_Top5_acc = self.test()
        return test_loss, test_Top1_acc, test_Top5_acc

    def draw_loss_acc_curve(self, training_losses, training_Top1_accuracies,
                            training_Top5_accuracies, validation_losses=None,
                            validation_Top1_accuracies=None,
                            validation_Top5_accuracies=None):
        """绘制损失曲线与准确率曲线。

        两张图分开存到 <save_path>/curves/ 下：loss_curve.png 与
        accuracy_curve.png。分两张是因为 loss（0~几）与 accuracy（0~100）
        量纲不同，画在同一套 y 轴上 loss 会被压成一条直线。
        """
        curves_dir = os.path.join(self.args.save_path, "curves")

        acc_series = [
            (training_Top1_accuracies, "Training Top-1 Accuracy"),
            (training_Top5_accuracies, "Training Top-5 Accuracy"),
        ]
        if validation_Top1_accuracies is not None:
            acc_series.append(
                (validation_Top1_accuracies, "Validation Top-1 Accuracy")
            )
        if validation_Top5_accuracies is not None:
            acc_series.append(
                (validation_Top5_accuracies, "Validation Top-5 Accuracy")
            )
        plotting.save_curve(
            os.path.join(curves_dir, "accuracy_curve.png"),
            "Accuracy Curve",
            "Accuracy (%)",
            acc_series,
        )

        loss_series = [(training_losses, "Training Loss")]
        if validation_losses is not None:
            loss_series.append((validation_losses, "Validation Loss"))
        plotting.save_curve(
            os.path.join(curves_dir, "loss_curve.png"),
            "Loss Curve",
            "Loss",
            loss_series,
        )

    def run(self):
        # 没显式指定产物目录时，每次 run() 都开一个新的时间戳目录，
        # 这样 switch_model / switch_criterion 后的实验不会互相覆盖
        if not self._explicit_save_path:
            self.args.save_path = self.default_save_path()
        self.num_classes = self.dataset_init()
        (
            training_losses,
            training_Top1_accuracies,
            training_Top5_accuracies,
            validation_losses,
            validation_Top1_accuracies,
            validation_Top5_accuracies,
            best_val_epoch,
            best_val_loss,
        ) = self.run_train()
        if validation_losses is not None:
            if self.args.use_test_csv:
                best_model_path = getattr(self, "best_model_path", None)
                if best_model_path and os.path.exists(best_model_path):
                    self.load_model(best_model_path)
        test_loss, test_Top1_acc, test_Top5_acc = self.run_test()
        self.draw_loss_acc_curve(
            training_losses=training_losses,
            training_Top1_accuracies=training_Top1_accuracies,
            training_Top5_accuracies=training_Top5_accuracies,
            validation_losses=validation_losses,
            validation_Top1_accuracies=validation_Top1_accuracies,
            validation_Top5_accuracies=validation_Top5_accuracies,
        )

        txt_file_path = os.path.join(self.args.save_path, "results.txt")
        os.makedirs(os.path.dirname(txt_file_path), exist_ok=True)
        # 必须显式 utf-8：Windows 下默认 cp936，配置记录里有中文路径
        # （本项目仓库路径就带中文）会写成乱码，事后没法核对实验配置
        with open(txt_file_path, "w", encoding="utf-8") as f:

            f.write(f"Best Validation Epoch: {best_val_epoch}\n")
            f.write(f"Best Validation Loss: {best_val_loss:.4f}\n")
            f.write(
                f"Test Loss: {test_loss:.4f}, "
                f"Top-1 Accuracy: {test_Top1_acc:.2f}%, "
                f"Top-5 Accuracy: {test_Top5_acc:.2f}%\n"
            )
            f.write(f"training_losses:{training_losses}\n")
            f.write(f"training_Top1_accuracies:{training_Top1_accuracies}\n")
            f.write(f"training_Top5_accuracies:{training_Top5_accuracies}\n")
            f.write(f"validation_losses:{validation_losses}\n")
            f.write(
                f"validation_Top1_accuracies:{validation_Top1_accuracies}\n"
            )
            f.write(
                f"validation_Top5_accuracies:{validation_Top5_accuracies}\n"
            )
            f.write(f"{self.args}\n")
        return (
            training_losses,
            training_Top1_accuracies,
            training_Top5_accuracies,
            validation_losses,
            validation_Top1_accuracies,
            validation_Top5_accuracies,
            test_loss,
            test_Top1_acc,
            test_Top5_acc,
        )

    def save_model(self, epoch):
        """保存当前模型权重，返回写入的文件路径。"""
        return checkpoint.save_checkpoint(
            self.model, self.args.save_path, epoch
        )

    def load_model(self, model_path):
        """加载权重到当前模型，并迁移到当前设备。"""
        self.model.load_state_dict(checkpoint.load_checkpoint(model_path))
        self.model.to(self.device)

    def get_model_macs_and_parameters(self):
        """统计模型乘加次数（MACs）与参数量，并写入报告文件。

        注意：thop.profile 返回的是 macs（乘加次数），不是 FLOPs。
        按惯例 1 mac ≈ 2 FLOPs，需要 FLOPs 时自行乘 2。
        """
        macs, params = profiling.get_model_macs_and_parameters(
            self.model,
            (1, self.in_channels, *self.args.input_size),
        )
        profiling.save_macs_and_parameters(macs, params, self.args.save_path)
        return macs, params

    def latency_test(self, warming_up_iterations=10, num_iterations=100,
                     save_report=True):
        """测单张推理延迟，返回平均延迟（毫秒）。

        参数:
            save_report: 是否把结果写成 <模型目录>/<设备>_latency.txt。
                纯性能对比（不做训练）时传 False，这样不需要产物目录可写。
        """
        self.model.eval()
        device = self.device
        is_cuda = device.type == "cuda"

        def sync():
            if is_cuda:
                torch.cuda.synchronize()

        # 固定输入。通道数跟模型走（灰度模型是 1），写死 3 会让灰度模型在这里
        # 直接 RuntimeError：weight 是 [16,1,3,3] 而输入是 [1,3,128,128]。
        input_tensor = torch.randn(
            1,
            self.in_channels,
            self.args.input_size[0],
            self.args.input_size[1],
        ).to(device)

        with torch.no_grad():
            # ---------------- warmup ----------------
            for _ in range(warming_up_iterations):
                _ = self.model(input_tensor)
            sync()   # warmup 后同步一次

            # ---------------- 计时 ----------------
            latencies = []
            for _ in range(num_iterations):
                if is_cuda:
                    # GPU 上用 Event 打时间戳
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    _ = self.model(input_tensor)
                    end_event.record()
                    sync()
                    latencies.append(start_event.elapsed_time(end_event))
                else:
                    start = time.perf_counter()
                    _ = self.model(input_tensor)
                    end = time.perf_counter()
                    latencies.append((end - start) * 1000.0)

        latencies = np.array(latencies)
        mean_lat = float(latencies.mean())
        median_lat = float(np.median(latencies))
        std_lat = float(latencies.std())
        p95_lat = float(np.percentile(latencies, 95))

        # ---------------- 保存 ----------------
        if save_report:
            # 注意写的是 save_path 的上一级：同一模型的所有运行共用一个
            # <设备>_latency.txt，后一次会覆盖前一次
            device_str = str(device).replace(":", "_")   # 避免 Windows 非法字符
            txt_file_path = os.path.join(
                self.args.save_path, "..", str(device_str) + "_latency.txt"
            )
            os.makedirs(os.path.dirname(txt_file_path), exist_ok=True)

            with open(txt_file_path, "w", encoding="utf-8") as f:
                f.write(f"Device          : {device}\n")
                f.write(f"Warmup iters    : {warming_up_iterations}\n")
                f.write(f"Test iters      : {num_iterations}\n")
                f.write(f"Average Latency : {mean_lat:.6f} ms\n")
                f.write(f"Median  Latency : {median_lat:.6f} ms\n")
                f.write(f"Std     Latency : {std_lat:.6f} ms\n")
                f.write(f"P95     Latency : {p95_lat:.6f} ms\n")

        # 返回平均值，保持和你原来接口一致
        return mean_lat


def _str2bool(value):
    """argparse 的布尔解析。

    不能直接写 type=bool：bool("False") 是 True（非空字符串都为真），
    于是 --keep_grayscale False 会被当成 True，静默用错通道数。
    """
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in ("1", "true", "yes", "y", "on"):
        return True
    if lowered in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"无法解析为布尔值: {value!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "按命名配置训练。不带参数即跑 "
            f"{DEFAULT_CONFIG}（快速验证链路，不做正式实验）。"
        )
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG, choices=available_configs(),
        help="配置名，定义见 src/configs.py",
    )
    # 下面全部是"覆盖项"：默认 None 表示不覆盖，用配置里的值。
    # 默认值只存在于 configs.py，这里不再复制一份 —— 那是重构前的病根。
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None,
                        choices=available_datasets())
    parser.add_argument("--keep_grayscale", type=_str2bool, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--train_csv", type=str, default=None)
    parser.add_argument("--val_csv", type=str, default=None)
    parser.add_argument("--test_csv", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--criterion", type=str, default=None)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--save_model", type=_str2bool, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--input_size", type=int, nargs=2, default=None,
                        metavar=("H", "W"))
    parser.add_argument("--seed", type=int, default=None)
    # 关掉某个 split：配置里没有 use_*_csv 开关（那是可推导的事实），
    # 所以用显式的 --no_* 表达"确实不要"，而不是靠布尔开关。
    parser.add_argument("--no_val", action="store_true",
                        help="不加载验证集")
    parser.add_argument("--no_test", action="store_true",
                        help="不加载测试集")
    args = parser.parse_args()

    # 只把用户显式给出的参数当作覆盖项（None 表示"没给"）
    overrides = {
        key: value
        for key, value in vars(args).items()
        if key not in ("config", "no_val", "no_test") and value is not None
    }
    if "input_size" in overrides:
        overrides["input_size"] = tuple(overrides["input_size"])
    if args.no_val:
        overrides["val_csv"] = None
    if args.no_test:
        overrides["test_csv"] = None

    config = build_config(args.config, **overrides)
    print(config.describe())
    print()
    Trainer(config.to_namespace()).run()
