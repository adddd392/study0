import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import Mini_ImageNet_Datasets
from models import get_model
import profiling
import checkpoint
import seeding
import device_utils
import os
import argparse
from tqdm import tqdm

import matplotlib.pyplot as plt
import time
import gc

import numpy as np


class trainer:
    def __init__(self, args):
        self.args = args
        self.device = device_utils.resolve_device(args.device)
        self.model = None
        # self.model_init()
        # self.criterion_init()
        # self.optimizer_init()
        # self.dataset_init()

    def switch_model(self, new_model_name):
        if self.device == "dml":
            self.model = None
            gc.collect()
            import torch_directml
            torch_directml.empty_cache()
        self.args.model_name = new_model_name
        self.args.save_path = os.path.join(r"..\saved_models", self.args.model_name, str(int(time.time())))

    def switch_criterion(self, new_criterion):
        self.args.criterion = new_criterion
        self.criterion_init()
        self.args.save_path = os.path.join(r"..\saved_models", self.args.model_name, str(int(time.time())))

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

    def dataset_init(self):
        label_to_index_map = None
        if self.args.use_train_csv:
            self.train_dataset = Mini_ImageNet_Datasets(csv_file=self.args.train_csv,
                                                        data_root=os.path.join(self.args.data_root, "train"),
                                                        inputsize=self.args.input_size,
                                                        label_to_index_map=label_to_index_map)
            label_to_index_map = self.train_dataset.get_map()
            self.train_loader = DataLoader(self.train_dataset,
                                           batch_size=self.args.batch_size,
                                           shuffle=True,
                                            num_workers=8,
                                            pin_memory=True,
                                            persistent_workers=True,
                                                                                   )
        if self.args.use_val_csv:
            self.val_dataset = Mini_ImageNet_Datasets(csv_file=self.args.val_csv,
                                                      data_root=os.path.join(self.args.data_root, "val"),
                                                      inputsize=self.args.input_size,
                                                      label_to_index_map=label_to_index_map)
            label_to_index_map = self.val_dataset.get_map()
            self.val_loader = DataLoader(self.val_dataset,
                                         batch_size=self.args.batch_size,
                                         shuffle=False,
                                         num_workers=8,
                                        pin_memory=True,
                                        persistent_workers=True,
                                        )
        if self.args.use_test_csv:
            self.test_dataset = Mini_ImageNet_Datasets(csv_file=self.args.test_csv,
                                                       data_root=os.path.join(self.args.data_root, "test"),
                                                       inputsize=self.args.input_size,
                                                       label_to_index_map=label_to_index_map)
            label_to_index_map = self.test_dataset.get_map()
            self.test_loader = DataLoader(self.test_dataset,
                                          batch_size=self.args.batch_size,
                                          shuffle=False,
                                        num_workers=8,
                                        pin_memory=True,
                                        persistent_workers=True,
                                         )
        return len(label_to_index_map)

    def optimizer_init(self):
        if self.args.optimizer == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        else:
            raise ValueError("Wrong optimizer name")

    def model_init(self, num_classes):
        self.model = get_model(self.args.model_name, num_classes)
        self.model.to(self.device)
        self.optimizer_init()
        self.criterion_init()
        self.args.save_path = os.path.join(r"..\saved_models", self.args.model_name, str(int(time.time())))

    def criterion_init(self):
        if self.args.criterion == "CrossEntropyLoss":
            self.criterion = nn.CrossEntropyLoss()
        elif self.args.criterion == "label_smoothing":
            self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        else:
            raise ValueError("Wrong criterion name")

    def train_one_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        Top1_correct = 0
        Top5_correct = 0
        total = 0
        for images, labels in tqdm(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item() * images.size(0)
            _, Top1_predicted = torch.max(outputs, 1)
            total += labels.size(0)
            Top1_correct += (Top1_predicted == labels).sum().item()

            _, Top5_predicted = torch.topk(outputs, 5, dim=1)
            Top5_correct += (Top5_predicted.eq(labels.view(-1, 1))).sum().item()
        epoch_Top1_acc = 100. * Top1_correct / total
        epoch_Top5_acc = 100. * Top5_correct / total
        epoch_loss /= total
        print(f"Epoch Loss: {epoch_loss:.4f}, Top-1 Accuracy: {epoch_Top1_acc:.2f}%")
        print(f"Top-5 Accuracy: {epoch_Top5_acc:.2f}%")
        return epoch_loss, epoch_Top1_acc, epoch_Top5_acc

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        Top1_correct = 0
        Top5_correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, Top1_predicted = torch.max(outputs, 1)
                total += labels.size(0)
                Top1_correct += (Top1_predicted == labels).sum().item()

                _, Top5_predicted = torch.topk(outputs, 5, dim=1)
                Top5_correct += (Top5_predicted.eq(labels.view(-1, 1))).sum().item()
        val_Top1_acc = 100. * Top1_correct / total
        val_Top5_acc = 100. * Top5_correct / total
        val_loss /= total
        print(f"Validation Loss: {val_loss:.4f}, Top-1 Accuracy: {val_Top1_acc:.2f}%")
        print(f"Top-5 Accuracy: {val_Top5_acc:.2f}%")
        return val_loss, val_Top1_acc, val_Top5_acc

    def test(self):
        self.model.eval()
        test_loss = 0.0
        Top1_correct = 0
        Top5_correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)
                _, Top1_predicted = torch.max(outputs, 1)
                total += labels.size(0)
                Top1_correct += (Top1_predicted == labels).sum().item()

                _, Top5_predicted = torch.topk(outputs, 5, dim=1)
                Top5_correct += (Top5_predicted.eq(labels.view(-1, 1))).sum().item()
        test_Top1_acc = 100. * Top1_correct / total
        test_Top5_acc = 100. * Top5_correct / total
        test_loss /= total
        print(f"Test Loss: {test_loss:.4f}, Top-1 Accuracy: {test_Top1_acc:.2f}%")
        print(f"Top-5 Accuracy: {test_Top5_acc:.2f}%")
        return test_loss, test_Top1_acc, test_Top5_acc

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
            if self.args.save_model and self.args.use_val_csv and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_epoch = epoch + 1
                self.best_model_path = self.save_model(epoch + 1)
        return training_losses, training_Top1_accuracies, training_Top5_accuracies, validation_losses, validation_Top1_accuracies, validation_Top5_accuracies, best_val_epoch, best_val_loss

    def run_test(self):
        if not self.args.use_test_csv:
            print("Test CSV not provided.")
            return None, None, None
        test_loss, test_Top1_acc, test_Top5_acc = self.test()
        return test_loss, test_Top1_acc, test_Top5_acc

    def draw_loss_acc_curve(self, training_losses, training_Top1_accuracies, training_Top5_accuracies, validation_losses=None, validation_Top1_accuracies=None, validation_Top5_accuracies=None):
        epochs = range(1, self.args.num_epochs + 1)
        fig, ax = plt.subplots()
        plt.plot(epochs, training_losses, label="Training Loss")
        if validation_losses is not None:
            plt.plot(epochs, validation_losses, label="Validation Loss")
        plt.title("Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        # plt.show()
        os.makedirs(os.path.join(self.args.save_path, "curves"), exist_ok=True)
        plt.savefig(os.path.join(self.args.save_path, "curves", "loss_curve.png"))
        # plt.figure()
        plt.close(fig)
        fig, ax = plt.subplots()
        plt.plot(epochs, training_Top1_accuracies, label="Training Top-1 Accuracy")
        plt.plot(epochs, training_Top5_accuracies, label="Training Top-5 Accuracy")

        if validation_Top1_accuracies is not None:
            plt.plot(epochs, validation_Top1_accuracies, label="Validation Top-1 Accuracy")
        if validation_Top5_accuracies is not None:
            plt.plot(epochs, validation_Top5_accuracies, label="Validation Top-5 Accuracy")
        plt.title("Accuracy Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        # plt.show()
        os.makedirs(os.path.join(self.args.save_path, "curves"), exist_ok=True)
        plt.savefig(os.path.join(self.args.save_path, "curves", "accuracy_curve.png"))
        plt.close(fig)

    def run(self):
        self.num_classes = self.dataset_init()
        training_losses, training_Top1_accuracies, training_Top5_accuracies, validation_losses, validation_Top1_accuracies, validation_Top5_accuracies, best_val_epoch, best_val_loss = self.run_train()
        if validation_losses is not None:
            if self.args.use_test_csv:
                best_model_path = getattr(self, "best_model_path", None)
                if best_model_path and os.path.exists(best_model_path):
                    self.load_model(best_model_path)
        test_loss, test_Top1_acc, test_Top5_acc = self.run_test()
        self.draw_loss_acc_curve(training_losses=training_losses,
                                 training_Top1_accuracies=training_Top1_accuracies,
                                 training_Top5_accuracies=training_Top5_accuracies,
                                 validation_losses=validation_losses,
                                 validation_Top1_accuracies=validation_Top1_accuracies,
                                 validation_Top5_accuracies=validation_Top5_accuracies)

        txt_file_path = os.path.join(self.args.save_path, "results.txt")
        os.makedirs(os.path.dirname(txt_file_path), exist_ok=True)
        with open(txt_file_path, "w") as f:

            f.write(f"Best Validation Epoch: {best_val_epoch}\n")
            f.write(f"Best Validation Loss: {best_val_loss:.4f}\n")
            f.write(f"Test Loss: {test_loss:.4f}, Top-1 Accuracy: {test_Top1_acc:.2f}%, Top-5 Accuracy: {test_Top5_acc:.2f}%\n")
            f.write(f"training_losses:{training_losses}\n")
            f.write(f"training_Top1_accuracies:{training_Top1_accuracies}\n")
            f.write(f"training_Top5_accuracies:{training_Top5_accuracies}\n")
            f.write(f"validation_losses:{validation_losses}\n")
            f.write(f"validation_Top1_accuracies:{validation_Top1_accuracies}\n")
            f.write(f"validation_Top5_accuracies:{validation_Top5_accuracies}\n")
            f.write(f"{self.args}\n")
        return training_losses, training_Top1_accuracies, training_Top5_accuracies, validation_losses, validation_Top1_accuracies, validation_Top5_accuracies, test_loss, test_Top1_acc, test_Top5_acc

    def save_model(self, epoch):
        """保存当前模型权重，返回写入的文件路径。"""
        return checkpoint.save_checkpoint(self.model, self.args.save_path, epoch)

    def load_model(self, model_path):
        """加载权重到当前模型，并迁移到当前设备。"""
        self.model.load_state_dict(checkpoint.load_checkpoint(model_path))
        self.model.to(self.device)

    def get_model_macs_and_parameters(self):
        """统计模型乘加次数（MACs）与参数量，并写入报告文件。

        注意：thop.profile 返回的是 macs（乘加次数），不是 FLOPs。
        按惯例 1 mac ≈ 2 FLOPs，需要 FLOPs 时自行乘 2。
        """
        macs, params = profiling.get_model_macs_and_parameters(self.model, (1, 3, *self.args.input_size))
        profiling.save_macs_and_parameters(macs, params, self.args.save_path)
        return macs, params

    def latency_test(self, warming_up_iterations=10, num_iterations=100):
        self.model.eval()
        device = self.device
        is_cuda = device.type == "cuda"

        def sync():
            if is_cuda:
                torch.cuda.synchronize()

        # 固定输入
        input_tensor = torch.randn(
            1, 3, self.args.input_size[0], self.args.input_size[1]
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
        device_str = str(device).replace(":", "_")      # 避免 Windows 非法字符
        txt_file_path = os.path.join(self.args.save_path, "..", str(device_str) + "_latency.txt")
        os.makedirs(os.path.dirname(txt_file_path), exist_ok=True)

        with open(txt_file_path, "w") as f:
            f.write(f"Device          : {device}\n")
            f.write(f"Warmup iters    : {warming_up_iterations}\n")
            f.write(f"Test iters      : {num_iterations}\n")
            f.write(f"Average Latency : {mean_lat:.6f} ms\n")
            f.write(f"Median  Latency : {median_lat:.6f} ms\n")
            f.write(f"Std     Latency : {std_lat:.6f} ms\n")
            f.write(f"P95     Latency : {p95_lat:.6f} ms\n")

        # 返回平均值，保持和你原来接口一致
        return mean_lat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Resnet-50")
    parser.add_argument("--data_root", type=str, default=r"..\data\Mini-ImageNet-Dataset")
    parser.add_argument("--use_train_csv", type=bool, default=True)
    parser.add_argument("--train_csv", type=str, default=r"..\data\Mini-ImageNet-Dataset\train.csv")
    parser.add_argument("--use_val_csv", type=bool, default=True)
    parser.add_argument("--val_csv", type=str, default=r"..\data\Mini-ImageNet-Dataset\val.csv")
    parser.add_argument("--use_test_csv", type=bool, default=True)
    parser.add_argument("--test_csv", type=str, default=r"..\data\Mini-ImageNet-Dataset\test.csv")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="dml")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--criterion", type=str, default="CrossEntropyLoss")
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--save_model", type=bool, default=True)
    parser.add_argument("--save_path", type=str, default=os.path.join(r"..\saved_models", parser.parse_args().model_name, str(int(time.time()))))
    parser.add_argument("--input_size", type=tuple, default=(224, 224))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    trainer_instance = trainer(args)
    trainer_instance.run()
