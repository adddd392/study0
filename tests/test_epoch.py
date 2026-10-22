"""trainer 的 epoch 级计算方法验收测试。

覆盖三个公开方法::

    loss, top1_acc, top5_acc = trainer.train_one_epoch()   # 训练一个 epoch
    loss, top1_acc, top5_acc = trainer.validate()          # 验证集评估
    loss, top1_acc, top5_acc = trainer.test()              # 测试集评估

契约:
- 三者都返回 (平均损失, Top-1 正确率(%), Top-5 正确率(%))；
- 平均损失按**样本数**加权，不是按批平均的均值
  （末批不满 batch_size 时两者不相等）；
- train_one_epoch 期间 model 处于 training 模式、梯度开启，
  并执行 optimizer.step() 更新参数；
- validate / test 期间 model 处于 eval 模式、梯度关闭，参数不变。

本测试用假模型（forward 直接返回输入 × 可学习标量）和最小可迭代对象，
不加载真实数据集，也不启用 DataLoader 的 worker 进程。
"""

import argparse
import math
import os
import sys

import torch
import torch.nn as nn

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from train import trainer  # noqa: E402

# 6 个样本、6 个类别。刻意让 Top-1 / Top-5 都有错有对。
LOGITS = torch.tensor(
    [
        [9.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [9.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [9.0, 8.0, 7.0, 6.0, 5.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 9.0],
        [0.0, 0.0, 0.0, 0.0, 9.0, 0.0],
        [0.0, 0.0, 9.0, 0.0, 0.0, 0.0],
    ]
)
LABELS = torch.tensor([0, 1, 5, 3, 4, 2])

# 测试集用另一份数据，便于证明 test() 读的是 test_loader 而非 val_loader。
LOGITS_TEST = torch.tensor(
    [
        [9.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 9.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 9.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 9.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 9.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 9.0],
    ]
)
LABELS_TEST = torch.tensor([0, 1, 2, 3, 4, 5])

BATCH_SIZE = 4  # 6 个样本 -> 两批 4 + 2，末批不满，用来检验加权方式


class FakeModel(nn.Module):
    """forward 返回 输入 * 可学习标量。

    把 logits 直接写进样本里（样本就是 logits 本身），这样不需要真的卷积，
    但仍保留一个可学习参数，供 optimizer.step() 作用。
    同时记录每次 forward 时的模型模式与梯度开关。
    """

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.training_seen = []
        self.grad_seen = []

    def forward(self, x):
        self.training_seen.append(self.training)
        self.grad_seen.append(torch.is_grad_enabled())
        return x * self.scale


class ListLoader:
    """最小可迭代批数据源，避免 DataLoader 的 worker 进程。"""

    def __init__(self, samples, labels, batch_size):
        self.samples = samples
        self.labels = labels
        self.batch_size = batch_size

    def __iter__(self):
        for start in range(0, len(self.labels), self.batch_size):
            end = start + self.batch_size
            yield self.samples[start:end], self.labels[start:end]

    def __len__(self):
        return math.ceil(len(self.labels) / self.batch_size)


def manual_metrics(logits, labels, topk=5):
    """不依赖 torch 的交叉熵/指标 API，手算 (平均损失, Top-1, Top-5)。

    用双精度算，避免与被测实现的 float32 累加顺序差异混在一起。
    """
    logits = logits.double()
    labels = labels.long()

    # log_softmax 手算：x - max - log(sum(exp(x - max)))
    shifted = logits - logits.max(dim=1, keepdim=True).values
    log_softmax = shifted - shifted.exp().sum(dim=1, keepdim=True).log()

    nll = -log_softmax.gather(1, labels.view(-1, 1)).squeeze(1)
    loss = nll.mean().item()

    top1 = (logits.argmax(dim=1) == labels).sum().item()
    order = logits.argsort(dim=1, descending=True)[:, :topk]
    hit = (order == labels.view(-1, 1)).any(dim=1).sum().item()

    total = labels.numel()
    return loss, 100.0 * top1 / total, 100.0 * hit / total


def make_trainer(**loaders):
    """构造一个只装了计算所需属性的 trainer，不走 dataset_init。"""
    obj = trainer(argparse.Namespace(device="cpu"))
    obj.device = torch.device("cpu")
    obj.model = FakeModel()
    obj.criterion = nn.CrossEntropyLoss()
    obj.optimizer = torch.optim.SGD(obj.model.parameters(), lr=1e-4)
    for name, loader in loaders.items():
        setattr(obj, name, loader)
    return obj


def case_validate_metrics() -> None:
    """validate 的三个指标必须与人手计算一致，且不更新参数。"""
    obj = make_trainer(
        val_loader=ListLoader(LOGITS, LABELS, BATCH_SIZE),
    )
    loss, top1, top5 = obj.validate()

    exp_loss, exp_top1, exp_top5 = manual_metrics(LOGITS, LABELS)

    assert abs(loss - exp_loss) < 1e-4, f"损失期望 {exp_loss}，实际 {loss}"
    assert abs(top1 - exp_top1) < 1e-6, f"Top-1 期望 {exp_top1}，实际 {top1}"
    assert abs(top5 - exp_top5) < 1e-6, f"Top-5 期望 {exp_top5}，实际 {top5}"

    # eval 模式、梯度关闭
    assert all(flag is False for flag in obj.model.training_seen), (
        "validate 期间 model 必须处于 eval 模式"
    )
    assert all(flag is False for flag in obj.model.grad_seen), (
        "validate 期间必须关闭梯度"
    )
    # 参数未被更新
    assert obj.model.scale.item() == 1.0, "validate 不应修改模型参数"
    print("validate (指标精确 + eval/no_grad): OK")


def case_loss_weighted_by_samples() -> None:
    """平均损失必须按样本数加权；本用例证明该断言确实有区分能力。"""
    batch_means = []
    for start in range(0, len(LABELS), BATCH_SIZE):
        end = start + BATCH_SIZE
        batch_loss, _, _ = manual_metrics(LOGITS[start:end], LABELS[start:end])
        batch_means.append(batch_loss)

    naive = sum(batch_means) / len(batch_means)  # 按批平均
    weighted = sum(
        m * len(LABELS[start:start + BATCH_SIZE])
        for m, start in zip(batch_means, range(0, len(LABELS), BATCH_SIZE))
    ) / len(LABELS)

    assert abs(naive - weighted) > 1e-3, (
        "用例设计失效：两种平均方式结果太接近，无法区分实现是否加权"
    )

    actual, _, _ = make_trainer(
        val_loader=ListLoader(LOGITS, LABELS, BATCH_SIZE),
    ).validate()
    assert abs(actual - weighted) < 1e-4, (
        f"损失应按样本数加权得 {weighted}，按批平均会得 {naive}，实际 {actual}"
    )
    print("validate (损失按样本数加权): OK")


def case_test_uses_own_loader() -> None:
    """test() 必须读 test_loader（数据与 val 不同，指标应不同）。"""
    obj = make_trainer(
        val_loader=ListLoader(LOGITS, LABELS, BATCH_SIZE),
        test_loader=ListLoader(LOGITS_TEST, LABELS_TEST, BATCH_SIZE),
    )
    loss, top1, top5 = obj.test()

    exp_loss, exp_top1, exp_top5 = manual_metrics(LOGITS_TEST, LABELS_TEST)
    assert abs(loss - exp_loss) < 1e-4, f"损失期望 {exp_loss}，实际 {loss}"
    assert abs(top1 - exp_top1) < 1e-6, f"Top-1 期望 {exp_top1}，实际 {top1}"
    assert abs(top5 - exp_top5) < 1e-6, f"Top-5 期望 {exp_top5}，实际 {top5}"

    # 这份测试数据全对，与 val 的 50% / 83.3% 明显不同，能区分用错了 loader
    assert top1 == 100.0 and top5 == 100.0, "测试集全对，指标应为 100%"

    assert all(flag is False for flag in obj.model.training_seen), (
        "test 期间 model 必须处于 eval 模式"
    )
    assert all(flag is False for flag in obj.model.grad_seen), (
        "test 期间必须关闭梯度"
    )
    print("test (读取 test_loader + eval/no_grad): OK")


def case_train_updates_parameters() -> None:
    """train_one_epoch 要开梯度、更新参数，且指标仍按同一套公式计算。"""
    obj = make_trainer(
        train_loader=ListLoader(LOGITS, LABELS, BATCH_SIZE),
    )
    loss, top1, top5 = obj.train_one_epoch()

    assert all(flag is True for flag in obj.model.training_seen), (
        "train_one_epoch 期间 model 必须处于 training 模式"
    )
    assert all(flag is True for flag in obj.model.grad_seen), (
        "train_one_epoch 期间必须开启梯度"
    )

    # 参数被 optimizer.step() 更新过
    assert obj.model.scale.item() != 1.0, (
        "训练后参数未变化，说明未执行 backward/step"
    )
    assert obj.model.scale.grad is not None, "参数没有梯度，说明未执行 backward"

    # 正数缩放不改变 argmax，故 Top-1/Top-5 仍应与手算一致
    exp_loss, exp_top1, exp_top5 = manual_metrics(LOGITS, LABELS)
    assert abs(top1 - exp_top1) < 1e-6, f"Top-1 期望 {exp_top1}，实际 {top1}"
    assert abs(top5 - exp_top5) < 1e-6, f"Top-5 期望 {exp_top5}，实际 {top5}"

    # 参数在训练中变化，损失只会轻微偏移，用较宽的容差
    assert abs(loss - exp_loss) < 0.1, (
        f"训练损失 {loss} 与初始参数下的 {exp_loss} 相差过大"
    )
    print("train_one_epoch (training 模式 + 参数更新 + 指标): OK")


def main() -> int:
    case_validate_metrics()
    case_loss_weighted_by_samples()
    case_test_uses_own_loader()
    case_train_updates_parameters()
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
