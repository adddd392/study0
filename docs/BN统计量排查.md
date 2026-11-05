# BN 统计量导致的评估失真：从 77% 到 96%

> 记录一次"模型没问题、评估有问题"的排查。结论先行，然后是完整证据链，
> 最后是下次遇到同类问题的判据清单。

## 1. 结论

| 项 | 内容 |
|---|---|
| 症状 | 训练准确率正常爬升（93% → 99%），验证/测试准确率恒为 **77.02%**，连续三轮一字不差 |
| 根因 | torchvision 的 MobileNetV3 出厂 BN 动量是 **0.01**（TensorFlow 配方），而本项目每轮只有 **143 步**，running 统计量来不及收敛 |
| 影响面 | eval 模式用的归一化参数里残留 24% 的初始化值，46 层 BN 串联放大 → 分类头退化为常数输出 |
| 修复 | 统一 BN 动量为 **0.1**（`src/models.py` 的 `BN_MOMENTUM`） |
| 效果 | 同配置同 seed 第 1 轮：验证 **77.02% → 96.12%**，测试 96.83% |
| 关键教训 | 训练指标正常不代表评估正常；BN 的 running 统计量在训练时**只写不读**，第一次读就是 eval |

## 2. 症状：三个信号

`barcode_mobilenet_128`（MobileNetV3-Large，灰度 128，batch 32）10 轮训练的原始输出：

```
Epoch1  train 93.04%   val 77.02%
Epoch2  train 97.05%   val 77.02%
Epoch3  train 98.47%   val 77.02%
Epoch4  train 98.91%   val 92.85%
Epoch5  train 98.73%   val 98.26%
```

三个可疑点，任意两个命中就该怀疑评估链路：

| # | 信号 | 本次实例 |
|---|---|---|
| ① | train 与 val 早期背离 >10 个点 | 93.04% vs 77.02% |
| ② | **连续多轮 val 数字完全相同** | 三轮都是 77.02% |
| ③ | val 恰好等于多数类占比 | 754/979 = 77.02% |

**②是最强的信号**：真实训练中验证准确率几乎不可能连续两轮一字不差。

## 3. 定位过程

### 3.1 挂 hook 看统计量

给全部 46 层 BN 挂前向 hook，打印当前 batch 的方差与模型缓冲里的 `running_var`：

| batch 数 | 当前 batch 方差 | running_var |
|---:|---:|---:|
| 1 | 0.0793 | 0.99079 |
| 21 | 0.0839 | 0.82501 |
| 61 | 0.0763 | 0.57900 |
| 101 | 0.0704 | 0.41406 |
| 143 | 0.0804 | 0.29915 |

第 1 步就能看出关系：`0.99 × 1.0 + 0.01 × 0.0793 = 0.99079`，**动量是 0.01 而不是 PyTorch 默认的 0.1**。

### 3.2 递推验证

用 $v_{k+1} = (1-m)v_k + m\sigma^2$ 复现上表：

| 动量 | 第 1 步 running_var（真值 0.079） |
|---|---:|
| m = 0.01 | **0.99079**（与实测一致） |
| m = 0.1 | 0.9079（如果真是 0.1，第 1 步就该是这个数） |

递推到 143 步，与实测**吻合到小数点后 4 位**，确认动量就是 0.01。

### 3.3 换掉统计量再评估

用训练集重算一套"正确"的统计量，套到**同一个检查点**上再评估：

| 权重来源 | 自带统计量 | 重算统计量 |
|---|---:|---:|
| DML 训练的 epoch-1 检查点 | 77.02% | **94.38%** |
| CPU 训练的 epoch-1 检查点 | 77.02% | **95.61%** |

**权重一个字节没变，只换归一化参数，精度差 17 个点。**
这条证据把"模型不行"彻底排除，问题锁定在评估环节。

### 3.4 排除设备因素

同一配置换 CPU 跑，同样是 77.02%。

> 这一步很重要：最初我把 77% 误判为"DML 后端算错"，花了半小时排查硬件。
> 后来对 DML 的算子逐项验证（Adam 更新差 9.3e-10、BN 累积差 3.6e-07、
> eval 前向 6.0e-08），全部与 CPU 等价，才确认与设备无关。

## 4. 根因：配方与预算不匹配

```python
# torchvision/models/mobilenetv3.py:154
norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
```

这是 TensorFlow 版 MobileNetV3 的配方（TF 的 `momentum=0.99` ≡ PyTorch 的
`momentum=0.01`）。它假设训练步数足够多，而本项目的预算远小于这个假设：

| 配置 | batch | 每轮步数 | 初值残留 $0.99^{步数}$ |
|---|---:|---:|---:|
| Mini-ImageNet | 32 | 38400/32 = 1200 | $5.8\times10^{-6}$ |
| `barcode_smoke` | 8 | 4569/8 = 571 | 0.32% |
| `barcode_mobilenet_128` | 32 | 4569/32 = 143 | **23.8%** |

要让初值残留降到 1% 以下，需要

$$\text{步数} > \frac{\ln 0.01}{\ln 0.99} \approx 458$$

这解释了全部历史现象：

| 现象 | 解释 |
|---|---|
| 第一周 Mini-ImageNet 基准数字正常 | 1200 步/轮，第 1 轮就收敛 |
| `barcode_smoke` 报出 92.34% 没暴露问题 | batch 8 → 571 步，刚好越过 458 的阈值 |
| 正式基线崩塌 | batch 32 → 143 步，远低于阈值 |

## 5. 修复

### 5.1 统一动量（`src/models.py`）

```python
BN_MOMENTUM = 0.1

def set_bn_momentum(model, momentum=BN_MOMENTUM):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = momentum
    return model
```

在 `get_model()` 返回前统一调用。**只改 momentum**，不动 `eps`（保持
torchvision 的 0.001）、`affine`、`track_running_stats`：一次只动一个变量。

改完后 143 步的初值残留是 $0.9^{143}\approx3\times10^{-6}$，第 1 轮即收敛。

### 5.2 运行时告警（`src/train.py`）

```python
def validation_looks_frozen(previous_accuracies, current_accuracy,
                            tolerance=1e-9):
    if not previous_accuracies:
        return False
    return abs(current_accuracy - previous_accuracies[-1]) <= tolerance
```

命中就在训练日志里打印警告。这次事故中连续三轮回 77.02% 却没人发现，
整条曲线被误读成"模型前几轮完全没学会"。

### 5.3 防复发测试（`tests/test_models.py`）

| 用例 | 断言 |
|---|---|
| `case_bn_momentum_unified` | 所有含 BN 的模型动量必须是 0.1（ConvNeXt 用 LayerNorm，跳过） |
| `case_bn_momentum_helper` | `set_bn_momentum` 只改动量，其它归一化参数不变 |

## 6. 验证

同配置、同 seed、同样 1 轮：

| 指标 | 修复前 | 修复后（CPU） | 修复后（DML） |
|---|---:|---:|---:|
| train Top-1 | 93.04% | 93.06% | 93.04% |
| **val Top-1** | **77.02%** | **96.12%** | **96.73%** |
| test Top-1 | 崩溃无产物 | 96.83% | 97.65% |
| balanced accuracy | — | 93.45% | 96.15% |
| C128 召回 | — | 87.17% | 93.36% |

**训练指标几乎没变，只有 eval 指标跳了 19 个点**——这本身就证明修的是评估而不是模型。

## 7. 判据清单（下次怎么快速识别）

| 检查 | 阈值 | 说明 |
|---|---|---|
| train 与 val 早期背离 | >10 个点 | 先查评估链路，别急着调模型 |
| 连续两轮 val 完全相同 | 任何情况 | 已由 `validation_looks_frozen` 自动告警 |
| val 是否等于多数类占比 | 是则高度可疑 | 表明模型在输出先验 |
| 检查点 `running_var` / 重算值 | >1.2 | 统计量未收敛 |
| 当前任务每轮步数 | <458 时警惕 | 动量 0.01 在该预算下无法收敛 |

## 8. 遗留

- 修复前的历史实验（10 轮 val 99.59% / test 97.45%）是错误动量下跑的，
  **轨迹与当前不可比**，已作废重跑。
- 该 bug 从项目第一次提交起就存在（`models.py` 的模型构建行从未被改动），
  不是重构引入的；它只在"每轮步数 < 458"的任务上显形。
