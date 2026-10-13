## 第一周：图像分类算法

### 1. 学习与调研要点

**以下每个要点，学习中必须要关注每个点的基本原理、数学计算以及与其他方法的差异（提升或适配某种特殊情况）。（后面几周相同）**

* **基础算子与底层机制**：
  * 卷积变体：
    * 标准卷积
    * 深度可分离卷积（Depthwise Separable Conv）
    * 空洞卷积（Dilated Conv）
    * 可变形卷积（DCNv2/v3）的原理
    * 感受野计算与参数量/计算量推导。

  * 激活函数：ReLU、GELU、SiLU（Swish）的数学公式、梯度曲线差异及其解决的梯度消失/饱和问题。
  * 损失函数：Cross-Entropy、Binary-Cross-Entropy、Label Smoothing Cross-Entropy、Focal Loss（针对极不平衡类别）。



* **模型架构演进**：

  * 经典架构：ResNet（残差连接原理与退化问题推导）。
  * 轻量化架构：MobileNetV3（深度分离卷积 + NAS搜索 + SE模块）、ShuffleNetV2（硬件友好度设计原则）。
  * 现代化与前沿架构：ConvNeXt / ConvNeXt-V2（借鉴 Transformer 的现代化 CNN）、Vision Transformer (ViT)、Swin Transformer（分层滑动窗口注意力）。


### 2. 动手实验任务

* **基准数据集**：CIFAR-100 或 Mini-ImageNet。
* **实验内容**：
  * 搭建统一训练框架，横向对比 3 个模型：ResNet-50、MobileNetV3-Large、ConvNeXt-Tiny。
  * 选定一个模型完成 1 组消融实验：对比标准 Cross-Entropy 与 Label Smoothing `（alpha=0.1）` 对最终 Top-1 准确率及模型过拟合程度的影响。

### 3. 实验记录量化指标

* **精度指标**：Top-1 Accuracy、Top-5 Accuracy。
* **硬件效率**：模型参数量（Params, M）、浮点运算量（FLOPs/MACs, G）、GPU/CPU 单张推理延迟（Latency, ms）。

### 4. 本周交付物清单

1. **源码仓库**：包含数据加载、训练脚本、验证评估脚本及单张图像推理测试脚本。

2. **基准报告**：《图像分类主流模型基准性能对比表》，需包含上述三个模型的精度、参数量、FLOPs 及消融实验 Loss/Acc 曲线图。

3. **周报文档 / PPT**：包含本周学习总结与实验汇总（20 分钟汇报材料）。