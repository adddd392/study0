# CNN 图像分类实验平台

一套可复现的图像分类训练/评估/推理代码，支持两个数据集：

| 数据集 | 类别数 | 输入 | 说明 |
|---|---|---|---|
| Mini-ImageNet-100 | 100 | RGB 224×224 | 第一周基准实验（ResNet-50 / MobileNetV3 / ConvNeXt） |
| 条码码制 | 2（C128 / EAN13） | 灰度，128 或 224 | 按源图键分组切分，防数据泄漏 |

设计重点是**单一来源**：配置、类别映射、输入通道数、路径基准各自只有一个定义处，
不在多个文件里重复。数据集的差异通过注册表 + builder 隔离，新增数据集不需要改 Trainer。

## 1. 目录结构

```
study/                      git 仓库
├── src/                    生产代码（14 个模块，扁平一层）
│   ├── configs.py          实验配置注册表 + 路径基准（唯一定义处）
│   ├── dataset_factory.py  数据集注册表 + 统一 builder（SplitBundle）
│   ├── datasets.py         两个数据集类 + 预处理管道
│   ├── models.py           模型注册表 + 首层通道数适配（灰度 ResNet）
│   ├── train.py            Trainer（训练/验证/测试/产物记录）+ 命令行
│   ├── run_exp.py          实验总入口（按配置名批量跑）
│   ├── inference.py        单张推理（预处理与标签映射按数据集分派）
│   ├── split_barcode.py    按源图键分组分层切分条码数据集（独立命令行）
│   └── 基础设施            plotting / checkpoint / profiling /
│                           seeding / device_utils / resnet
├── tests/                  17 个独立测试文件（可执行脚本，非 pytest）
├── dataset/                条码三份切分 CSV（已入库）
├── docs/                   学习笔记、对比文档、周报 PPT
├── .flake8                 PEP8 79 列 + 单进程检查
└── requirements.txt        锁定的依赖版本
```

## 2. 环境

```powershell
# Python 3.10（实测 3.10.20）
pip install -r requirements.txt
```

**所有命令都在仓库根目录 `study/` 下执行**（路径基准基于 `__file__`，不依赖当前目录）。

## 3. 数据与产物

| 位置 | 内容 | 是否入库 |
|---|---|---|
| `..\data\Mini-ImageNet-Dataset\` | ImageNet 图像，按 split/label 分目录 | ❌ 仓库外 |
| `dataset\barber.csv` | 切分前的原始标注（6527 行） | ✅ |
| `dataset\barcode_{train,val,test}.csv` | 条码切分结果（4569/979/979 行） | ✅ |
| `..\saved_models\<数据集>\<模型>\<时间戳>\` | 训练产物 | ❌ 仓库外 |

`barber.csv` 的 `image_path` 列是绝对路径（作者本机 `D:\...`）。换机器后
需要重跑 `split_barcode.py`，或自行改写该列再切分。

单次实验的产物（每次实验一个时间戳目录，互不覆盖）：

```
..\saved_models\barcode\MobileNetV3-Large\
└── 1762230000\                 时间戳目录
    ├── results.txt             指标 + 分类报告 + 完整配置记录（含配置名）
    ├── macs_and_parameters.txt 模型复杂度（benchmark 或显式调用时生成）
    ├── cpu_latency.txt         推理延迟（同上）
    ├── curves\*.png            损失/准确率曲线
    └── models\model_epoch_N.pth
```

`results.txt` 里的分类报告包含混淆矩阵、每类召回/精确率/F1、balanced accuracy
和 macro F1。2 类任务上务必看 balanced accuracy，不要只看总体准确率。

条码切分全部按**源图键**分组：`<源图键>_regionNN.png` 这类来自同一张照片的裁剪
永远落在同一个 split 里。`split_barcode.verify_split()` 会校验三份数据无交叉。

## 4. 快速开始

```powershell
# 看有哪些配置（表格可直接贴报告）
python src\run_exp.py --list

# 跑一个实验（barcode_mobilenet_128，10 轮约 5 分钟）
python src\run_exp.py

# 批量跑多个配置
python src\run_exp.py --configs barcode_mobilenet_128 barcode_resnet224

# 只测延迟/复杂度，不训练、不产生任何文件
python src\run_exp.py --benchmark --configs barcode_smoke --devices cpu

# 单次训练：不带参数 = barcode_smoke（1 轮，验证链路）
python src\train.py
python src\train.py --config barcode_mobilenet_128 --num_epochs 20 --batch_size 64

# 单张推理（必须已有训练产物；不带 --checkpoint 时自动取最近一次）
python src\inference.py
python src\inference.py --dataset barcode --checkpoint <某个 .pth 路径>

# 重新切分条码数据集
python src\split_barcode.py --help
```

## 5. 配置系统

`src/configs.py` 是**默认值的唯一来源**。命令行只做覆盖，不重复写默认值：

```powershell
python src\train.py --config barcode_mobilenet_128 --num_epochs 20
```

| 配置名 | 数据集 | 模型 | 输入 | 通道 | epoch |
|---|---|---|---|---|---|
| `mini_imagenet_resnet50` | mini_imagenet | ResNet-50 | 224 | 3 | 30 |
| `mini_imagenet_mobilenet` | mini_imagenet | MobileNetV3-Large | 224 | 3 | 30 |
| `mini_imagenet_convnext` | mini_imagenet | ConvNeXt-Tiny | 224 | 3 | 30 |
| `barcode_mobilenet_128` | barcode | MobileNetV3-Large | 128 | 1 | 10 |
| `barcode_resnet224` | barcode | ResNet-50 | 224 | 1 | 10 |
| `barcode_smoke` | barcode | MobileNetV3-Large | 128 | 1 | 1 |

三条使用约定：

- **改配置用 `replace()`**，不要就地改字段（配置是 `frozen` 的）：
  `replace(build_config("barcode_smoke"), num_epochs=5)`
- **拼错字段名会报错**：`build_config("x", batchsize=64)` 抛 `TypeError`
- **配置里没有 `in_channels` 和 `use_*_csv`**：前者由数据集声明（`SplitBundle`），
  后者由 CSV 路径是否为 `None` 推导。可推导的事实不进配置。

## 6. 测试

17 个测试文件，都是**可直接运行的脚本**（不是 pytest 用例）：

```powershell
# 跑全部
Get-ChildItem tests\*.py | ForEach-Object { & python $_.FullName }

# 跑单个
python tests\test_configs.py
```

约定：

- 临时文件只写 `<仓库>\.tmp_tests\`（系统临时目录在受限环境下不可写）
- DataLoader 在测试里一律 `num_workers=0`；Windows 上多进程会起命名管道，
  受限环境直接 `PermissionError`
- 改动 `src/` 后必须全量跑一遍，17 个文件是回归网

## 7. 代码约定

- **PEP8，行宽 79**：`python -m flake8` 直接可用（`.flake8` 里已固定
  `jobs = 1`——多文件并行检查会起 multiprocessing 池，在受限环境下崩溃）
- **每个模块、每个函数写清"为什么"**，而不是复述代码做了什么
- **提交信息**：`<type>(<scope>): <摘要>`，`type` ∈
  `feat/fix/refactor/style/test/docs/chore`；一笔提交只做一件事；
  `style:` 类提交必须能证明零行为变化（固定 seed 前后跑同一个配置，数值逐位相同）

## 8. 文档索引

| 文档 | 内容 | 对应代码 |
|---|---|---|
| `docs/note.md` | 每周学习要点总纲 | — |
| `docs/卷积对比.md` | 标准/深度可分离/空洞卷积的参数量与 MACs 推导 | `models.py`，图见 `docs/figures/formulas/conv_*` |
| `docs/激活函数对比.md` | ReLU / GELU / SiLU 公式与梯度 | 各架构内部 |
| `docs/损失函数对比.md` | CE / BCE / Focal / Label Smoothing | `train.Trainer.criterion_init()` |
| `docs/模型架构对比.md` | ResNet → MobileNetV3 → ConvNeXt 演进 | `models.MODEL_BUILDERS` |
| `docs/基准报告.md` | 第一周 Mini-ImageNet-100 基准 | `run_exp.py` 的 `mini_imagenet_*` 配置 |
| `docs/第一周汇报.pptx` | 第一周汇报 | — |
| `docs/第三周汇报.pptx` + `_讲稿.md` | 第三周汇报（含 YOLO 优化） | — |

## 9. 已知限制

1. **2 类任务上 Top-5 没有意义**：`min(5, num_classes)` 会压成 Top-2，恒等于 100%。
   看条码结果请以 `results.txt` 里的分类报告为准（balanced accuracy + 每类召回）。
2. **类别与成像尺寸强相关**（重要）：条码数据集中 C128 的长宽比中位数 0.25、
   EAN13 是 0.50，只用 `log(高/宽)` 一个数值就能在测试集拿到 **90.4%**，
   两个尺寸特征的线性分类器是 **91.2%**（多数类基线 76.9%）。
   任何"高精度"结果都要和这个对照一起看，否则可能只是学了尺寸。
3. **宽高比失真**：现在方形拉伸到 224，C128 纵向放大 3.2×、EAN13 放大 0.95×，
   条宽比例被破坏。改进方向是保比例 + padding 或裁主体，但要同时防第 2 条的捷径。
4. **数据增强未接**：`BarcodeDataset` 的 `transform` 参数已就绪但传的是 `None`。
5. **DirectML 更慢**：本机实测 MobileNetV3@128 是 CPU 的 3 倍慢，ResNet-50@224 直接失败。
   `--device` 默认已改为 `cpu`。
6. **小类召回偏低**：1 轮冒烟的总体准确率 92.34%，但 balanced accuracy 只有
   83.87%，C128 召回 68.14%（226 个里错 72 个）。这正是"只看准确率会漏掉"的问题，
   正式基线要在 10 轮后重新评估。
