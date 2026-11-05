"""实验配置的单一来源。

为什么要这个模块
----------------
重构前，同一批参数散在三个地方，而且默认值互不相同：

    src/run_exp.py        手写 dict  batch_size=32  num_epochs=30  device="cuda"
    src/train.py          argparse   batch_size=4   num_epochs=10  device="dml"
    src/dataset_factory.py getattr(...) 又各自给了兜底默认值

同一个参数的三个"真相"意味着：改一处、另外两处不管，实验就悄悄跑成别的
东西。更糟的是手写 dict 拼错键名**不会报错** —— Trainer 用 getattr 取不到时
就用兜底值，于是你跑了半小时才发现 batch_size 一直是默认的 4。

本模块的做法
------------
1. 所有默认值只在 ExperimentConfig 里写一次（单一来源）；
2. 用 @dataclass(frozen=True) 而不是 dict：字段名拼错立刻 TypeError，
   配置创建后不可变（Trainer 想改也改不了，只能改它自己的 Namespace 副本）；
3. 用命名注册表把常用组合固化下来，实验之间只差几个字段时用
   dataclasses.replace() 派生，不再复制十几行；
4. to_namespace() 是**适配层**：Trainer 继续按老接口读 args，
   配置侧却可以删掉冗余字段（见下）。

一条设计原则值得记住
--------------------
**配置只放「可以自由选的决策」，不放「能被推导的事实」。**

    batch_size / learning_rate / epochs   -> 决策，放进配置
    in_channels                           -> fact，由数据集声明（SplitBundle）
    use_val_csv                           -> fact，由 val_csv 是否为 None 推导

把可推导的东西写进配置，就等于创造了第二个真相，迟早对不上。

本模块不校验 optimizer / criterion 的取值 —— 那属于 Trainer 的知识，
在配置层再抄一份注册表只会又多一处要同步。配置只校验自己的完整性与自洽性。
"""

import os
from dataclasses import asdict, dataclass, fields, replace
from types import SimpleNamespace
from typing import Callable, Dict, Optional, Tuple

from dataset_factory import available_datasets

# ---------------------------------------------------------------------------
# 路径基准
# ---------------------------------------------------------------------------
# 以本文件位置为基准算绝对路径，这样从任何工作目录启动都能找到数据。
# 重构前用 r"..\data\..." 是相对**当前工作目录**的，只有 cd 到 src/ 才正确。
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MINI_IMAGENET_ROOT = os.path.abspath(
    os.path.join(PROJECT_ROOT, "..", "data", "Mini-ImageNet-Dataset")
)
BARCODE_DIR = os.path.join(PROJECT_ROOT, "dataset")
SAVED_ROOT = os.path.abspath(
    os.path.join(PROJECT_ROOT, "..", "saved_models")
)
r"""训练产物的根目录，放在仓库外（和 data 同级），避免大文件进 git。

必须是绝对路径：Trainer 的 default_save_path() 曾经用 r"..\saved_models"，
从 src/ 启动落到 study\saved_models，从仓库根启动落到 代码\saved_models，
同一份配置换个启动目录就写到两个地方去了。
"""


@dataclass(frozen=True)
class ExperimentConfig:
    """一次实验的完整配置。

    frozen=True 的效果：创建之后任何字段赋值都抛 FrozenInstanceError。
    想改就派生一个新的：

        config = replace(build_config("barcode_mobilenet_128"), num_epochs=20)

    字段分组（顺序即打印顺序）：
        身份 / 数据 / 模型 / 训练 / 产物
    """

    # ---- 身份 ----
    name: str
    """配置名。写进日志和 results.txt，用来回答"这个数字是哪次实验的"。"""

    dataset: str
    """数据集名，必须是 dataset_factory 注册表里的键。"""

    # ---- 数据 ----
    data_root: Optional[str] = None
    """Mini-ImageNet 的图像根目录（会按 split 再拼一层）。
    条码集的路径直接来自 CSV，此项留 None。"""

    train_csv: Optional[str] = None
    val_csv: Optional[str] = None
    test_csv: Optional[str] = None
    """三份切分 CSV。值为 None 表示不加载该 split。

    注意这里**没有** use_train_csv 之类的开关：是否使用完全由路径是否为
    None 推导，结构上不可能出现"开关是 False 但路径还在"的矛盾。
    Trainer 需要的老字段在 to_namespace() 里补出来。"""

    input_size: Tuple[int, int] = (224, 224)
    """网络输入尺寸 (H, W)。必须与 checkpoint 对应的训练尺寸一致。"""

    keep_grayscale: bool = True
    """条码集是否保持单通道。真正决定模型通道数的是数据集（SplitBundle），
    这里只是传给数据集的一个构建选项。"""

    # ---- 模型 ----
    model_name: str = "Resnet-50"

    # ---- 训练 ----
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3
    optimizer: str = "Adam"
    criterion: str = "CrossEntropyLoss"
    seed: int = 0

    device: str = "cpu"
    """本机实测：DirectML 比 CPU 慢（MobileNetV3@128：0.66s/批 vs 0.22s/批），
    且 Resnet-50@224 直接失败。所以默认 cpu。"""

    num_workers: int = 8
    """DataLoader 进程数。0 表示主进程加载（调试时用，避免子进程吃掉报错栈）。"""

    # ---- 产物 ----
    save_model: bool = True
    """是否保存**按验证损失选出的最佳轮**权重（model_epoch_<N>.pth）。"""

    save_last: bool = False
    """是否额外保存**最后一轮**的权重（固定文件名 model_last.pth，每轮覆盖）。

    默认关闭，因为大多数实验只需要最佳轮那份；需要"训练结束时的模型"
    （继续训练、观察收敛终点）时打开。与 save_model 互相独立：
    save_model=False + save_last=True 也能只留最后一轮。
    """

    save_path: Optional[str] = None
    """None 表示由 Trainer 生成 saved_models/<dataset>/<model>/<时间戳>。
    测试里显式给一个临时目录，避免污染 saved_models。"""

    # -- 校验 ---------------------------------------------------------------

    def validate(self, check_files: bool = True):
        """校验配置自身的完整性与自洽性，返回 self 以便链式调用。

        只做"配置层能知道的事"，不去复制 Trainer 的合法值表。

        异常:
            ValueError: 任一约束不满足，信息里带上具体字段与原因。
        """
        if self.dataset not in DATASET_REGISTRY_NAMES:
            raise ValueError(
                f"未知数据集 {self.dataset!r}，可选：{DATASET_REGISTRY_NAMES}"
            )

        if len(self.input_size) != 2:
            raise ValueError(
                f"input_size 必须是 (H, W)，实际收到 {self.input_size!r}"
            )
        if any(int(side) <= 0 for side in self.input_size):
            raise ValueError(f"input_size 必须为正，实际 {self.input_size!r}")

        if self.train_csv is None:
            raise ValueError(
                f"配置 {self.name} 没有 train_csv：没有训练集就无法确定类别映射"
            )

        for field_name in ("num_epochs", "batch_size", "num_workers"):
            value = getattr(self, field_name)
            if int(value) < 0 or (
                field_name != "num_workers" and int(value) == 0
            ):
                raise ValueError(f"{field_name} 取值不合法：{value}")

        if not str(self.device).strip():
            raise ValueError("device 不能为空")

        if check_files:
            for field_name in ("train_csv", "val_csv", "test_csv"):
                path = getattr(self, field_name)
                if path is not None and not os.path.isfile(path):
                    raise FileNotFoundError(
                        f"{field_name} 不存在：{os.path.abspath(path)}"
                    )
        return self

    # -- 适配层 -------------------------------------------------------------

    def to_namespace(self) -> SimpleNamespace:
        """转换成 Trainer 认识的老式 Namespace。

        这一层是"适配器"：对外（Trainer）保持兼容，对内（配置）保持干净。
        两个 use_*_csv 字段在这里由路径推导出来，配置侧不用维护。

        为什么不直接让 Trainer 用 dataclass：
        配置需要**不可变**（防止跑一半被改），训练过程需要**可变**的运行状态
        （save_path、best_model_path 等）。两者职责不同，用一层转换隔开比
        混在一个对象里清楚。
        """
        values = asdict(self)
        values["use_train_csv"] = self.train_csv is not None
        values["use_val_csv"] = self.val_csv is not None
        values["use_test_csv"] = self.test_csv is not None
        return SimpleNamespace(**values)

    # -- 展示 ---------------------------------------------------------------

    def describe(self) -> str:
        """返回对齐的配置表，直接贴进实验记录或报告。"""
        lines = []
        width = max(len(field.name) for field in fields(self))
        for field in fields(self):
            lines.append(
                f"  {field.name:<{width}} : {getattr(self, field.name)}"
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        return (
            f"ExperimentConfig({self.name}, {self.dataset}, "
            f"{self.model_name})"
        )


# 只是为了让 validate() 的报错信息不必每次现算
DATASET_REGISTRY_NAMES = available_datasets()


# ---------------------------------------------------------------------------
# 命名配置
# ---------------------------------------------------------------------------
# 每个配置是一个工厂函数。共用部分用 _mini_imagenet() / _barcode() 作为
# 基础模板，再 replace() 出变体 —— 避免十几行 × N 份复制粘贴。

def _mini_imagenet(name: str, **overrides) -> ExperimentConfig:
    """Mini-ImageNet 基础模板（100 类、RGB、三通道）。"""
    base = dict(
        name=name,
        dataset="mini_imagenet",
        data_root=MINI_IMAGENET_ROOT,
        train_csv=os.path.join(MINI_IMAGENET_ROOT, "train.csv"),
        val_csv=os.path.join(MINI_IMAGENET_ROOT, "val.csv"),
        test_csv=os.path.join(MINI_IMAGENET_ROOT, "test.csv"),
        input_size=(224, 224),
        keep_grayscale=False,
        num_epochs=30,
        batch_size=32,
        learning_rate=1e-3,
        seed=0,
        device="cpu",
    )
    base.update(overrides)
    return ExperimentConfig(**base)


def _barcode(name: str, **overrides) -> ExperimentConfig:
    """条码集基础模板（2 类、灰度、按源图键分组切分）。"""
    base = dict(
        name=name,
        dataset="barcode",
        data_root=None,
        train_csv=os.path.join(BARCODE_DIR, "barcode_train.csv"),
        val_csv=os.path.join(BARCODE_DIR, "barcode_val.csv"),
        test_csv=os.path.join(BARCODE_DIR, "barcode_test.csv"),
        input_size=(224, 224),
        keep_grayscale=True,
        num_epochs=10,
        batch_size=32,
        learning_rate=1e-3,
        seed=0,
        device="cpu",
    )
    base.update(overrides)
    return ExperimentConfig(**base)


def _make_mini_imagenet_resnet50() -> ExperimentConfig:
    """Mini-ImageNet + ResNet-50：论文里的对标配置。"""
    return _mini_imagenet("mini_imagenet_resnet50", model_name="Resnet-50")


def _make_mini_imagenet_mobilenet() -> ExperimentConfig:
    """Mini-ImageNet + MobileNetV3：轻量模型对照。"""
    return _mini_imagenet(
        "mini_imagenet_mobilenet", model_name="MobileNetV3-Large"
    )


def _make_mini_imagenet_convnext() -> ExperimentConfig:
    """Mini-ImageNet + ConvNeXt：现代化架构对照。"""
    return _mini_imagenet(
        "mini_imagenet_convnext", model_name="ConvNeXt-Tiny"
    )


def _make_barcode_mobilenet_128() -> ExperimentConfig:
    """条码集快速迭代配置：MobileNetV3 + 128，一轮约 30 秒。

    本机 CPU 实测：0.222 秒/批 × 143 批 ≈ 0.5 分钟/epoch。
    做 A/B 实验时用这个，跑满 10 轮约 5 分钟。
    """
    return _barcode(
        "barcode_mobilenet_128",
        model_name="MobileNetV3-Large",
        input_size=(128, 128),
    )


def _make_barcode_resnet224() -> ExperimentConfig:
    """条码集最终配置：ResNet-50 + 224，一轮约 6 分钟。"""
    return _barcode(
        "barcode_resnet224",
        model_name="Resnet-50",
        input_size=(224, 224),
    )


def _make_barcode_mobilenet_224() -> ExperimentConfig:
    """条码集 224 配置：MobileNetV3 + 224。

    128 输入的最终特征图只有 4×4，而条码靠条/空宽度比例区分，需要更细的
    空间分辨率。实测 10 轮：test 98.88% → 99.39%，错误 11 → 6
    （见 docs/条码实验记录.md）。
    """
    return _barcode(
        "barcode_mobilenet_224",
        model_name="MobileNetV3-Large",
        input_size=(224, 224),
    )


def _make_barcode_mobilenet_224_s16() -> ExperimentConfig:
    """条码集 224 + 累计步长 16：把分类头前的特征图从 7x7 提到 14x14。

    与 barcode_mobilenet_224 只差模型名（MobileNetV3-Large-S16），
    后者只把最深的 stride-2 卷积改回 1。参数完全不变，MACs 1.62 倍。
    用途：判断"网络分辨率"是否还是瓶颈——若 14x14 不再涨点，说明瓶颈已经
    转移到输入信息量（图像本身太糊/太小），不该继续堆分辨率。
    """
    return _barcode(
        "barcode_mobilenet_224_s16",
        model_name="MobileNetV3-Large-S16",
        input_size=(224, 224),
    )


def _make_barcode_mobilenet_224_focal() -> ExperimentConfig:
    """条码集 224 + Focal Loss：小类加权的对照配置。

    动机来自错误审计：11 个错误里 9 个是 C128（小类）被漏判，而 C128
    的样本数只有 EAN13 的 1/3.3。Focal Loss 的 (1-p_t)^gamma 把易分样本
    的梯度压下去，alpha 按逆频率给少数类加权。
    """
    return _barcode(
        "barcode_mobilenet_224_focal",
        model_name="MobileNetV3-Large",
        input_size=(224, 224),
        criterion="focal",
    )


def _make_barcode_mobilenet_224_label_smoothing() -> ExperimentConfig:
    """条码集 224 + Label Smoothing α=0.1：抑制过拟合的对照配置。

    基线在末两轮 val loss 从 0.022 涨到 0.145，是过拟合的直接证据。
    """
    return _barcode(
        "barcode_mobilenet_224_label_smoothing",
        model_name="MobileNetV3-Large",
        input_size=(224, 224),
        criterion="label_smoothing",
    )


def _make_barcode_smoke() -> ExperimentConfig:
    """冒烟配置：1 轮、小批，专门验证"整条链路通不通"，不看精度。"""
    return _barcode(
        "barcode_smoke",
        model_name="MobileNetV3-Large",
        input_size=(128, 128),
        num_epochs=1,
        batch_size=8,
        num_workers=0,
    )


# 配置名 -> 工厂函数。新增实验只需在这里加一行。
CONFIGS: Dict[str, Callable[[], ExperimentConfig]] = {
    "mini_imagenet_resnet50": _make_mini_imagenet_resnet50,
    "mini_imagenet_mobilenet": _make_mini_imagenet_mobilenet,
    "mini_imagenet_convnext": _make_mini_imagenet_convnext,
    "barcode_mobilenet_128": _make_barcode_mobilenet_128,
    "barcode_mobilenet_224": _make_barcode_mobilenet_224,
    "barcode_mobilenet_224_s16": _make_barcode_mobilenet_224_s16,
    "barcode_mobilenet_224_focal": _make_barcode_mobilenet_224_focal,
    "barcode_mobilenet_224_label_smoothing": (
        _make_barcode_mobilenet_224_label_smoothing
    ),
    "barcode_resnet224": _make_barcode_resnet224,
    "barcode_smoke": _make_barcode_smoke,
}

DEFAULT_CONFIG = "barcode_smoke"
"""命令行不带参数时用哪一个。

选 smoke 而不是 mini_imagenet_resnet50：不带参数就启动一次 1~2 小时的训练
是不友好的默认值，而"先验证链路通不通"是任何人第一步都想做的事。"""


def available_configs() -> list:
    """返回所有配置名（按注册顺序）。"""
    return list(CONFIGS.keys())


def build_config(name: str, **overrides) -> ExperimentConfig:
    """按名字取配置，并应用覆盖项。

    参数:
        name: CONFIGS 的键。
        **overrides: 想改的字段。**拼错字段名会抛 TypeError**（由
            dataclasses.replace 保证），而不是静默忽略 —— 这正是用
            dataclass 而不是 dict 的核心收益。

    返回:
        校验过的 ExperimentConfig。
    """
    if name not in CONFIGS:
        raise ValueError(
            f"未知配置 {name!r}，可选：{available_configs()}"
        )
    config = CONFIGS[name]()
    if overrides:
        config = replace(config, **overrides)
    return config.validate()


def config_table(names=None) -> str:
    """把多个配置的关键字段排成表，用于实验前核对。"""
    names = names or available_configs()
    headers = (
        "配置", "数据集", "模型", "输入", "通道", "epoch", "batch", "lr", "设备"
    )
    rows = []
    for name in names:
        config = CONFIGS[name]()
        rows.append(
            (
                config.name,
                config.dataset,
                config.model_name,
                f"{config.input_size[0]}x{config.input_size[1]}",
                "1(灰度)" if config.keep_grayscale else "3",
                str(config.num_epochs),
                str(config.batch_size),
                f"{config.learning_rate:g}",
                config.device,
            )
        )
    widths = [
        max(len(str(headers[i])), *(len(str(row[i])) for row in rows))
        for i in range(len(headers))
    ]
    line = "  ".join(
        str(headers[i]).ljust(widths[i]) for i in range(len(headers))
    )
    output = [line, "-" * len(line)]
    for row in rows:
        output.append(
            "  ".join(
                str(row[i]).ljust(widths[i]) for i in range(len(headers))
            )
        )
    return "\n".join(output)
