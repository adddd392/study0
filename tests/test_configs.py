"""configs.py 的验收测试。

契约::

    config = build_config("barcode_mobilenet_128")
    trainer = Trainer(config.to_namespace())

    1. 【单一来源】所有默认值只写在 ExperimentConfig 里；配置能定位到
       数据集 CSV，且路径是绝对路径（不受启动目录影响）。
    2. 【拼错就报错】build_config(batchsize=...) 必须抛 TypeError，
       而不是像 dict 那样静默忽略——这是用 dataclass 的核心收益。
    3. 【不可变】配置创建后赋值必须抛 FrozenInstanceError。
    4. 【适配层】to_namespace() 补出 Trainer 需要的 use_*_csv，
       由对应 CSV 是否为 None 推导，结构上不可能自相矛盾。
    5. 【校验】非法 input_size / epoch / 数据集名 / 不存在的 CSV
       必须在训练开始前就报错。
    6. 【端到端】配置 → Namespace → Trainer → dataset_init 能跑通，
       且 in_channels 仍然由数据集声明（配置里不写这个字段）。

运行::

    python tests/test_configs.py
"""

import dataclasses
import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from configs import (  # noqa: E402
    DEFAULT_CONFIG,
    ExperimentConfig,
    available_configs,
    build_config,
    config_table,
)
from train import Trainer  # noqa: E402


def case_registry() -> None:
    """注册表里的配置都能构建，未知名字报错并列出候选。"""
    names = available_configs()
    assert names, "配置注册表为空"
    assert DEFAULT_CONFIG in names, (
        f"默认配置 {DEFAULT_CONFIG} 不在注册表里"
    )

    for name in names:
        config = build_config(name)
        assert config.name == name, f"{name} 的 name 字段不一致: {config.name}"

    try:
        build_config("不存在的配置")
    except ValueError as exc:
        assert "未知配置" in str(exc), f"报错信息不明确: {exc}"
        assert "barcode_smoke" in str(exc), "报错里应列出可选配置"
    else:
        raise AssertionError("未知配置名应抛 ValueError")
    print("注册表: OK")


def case_paths_are_absolute() -> None:
    """数据路径必须是绝对路径，否则换个工作目录就找不到数据。"""
    for name in ("barcode_mobilenet_128", "mini_imagenet_resnet50"):
        config = build_config(name)
        for field_name in ("train_csv", "val_csv", "test_csv"):
            path = getattr(config, field_name)
            assert os.path.isabs(path), (
                f"{name}.{field_name} 不是绝对路径: {path}"
            )
            assert os.path.isfile(path), f"{name}.{field_name} 不存在: {path}"

    # data_root 只有 Mini-ImageNet 用得上
    assert build_config("barcode_smoke").data_root is None, (
        "条码集的路径来自 CSV，data_root 必须为 None"
    )
    print("路径为绝对路径: OK")


def case_no_derived_fields_in_config() -> None:
    """配置里不能出现可推导的字段（in_channels / use_*_csv）。"""
    field_names = {
        field.name for field in dataclasses.fields(ExperimentConfig)
    }
    for forbidden in ("in_channels", "use_train_csv", "use_val_csv",
                      "use_test_csv"):
        assert forbidden not in field_names, (
            f"{forbidden} 是可推导的，不该出现在配置字段里"
        )
    print("配置不含可推导字段: OK")


def case_to_namespace_adapts() -> None:
    """适配层：use_*_csv 由路径推导，两边永远一致。"""
    config = build_config("barcode_smoke")
    namespace = config.to_namespace()
    assert namespace.use_train_csv is True
    assert namespace.use_val_csv is True
    assert namespace.use_test_csv is True
    # 配置名必须带过去：results.txt 里记录的是 Namespace，
    # 没有 name 就答不上"这个数字是哪次实验的"
    assert namespace.name == config.name, "Namespace 丢了配置名"

    # 去掉验证集 -> 开关必须跟着变 False，不可能自相矛盾
    without_val = dataclasses.replace(config, val_csv=None)
    namespace = without_val.to_namespace()
    assert namespace.use_val_csv is False, (
        "val_csv 为 None 时 use_val_csv 必须为 False"
    )
    assert namespace.train_csv == config.train_csv, "其余字段不该受影响"
    print("适配层推导 use_*_csv: OK")


def case_frozen() -> None:
    """配置创建后不可变。"""
    config = build_config("barcode_smoke")
    try:
        config.batch_size = 64
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("frozen 配置应禁止赋值")
    print("配置不可变: OK")


def case_typo_rejected() -> None:
    """拼错字段名必须立刻报错，而不是静默忽略。"""
    try:
        build_config("barcode_smoke", batchsize=64)
    except TypeError as exc:
        assert "batchsize" in str(exc), f"报错没指出拼错的字段: {exc}"
    else:
        raise AssertionError("拼错字段名应抛 TypeError")
    print("拼错字段名被拒绝: OK")


def case_validate_rejects_bad_values() -> None:
    """各类非法取值都要在训练前报错。"""
    cases = []

    # input_size 不是二元组
    cases.append(
        ("input_size=(224,)", lambda: build_config(
            "barcode_smoke", input_size=(224,))
         )
    )
    # input_size 有非正数
    cases.append(
        ("input_size=(0, 128)", lambda: build_config(
            "barcode_smoke", input_size=(0, 128))
         )
    )
    # epoch 为 0
    cases.append(
        ("num_epochs=0", lambda: build_config("barcode_smoke", num_epochs=0))
    )
    # 未知数据集
    cases.append(
        ("dataset='不存在'", lambda: build_config(
            "barcode_smoke", dataset="不存在")
         )
    )
    # 设备为空
    cases.append(
        ("device=''", lambda: build_config("barcode_smoke", device=""))
    )

    for label, build in cases:
        try:
            build()
        except (ValueError, TypeError) as exc:
            assert str(exc), f"{label} 的报错信息为空"
        else:
            raise AssertionError(f"{label} 应该被 validate 拦下")
    print("非法取值被拦下: OK")


def case_missing_csv_rejected() -> None:
    """指向不存在的 CSV 要报 FileNotFoundError，不能等到训练中途才炸。"""
    try:
        build_config(
            "barcode_smoke", train_csv=os.path.join(_SRC, "不存在的.csv")
        )
    except FileNotFoundError as exc:
        assert "train_csv" in str(exc), f"报错没指出字段: {exc}"
    else:
        raise AssertionError("不存在的 CSV 应抛 FileNotFoundError")

    # 关掉文件检查时可以构造出来（用于只在内存里描述配置的场景）
    config = ExperimentConfig(
        name="dry", dataset="barcode", train_csv="whatever.csv"
    )
    config.validate(check_files=False)
    print("不存在的 CSV 被拒绝: OK")


def case_trainer_integration() -> None:
    """配置 -> Namespace -> Trainer 全链路，通道数仍由数据集群决定。"""
    config = build_config("barcode_smoke")
    trainer = Trainer(config.to_namespace())

    num_classes = trainer.dataset_init()
    assert num_classes == 2, f"类别数期望 2，实际 {num_classes}"
    assert trainer.in_channels == 1, (
        f"in_channels 应来自数据集（1），实际 {trainer.in_channels}；"
        "若这里是 3，说明配置里的通道数被当成了真相来源"
    )
    assert trainer.dataset_name == "barcode", (
        f"Trainer 未识别数据集: {trainer.dataset_name}"
    )
    print("配置 -> Trainer 集成: OK")


def case_default_save_path_absolute() -> None:
    """自动生成的产物目录必须是绝对路径，不能随启动目录漂移。"""
    config = build_config("barcode_smoke")
    trainer = Trainer(config.to_namespace())
    save_path = trainer.args.save_path
    assert os.path.isabs(save_path), (
        f"save_path 应为绝对路径，实际 {save_path!r}；"
        "相对路径会让同一份配置换个启动目录就写到两个地方"
    )
    assert "barcode" in save_path, f"产物目录应含数据集名: {save_path}"
    assert "MobileNetV3-Large" in save_path, f"产物目录应含模型名: {save_path}"

    # 显式指定时必须原样保留，不能被自动生成覆盖
    explicit = os.path.join(_SRC, "..", ".tmp_tests", "explicit")
    trainer = Trainer(
        build_config("barcode_smoke", save_path=explicit).to_namespace()
    )
    assert trainer.args.save_path == explicit, (
        f"显式 save_path 被改写了: {trainer.args.save_path!r}"
    )
    print("产物目录为绝对路径: OK")


def case_describe_and_table() -> None:
    """describe() 与 config_table() 供实验记录/报告直接使用。"""
    config = build_config("barcode_mobilenet_128")
    text = config.describe()
    for key in ("name", "dataset", "model_name", "input_size", "device"):
        assert key in text, f"describe() 缺少字段 {key}"

    table = config_table(["barcode_mobilenet_128", "barcode_resnet224"])
    assert "128x128" in table, "配置表里应有输入尺寸"
    assert "224x224" in table, "配置表里应有输入尺寸"
    assert "barcode_mobilenet_128" in table
    print("describe / config_table: OK")


def main() -> int:
    case_registry()
    case_paths_are_absolute()
    case_no_derived_fields_in_config()
    case_to_namespace_adapts()
    case_frozen()
    case_typo_rejected()
    case_validate_rejects_bad_values()
    case_missing_csv_rejected()
    case_trainer_integration()
    case_default_save_path_absolute()
    case_describe_and_table()
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
