"""实验总入口:按配置名跑训练与性能测试。

重构前这里是 50 行硬编码:一份手写 dict + 9 个实验步骤直接铺在
`if __name__ == "__main__":` 下面。问题:

    * 想只跑其中一个实验,只能注释掉其余代码;
    * 想改一个参数,要在 4 个地方同步;
    * 跑失败后无法知道"第三次实验"的确切配置是什么;
    * device="cuda" 写死,本机没有 CUDA 就在第一次 switch_device 崩掉。

现在拆成三个职责分明的函数:

    run_one()    训练一个配置,并记录模型复杂度
    benchmark()  只在设备间比延迟,不训练
    main()       解析参数、决定跑哪些配置

配置本身不在这里,全部来自 configs.py 的注册表 —— 这是"单一来源"的落点。

用法:

    python src\\run_exp.py --list                      # 看所有配置
    python src\\run_exp.py                             # 跑默认实验
    python src\\run_exp.py --configs A B                # 指定多个
    python src\\run_exp.py --benchmark --devices cpu    # 只测延迟
"""

import argparse
import os
import sys

# 允许直接 `python src\run_exp.py`，不必手动设置 PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import (  # noqa: E402
    available_configs,
    build_config,
    config_table,
)
from profiling import get_model_macs_and_parameters  # noqa: E402
from train import Trainer  # noqa: E402

DEFAULT_EXPERIMENTS = ("barcode_mobilenet_128",)
"""默认只跑条码集的快速配置（一轮约 30 秒，10 轮约 5 分钟）。

重构前默认是 9 次 30 轮训练，跑满要好几个小时 —— 一个不该让人无意中
触发的默认值。想跑大实验显式写 --configs。
"""


def run_one(config_name, **overrides):
    """训练一个配置,并记录 MACs 与参数量。

    参数:
        config_name: configs.py 注册表里的名字。
        **overrides: 临时覆盖项（如 save_path / num_workers）。

    返回:
        Trainer 实例,方便调用方接着做别的事（比如再测一次延迟）。
    """
    config = build_config(config_name, **overrides)
    print(f"\n{'=' * 72}")
    print(f"实验: {config.name}")
    print(f"{'=' * 72}")
    print(config.describe())
    print()

    trainer = Trainer(config.to_namespace())
    trainer.run()
    trainer.get_model_macs_and_parameters()
    return trainer


def benchmark(config_name, devices=("cpu",), **overrides):
    """不做训练,只比同一模型在不同设备上的推理延迟。

    这里**不调用** Trainer.get_model_macs_and_parameters()：那个方法会把
    报告写进 save_path，而纯测性能不该产生任何产物（也就不会因为产物目录
    不可写而失败）。直接调 profiling 拿数字打印。

    类别数从数据集问出来（它属于数据集/SplitBundle,不属于配置),
    这样就不必在配置里手写 num_classes —— 写错了是真能静默跑偏的。
    """
    config = build_config(config_name, **overrides)
    trainer = Trainer(config.to_namespace())
    num_classes = trainer.dataset_init()
    trainer.model_init(num_classes)

    print(
        f"  模型 {config.model_name} | 类别数 {num_classes} | "
        f"输入 {config.input_size[0]}x{config.input_size[1]} | "
        f"通道 {trainer.in_channels}"
    )
    macs, params = get_model_macs_and_parameters(
        trainer.model, (1, trainer.in_channels, *config.input_size)
    )
    print(f"  MACs {macs / 1e9:.3f} G | 参数 {params:,}")
    for device_name in devices:
        trainer.switch_device(device_name)
        # save_report=False：纯测性能，不写任何文件
        latency = trainer.latency_test(save_report=False)
        print(f"  {device_name:<5} 平均延迟 {latency:.3f} ms")
    return trainer


def main(argv=None):
    """命令行入口。"""
    parser = argparse.ArgumentParser(
        description="按命名配置跑实验（配置定义见 src/configs.py）"
    )
    parser.add_argument(
        "--configs", nargs="+", default=list(DEFAULT_EXPERIMENTS),
        metavar="NAME", help=f"要跑的配置名，可选：{available_configs()}",
    )
    parser.add_argument(
        "--list", action="store_true", help="只打印配置总表，不跑实验"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="只测延迟，不训练"
    )
    parser.add_argument(
        "--devices", nargs="+", default=["cpu"],
        help="benchmark 模式下要对比的设备",
    )
    parser.add_argument(
        "--save_path", default=None,
        help="覆盖产物目录（默认由 Trainer 生成带时间戳的目录）",
    )
    parser.add_argument(
        "--num_workers", type=int, default=None,
        help="覆盖 DataLoader 进程数（Windows 下调试报错时用 0）",
    )
    args = parser.parse_args(argv)

    overrides = {
        key: value
        for key, value in (
            ("save_path", args.save_path),
            ("num_workers", args.num_workers),
        )
        if value is not None
    }

    if args.list:
        print(config_table())
        return 0

    if args.benchmark:
        for config_name in args.configs:
            benchmark(config_name, args.devices, **overrides)
        return 0

    for config_name in args.configs:
        run_one(config_name, **overrides)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
