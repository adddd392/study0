"""标签映射构建。

两种数据集的映射规则不同：
- Mini-ImageNet：按 label 在 CSV 中首次出现的顺序编号
- 条码：类别次序必须固定（生产环境传 fixed_classes），否则换一份 CSV
  就可能让 C128 从 0 变成 1，预测结果整体错位
"""


import pandas as pd


def build_label_to_index_map(csv_file):
    """按训练时的规则构建 label -> index 映射。

    规则:按 label 在 CSV 中**首次出现的顺序**依次编号 0,1,2...

    推理时必须用 train.csv(而不是 val/test.csv)调用本函数,才能得到
    与训练完全一致的下标语义;否则预测出的下标对应的类别会整体错位。

    参数:
        csv_file: 训练集 CSV 路径,需包含 label 列。
    返回:
        dict,形如 {'n01532829': 0, 'n01558993': 1, ...}
    """
    df = pd.read_csv(csv_file)
    label_classes = df["label"].unique()
    return {label: i for i, label in enumerate(label_classes)}


def build_barcode_class_mapping(label_names, fixed_classes=None):
    """构建 (class_names, class_to_idx),并拒绝 CSV 里的未知类别。

    为什么要有固定类别:训练与推理必须用同一套下标语义。若每次按 CSV
    内容排序生成,换一份 CSV 就可能让 C128 从 0 变成 1,预测结果整体错位。
    因此生产环境应始终传 fixed_classes。

    参数:
        label_names: CSV 中出现过的类别名。会先统一 strip + 大写,
            因此 'c128 ' 和 'C128' 视为同一类。
        fixed_classes: 固定类别次序(同样会归一化);None 表示按内容排序生成。
    返回:
        (class_names, class_to_idx),形如
        (['C128', 'EAN13'], {'C128': 0, 'EAN13': 1})
    异常:
        ValueError: CSV 中出现了 fixed_classes 之外的类别。
    """
    # 两侧都归一化,调用方就不必自己记得先转大写
    normalized = {str(name).strip().upper() for name in label_names}
    if fixed_classes is None:
        # 兜底路径:没有给定次序时才按内容排序
        class_names = sorted(normalized)
    else:
        # 统一转大写,避免 fixed_classes 里写 'c128' 就匹配不上
        class_names = [str(name).strip().upper() for name in fixed_classes]
        unknown = normalized - set(class_names)
        if unknown:
            # 必须报错而不是忽略:被忽略的样本会让标签变 NaN,
            # 进而让 CrossEntropyLoss 收到浮点标签并在训练中段才崩
            raise ValueError(
                f"CSV 包含未知类别: {sorted(unknown)},"
                f"当前 fixed_classes={class_names}。"
                "请把新类别加入 fixed_classes,或确认数据是否混入了别的码制。"
            )
    class_to_idx = {
        name: index
        for index, name in enumerate(class_names)
    }
    return class_names, class_to_idx
