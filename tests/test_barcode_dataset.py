"""BarcodeDataset 的验收测试。

契约::

    dataset[i] -> (image, label)

    image: float32 张量,形状恒为 (C, H, W),H/W 等于 target_size,
           C 由 keep_grayscale 决定(True -> 1,False -> 3),取值 [-1, 1]。
    label: python int,取值 [0, num_classes)。

    核心不变量:
    1. 【形状恒定】不管有没有提供 transform,输出形状都只由 target_size 决定。
       否则 DataLoader 无法拼 batch —— 这是重构前最大的坑。
    2. 【基础管道恒执行】增强只负责改像素,Resize/Normalize/ToTensorV2
       由数据集统一做;增强里误带 ToTensorV2 必须立刻报 TypeError。
    3. 【类别次序固定】给定 fixed_classes 时,下标由它决定而不是 CSV 顺序;
       出现未知类别必须抛 ValueError,不能静默变成 NaN 标签。
    4. 【异常不静默】增强失败只允许降级有限次数,超过上限必须报错,
       避免"训练照跑但预处理全错"。
    5. 【缓存不改语义】cache_images=True 与 False 的输出必须逐位相同。

本文件用临时目录里生成的合成条码图,不依赖任何外部数据集,
因此 clone 下来就能跑。

运行::

    python tests/test_barcode_dataset.py
"""

import contextlib
import os
import shutil
import sys
import uuid

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from datasets import (  # noqa: E402
    AUGMENT_C39,
    AUGMENT_NONE,
    AUGMENT_SYN,
    BarcodeDataset,
    build_barcode_class_mapping,
)

# 工作区内的临时目录根，已在 .gitignore 中排除
_TMP_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".tmp_tests")
)

# 真实数据集(可能不存在,存在时额外跑一次规模检查)
REAL_CSV = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "dataset", "barber.csv")
)


@contextlib.contextmanager
def temp_dir():
    """在工作区内创建临时目录，退出时清理。

    不使用 tempfile.mkdtemp：系统临时目录在本机沙箱下不可写
    （PermissionError），且它建出的目录权限过严（0o700）。
    改用工作区内的普通 mkdir，权限按默认继承。
    """
    os.makedirs(_TMP_ROOT, exist_ok=True)
    path = os.path.join(_TMP_ROOT, uuid.uuid4().hex)
    os.makedirs(path, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class InvertImage:
    """把所有像素取反的假增强:用于验证增强确实被调用,且顺序正确。"""

    def __call__(self, image, **kwargs):
        return {"image": 255 - image}


class BrokenTransform:
    """必定抛异常的假增强:用于验证失败上限机制。"""

    def __call__(self, image, **kwargs):
        raise RuntimeError("故意失败")


def make_bar_image(height, width, channels=1):
    """生成一张竖条码样式的合成图,通道数由 channels 决定。"""
    img = np.full((height, width), 255, dtype=np.uint8)
    # 每 6 列画 2 列黑条,模拟条空交替
    for col in range(0, width, 6):
        img[:, col:col + 2] = 0
    if channels == 1:
        return img
    # 彩色图:三个通道给不同值,便于验证 BGR->RGB 的转换方向
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if channels == 4:
        alpha = np.full((height, width, 1), 255, dtype=np.uint8)
        img = np.concatenate([img, alpha], axis=2)
    return img


def write_png(path, image):
    """把 ndarray 写成 PNG。

    不能用 cv2.imwrite:它在本项目路径(含中文)下会静默失败。
    改为先用 cv2.imencode 编码到内存,再用 Python 的文件层写字,
    这样路径解析交给 Python,不受 OpenCV 窄字符 API 的限制。
    """
    ok, buffer = cv2.imencode(".png", image)
    assert ok, f"PNG 编码失败: {path}"
    with open(path, "wb") as handle:
        handle.write(buffer.tobytes())


def make_fake_dataset(root, labels=("C128", "C128", "EAN13", "EAN13"),
                      domains=("photo", "photo", "photo", "synthetic")):
    """在 root 下造 4 张图 + 一份 CSV,返回 (csv_path, 图像路径列表)。

    刻意混入 3 通道与 4 通道图,用来覆盖 _read_image 的通道分支。
    """
    specs = [(40, 120, 1), (60, 200, 1), (30, 90, 3), (30, 90, 4)]
    paths = []
    for index, (height, width, channels) in enumerate(specs):
        path = os.path.join(root, f"img_{index}.png")
        write_png(path, make_bar_image(height, width, channels))
        paths.append(path)
    csv_path = os.path.join(root, "labels.csv")
    pd.DataFrame({
        "image_path": paths,
        "label_name": list(labels),
        "source_domain": list(domains),
    }).to_csv(csv_path, index=False)
    return csv_path, paths


def assert_tensor(tensor, channels, height, width, where):
    """统一断言:张量形状、dtype、值域。"""
    assert isinstance(tensor, torch.Tensor), f"{where}: 不是 torch.Tensor"
    expected = (channels, height, width)
    assert tuple(tensor.shape) == expected, (
        f"{where}: 形状期望 {expected},实际 {tuple(tensor.shape)}"
    )
    assert tensor.dtype.is_floating_point, f"{where}: 期望浮点,实际 {tensor.dtype}"
    assert float(tensor.min()) >= -1.0 - 1e-5, f"{where}: 下界越界"
    assert float(tensor.max()) <= 1.0 + 1e-5, f"{where}: 上界越界"


# ---------------------------------------------------------------------------
# 1. 类名映射
# ---------------------------------------------------------------------------

def case_class_mapping() -> None:
    """给定 fixed_classes 时按它定次序;未知类别必须报错;None 时按内容排序。"""
    names, mapping = build_barcode_class_mapping(
        ["EAN13", "C128", "EAN13"], fixed_classes=("C128", "EAN13")
    )
    assert names == ["C128", "EAN13"], f"类别次序错: {names}"
    assert mapping == {"C128": 0, "EAN13": 1}, f"映射错: {mapping}"

    # 小写也要能匹配(内部统一转大写)
    _, mapping_lower = build_barcode_class_mapping(
        ["c128"], fixed_classes=("c128",)
    )
    assert mapping_lower == {"C128": 0}, f"小写未归一: {mapping_lower}"

    # 未知类别必须报错,否则标签会变 NaN
    try:
        build_barcode_class_mapping(["C39"], fixed_classes=("C128", "EAN13"))
    except ValueError as exc:
        assert "C39" in str(exc), f"报错信息没提到未知类别: {exc}"
    else:
        raise AssertionError("未知类别应抛 ValueError")

    # 不给固定次序时按内容排序
    names_auto, mapping_auto = build_barcode_class_mapping(
        ["EAN13", "C128", "C128"], fixed_classes=None
    )
    assert names_auto == ["C128", "EAN13"], f"自动排序错: {names_auto}"
    assert mapping_auto == {"C128": 0, "EAN13": 1}
    print("类别映射: OK")


# ---------------------------------------------------------------------------
# 2. 输出形状与通道
# ---------------------------------------------------------------------------

def case_shape_without_transform(tmp: str) -> None:
    """不传 transform 也必须得到 target_size 的形状(重构前的核心缺陷)。"""
    csv_path, _ = make_fake_dataset(tmp)
    dataset = BarcodeDataset(
        csv_file=csv_path, target_size=(64, 64), verbose=False
    )
    # 4 张原图尺寸各不相同,输出必须完全一致
    for index in range(len(dataset)):
        image, label = dataset[index]
        assert_tensor(image, 1, 64, 64, f"样本{index}")
        assert isinstance(label, int), f"标签应是 int,实际 {type(label)}"
    print("不传 transform 仍按 target_size 输出: OK")


def case_keep_grayscale_false(tmp: str) -> None:
    """keep_grayscale=False 时输出三通道,灰度图也要扩展成 RGB。"""
    csv_path, _ = make_fake_dataset(tmp)
    dataset = BarcodeDataset(
        csv_file=csv_path, target_size=(32, 32), keep_grayscale=False,
        verbose=False,
    )
    for index in range(len(dataset)):
        image, _ = dataset[index]
        assert_tensor(image, 3, 32, 32, f"样本{index}")
    print("keep_grayscale=False 输出三通道: OK")


def case_target_size_respected(tmp: str) -> None:
    """target_size 必须真正生效(重构前传了 transform 就被忽略)。"""
    csv_path, _ = make_fake_dataset(tmp)
    dataset = BarcodeDataset(
        csv_file=csv_path, target_size=(32, 96), verbose=False
    )
    image, _ = dataset[0]
    assert_tensor(image, 1, 32, 96, "target_size=(32,96)")
    print("target_size 生效: OK")


# ---------------------------------------------------------------------------
# 3. 增强管道
# ---------------------------------------------------------------------------

def case_augment_applied(tmp: str) -> None:
    """增强必须在归一化之前生效:取反后的结果应等于原结果的相反数。"""
    csv_path, _ = make_fake_dataset(tmp)
    plain = BarcodeDataset(csv_file=csv_path, target_size=(32, 32),
                           verbose=False)
    inverted = BarcodeDataset(csv_file=csv_path, target_size=(32, 32),
                              transform=InvertImage(), verbose=False)
    base, _ = plain[0]
    flipped, _ = inverted[0]
    # (255-x) 归一化后正好是 x 归一化结果的相反数
    assert torch.allclose(flipped, -base, atol=1e-5), (
        "增强结果不等于取反,说明增强没有作用在归一化之前"
    )
    print("增强在归一化之前生效: OK")


def case_augment_with_totensor_rejected(tmp: str) -> None:
    """增强里误带 ToTensorV2 必须立刻抛 TypeError 并说明原因。"""
    from albumentations.pytorch import ToTensorV2
    import albumentations as A

    csv_path, _ = make_fake_dataset(tmp)
    dataset = BarcodeDataset(
        csv_file=csv_path, target_size=(32, 32),
        transform=A.Compose([ToTensorV2()]), verbose=False,
    )
    try:
        dataset[0]
    except TypeError as exc:
        assert "ToTensorV2" in str(exc), f"报错信息没说清原因: {exc}"
    else:
        raise AssertionError("增强返回张量时应抛 TypeError")
    print("增强含 ToTensorV2 被拒绝: OK")


def case_augment_failure_capped(tmp: str) -> None:
    """增强失败前 N 次降级,第 N+1 次必须报错,不能无限静默降级。"""
    csv_path, _ = make_fake_dataset(tmp)
    dataset = BarcodeDataset(
        csv_file=csv_path, target_size=(32, 32),
        transform=BrokenTransform(), verbose=False,
    )
    dataset.MAX_TRANSFORM_FAILURES = 2

    # 前 2 次:降级为基础预处理,形状仍然正确
    for index in range(2):
        image, _ = dataset[index]
        assert_tensor(image, 1, 32, 32, f"降级样本{index}")

    # 第 3 次超过上限,必须抛错
    try:
        dataset[2]
    except RuntimeError as exc:
        assert "增强管道失败" in str(exc), f"报错信息不明确: {exc}"
    else:
        raise AssertionError("超过失败上限时应抛 RuntimeError")
    print("增强失败次数上限生效: OK")


def case_augment_routing(tmp: str) -> None:
    """增强分流:合成数据走 transform_syn,优先级高于 C39。"""
    # 第 4 张标为 synthetic,应命中 transform_syn
    csv_path, _ = make_fake_dataset(tmp)
    dataset = BarcodeDataset(
        csv_file=csv_path, target_size=(32, 32),
        transform_syn=InvertImage(), verbose=False,
    )
    tags = dataset._augment_tags.tolist()
    assert tags == [AUGMENT_NONE, AUGMENT_NONE, AUGMENT_NONE, AUGMENT_SYN], (
        f"分流标记错: {tags}"
    )
    print("transform_syn 分流: OK")

    # C39 样本命中 transform_c39
    root = os.path.join(tmp, "c39")
    os.makedirs(root)
    csv_c39, _ = make_fake_dataset(
        root, labels=("C39", "C39", "EAN13", "EAN13"),
        domains=("photo", "photo", "photo", "photo"),
    )
    ds_c39 = BarcodeDataset(
        csv_file=csv_c39, target_size=(32, 32),
        fixed_classes=("C128", "EAN13", "C39"),
        transform_c39=InvertImage(), verbose=False,
    )
    tags_c39 = ds_c39._augment_tags.tolist()
    assert tags_c39 == [AUGMENT_C39, AUGMENT_C39,
                        AUGMENT_NONE, AUGMENT_NONE], f"C39 标记错: {tags_c39}"
    print("transform_c39 分流: OK")


# ---------------------------------------------------------------------------
# 4. 缓存与索引
# ---------------------------------------------------------------------------

def case_cache_images_equal(tmp: str) -> None:
    """cache_images=True 与 False 的输出必须逐位相同。"""
    csv_path, _ = make_fake_dataset(tmp)
    lazy = BarcodeDataset(csv_file=csv_path, target_size=(32, 32),
                          verbose=False)
    cached = BarcodeDataset(csv_file=csv_path, target_size=(32, 32),
                            cache_images=True, verbose=False)
    assert len(cached._cache) == len(lazy), "缓存数量与样本数不一致"
    for index in range(len(lazy)):
        lazy_image, lazy_label = lazy[index]
        cached_image, cached_label = cached[index]
        assert lazy_label == cached_label, f"样本{index}: 标签不一致"
        assert torch.equal(lazy_image, cached_image), (
            f"样本{index}: 缓存与非缓存输出不一致"
        )
    print("cache_images 不改变结果: OK")


def case_negative_index(tmp: str) -> None:
    """负下标应与正下标的结尾对齐。"""
    csv_path, _ = make_fake_dataset(tmp)
    dataset = BarcodeDataset(csv_file=csv_path, target_size=(32, 32),
                             verbose=False)
    tail, tail_label = dataset[len(dataset) - 1]
    negative, negative_label = dataset[-1]
    assert tail_label == negative_label, "负下标标签不一致"
    assert torch.equal(tail, negative), "负下标图像不一致"
    print("负下标: OK")


# ---------------------------------------------------------------------------
# 5. 工具方法
# ---------------------------------------------------------------------------

def case_validate_files(tmp: str) -> None:
    """validate_files 应发现磁盘上缺失的图片。"""
    root = os.path.join(tmp, "missing")
    os.makedirs(root)
    csv_path, paths = make_fake_dataset(root)
    dataset = BarcodeDataset(csv_file=csv_path, target_size=(32, 32),
                             verbose=False)
    assert dataset.validate_files() == [], "正常数据不应报告缺失"

    os.remove(paths[1])
    missing = dataset.validate_files()
    assert len(missing) == 1, f"应报告 1 个缺失,实际 {len(missing)}"
    assert os.path.basename(missing[0]) == "img_1.png", f"缺失项错: {missing}"
    print("validate_files: OK")


def case_worker_init_fn() -> None:
    """worker_init_fn 应关闭 cv2 线程池,避免与 DataLoader 抢核。"""
    cv2.setNumThreads(4)
    assert cv2.getNumThreads() > 1, "前置条件不成立:线程数没被设成多线程"
    BarcodeDataset.worker_init_fn(0)
    # setNumThreads(0) 的语义是「关闭并行」,OpenCV 内部按单线程跑,
    # 但 getNumThreads() 报告的是 1 而不是 0,所以这里断言 <= 1
    assert cv2.getNumThreads() <= 1, (
        f"调用后 cv2 应退化为单线程,实际 {cv2.getNumThreads()}"
    )
    print("worker_init_fn 关闭 cv2 线程: OK")


def case_repr_and_map(tmp: str) -> None:
    """repr 与 get_map 的基本行为。"""
    csv_path, _ = make_fake_dataset(tmp)
    dataset = BarcodeDataset(csv_file=csv_path, target_size=(32, 32),
                             verbose=False)
    text = repr(dataset)
    assert "BarcodeDataset" in text, f"repr 内容异常: {text}"
    assert "classes=2" in text, f"repr 未包含类别数: {text}"
    assert dataset.get_map() is dataset.class_to_idx, "get_map 应返回同一对象"
    print("repr / get_map: OK")


# ---------------------------------------------------------------------------
# 6. 真实数据集(可选)
# ---------------------------------------------------------------------------

def case_real_dataset() -> None:
    """真实 barber.csv 的规模检查;文件不存在时跳过。"""
    if not os.path.isfile(REAL_CSV):
        print("真实数据集: 跳过(未找到 dataset/barber.csv)")
        return
    dataset = BarcodeDataset(csv_file=REAL_CSV, target_size=(224, 224),
                             verbose=False)
    assert len(dataset) == 6527, f"样本数期望 6527,实际 {len(dataset)}"
    assert dataset.num_classes == 2
    assert dataset.class_to_idx == {"C128": 0, "EAN13": 1}
    assert dataset.validate_files() == [], "存在缺失图像"
    image, label = dataset[0]
    assert_tensor(image, 1, 224, 224, "真实样本0")
    assert 0 <= label < dataset.num_classes
    print("真实数据集规模: OK")


def main() -> int:
    with temp_dir() as tmp:
        case_class_mapping()
        case_shape_without_transform(tmp)
        case_keep_grayscale_false(tmp)
        case_target_size_respected(tmp)
        case_augment_applied(tmp)
        case_augment_with_totensor_rejected(tmp)
        case_augment_failure_capped(tmp)
        case_augment_routing(tmp)
        case_cache_images_equal(tmp)
        case_negative_index(tmp)
        case_validate_files(tmp)
        case_worker_init_fn()
        case_repr_and_map(tmp)
        case_real_dataset()
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
