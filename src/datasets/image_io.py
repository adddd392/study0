"""图像读取：兼容中文路径，统一通道数。

为什么单独成模块：BarcodeDataset 与 inference.py 必须用**同一套**
读图逻辑。训练怎么读、推理就怎么读，否则通道顺序或灰度化方式一旦
不一致，就是一处查不出来的分布漂移。
"""


import cv2
import numpy as np


def read_image_any_path(img_path):
    """读图,兼容含中文/空格的路径。

    为什么不用 cv2.imread:它在 Windows 上走窄字符 API,路径里只要出现
    中文就静默返回 None(本项目仓库路径就带中文,实测必踩)。改用
    「numpy 读字节 + cv2.imdecode」,把路径解析交给 Python 的文件层,
    OpenCV 只负责解码内存中的字节,不再碰路径。

    异常:
        FileNotFoundError: 文件不存在、为空或无法解码。
    """
    try:
        buffer = np.fromfile(img_path, dtype=np.uint8)
    except OSError as exc:
        # 文件不存在、是目录、无权限都归到这里
        raise FileNotFoundError(f"图像无法读取: {img_path}") from exc
    if buffer.size == 0:
        raise FileNotFoundError(f"图像为空文件: {img_path}")
    # IMREAD_UNCHANGED 保留原始通道数,由调用方决定怎么转换
    img = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"图像无法解码(格式不支持?): {img_path}")
    return img


def read_barcode_image(img_path, keep_grayscale=True):
    """读一张条码图,并按 keep_grayscale 统一通道数。

    放在模块级而不是类里,是为了让推理端(inference.py)复用**同一套**
    通道处理:训练怎么读,推理就必须怎么读,否则分布漂移。

    返回:
        keep_grayscale=True  -> (H, W) 单通道 ndarray
        keep_grayscale=False -> (H, W, 3) RGB ndarray
    异常:
        FileNotFoundError: 路径不存在或文件无法解码。
        ValueError: 通道数不支持。
    """
    img = read_image_any_path(img_path)
    channels = 1 if img.ndim == 2 else img.shape[2]
    if keep_grayscale:
        # 彩色/带 alpha 的一律转灰度;本来就是灰度的原样返回
        if channels == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif channels == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        # 统一成 RGB 三通道:OpenCV 默认是 BGR,不转换会让红蓝互换
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif channels == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif channels == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            raise ValueError(f"不支持的图像通道数: {channels} ({img_path})")
    return img
