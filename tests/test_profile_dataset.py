"""datasets/profile.py 的黄金测试。

锁定八件事：
    1. 契约：(2, L) float32 张量 + 整数标签，标签语义与 BarcodeDataset 一致
    2. class_counts 正确
    3. 缓存能命中，且**参数变化后自动失效**（指纹机制真的起作用）
    4. on_fail='error' 抛错、'skip' 丢弃并计数（不静默补零）
    5. 全黑图不产生 NaN
    6. pre_size 控制组可用且与主口径不同
    7. 相对路径 + data_root 能解析
    8. 重复构建结果逐位一致

运行::

    python tests/test_profile_dataset.py
"""

import os
import shutil
import sys

import cv2
import numpy as np
import pandas as pd

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from datasets.profile import ProfileDataset, frontend_fingerprint  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_TMP = os.path.abspath(os.path.join(_HERE, "..", ".tmp_tests", "_profile_ds"))


def _write_image(path, img):
    """写图。必须走 imencode + tofile。

    本仓库的工作目录含中文（...\\代码\\study），cv2.imwrite 在非 ASCII
    路径上会**静默返回 False**，图根本不会落盘。这与 image_io 里用
    np.fromfile + imdecode 读图是同一个原因。
    """
    ok, buffer = cv2.imencode(".png", img)
    assert ok, "PNG 编码失败"
    buffer.tofile(path)


def _make_barcode(path, width=200, height=60, bar=5):
    """造一张竖直条码图并存盘。"""
    img = np.zeros((height, width), np.uint8)
    for x in range(width):
        img[:, x] = 255 if (x // bar) % 2 == 0 else 0
    _write_image(path, img)


def _setup_dir():
    """准备临时目录与 6 张图（3 张 C128、3 张 EAN13，条宽差异明显）。"""
    if os.path.isdir(_TMP):
        shutil.rmtree(_TMP)
    os.makedirs(_TMP)
    rows = []
    # 两类图故意用不同条宽：C128 密（bar=3），EAN13 疏（bar=7）
    for index, (label, bar) in enumerate([("C128", 3), ("C128", 3),
                                          ("C128", 4), ("EAN13", 7),
                                          ("EAN13", 7), ("EAN13", 8)]):
        name = "img_%d.png" % index
        _make_barcode(os.path.join(_TMP, name), width=180 + index * 10,
                      height=60, bar=bar)
        rows.append({"image_path": os.path.join(_TMP, name),
                     "label_name": label, "original_type": label})
    csv_path = os.path.join(_TMP, "manifest.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def case_contract() -> None:
    """形状 / 类型 / 标签语义。"""
    csv_path = _setup_dir()
    ds = ProfileDataset(csv_path, target_length=256, verbose=False)
    assert len(ds) == 6, "样本数应为 6, 实际 %d" % len(ds)
    sample, label = ds[0]
    assert tuple(sample.shape) == (2, 256), "形状应为 (2, 256)"
    assert sample.dtype.is_floating_point, "dtype 应为浮点"
    assert isinstance(label, int), "标签应是 python int"
    # C128 -> 0, EAN13 -> 1，与 BarcodeDataset 的固定次序一致
    assert ds.class_names == ["C128", "EAN13"], "类别次序错误: %r" % (
        ds.class_names,)
    assert [int(y) for y in ds.labels] == [0, 0, 0, 1, 1, 1], "标签映射错误"
    assert ds.class_counts().tolist() == [3, 3], "类别计数错误"
    assert np.all(np.isfinite(ds.X)), "剖面含 NaN/Inf"
    print("契约: OK")


def case_cache_roundtrip() -> None:
    """写缓存后第二次应命中，且数据逐位一致。"""
    csv_path = _setup_dir()
    cache = os.path.join(_TMP, "cache.npz")
    ds1 = ProfileDataset(csv_path, target_length=256, cache_path=cache,
                         verbose=False)
    assert os.path.isfile(cache), "缓存文件未生成"
    ds2 = ProfileDataset(csv_path, target_length=256, cache_path=cache,
                         verbose=True)
    assert np.array_equal(ds1.X, ds2.X), "缓存命中但数据不一致"
    assert np.array_equal(ds1.labels, ds2.labels), "缓存标签不一致"
    print("缓存往返: OK")


def case_cache_invalidated_by_params() -> None:
    """口径参数变化后旧缓存必须失效，不能静默复用。"""
    csv_path = _setup_dir()
    cache = os.path.join(_TMP, "cache_param.npz")
    ds1 = ProfileDataset(csv_path, target_length=256, hp_frac=0.06,
                         cache_path=cache, verbose=False)
    # 换 hp_frac：指纹不同，必须重算
    ds2 = ProfileDataset(csv_path, target_length=256, hp_frac=0.20,
                         cache_path=cache, verbose=False)
    assert not np.array_equal(ds1.X, ds2.X), (
        "hp_frac 改了但结果完全相同 —— 缓存失效机制没起作用"
    )
    # 换 L：形状就不同了
    ds3 = ProfileDataset(csv_path, target_length=128, hp_frac=0.20,
                         cache_path=cache, verbose=False)
    assert ds3.X.shape[-1] == 128, "L 变化后形状未更新"
    # 指纹函数本身必须敏感
    assert (frontend_fingerprint(256, 0.06, None)
            != frontend_fingerprint(256, 0.20, None)), "指纹对 hp_frac 不敏感"
    assert (frontend_fingerprint(256, 0.06, None)
            != frontend_fingerprint(256, 0.06, (50, 20))), "指纹对 pre_size 不敏感"
    print("缓存指纹失效: OK")


def _bad_manifest():
    """准备一份"6 张好图 + 1 条不存在路径"的清单，返回其路径。

    单独成函数是为了让 error / skip 两个用例各自独立：谁都不依赖
    "上一个用例留下了什么文件"（否则用例顺序一变就挂）。
    """
    csv_path = _setup_dir()
    frame = pd.read_csv(csv_path)
    frame.loc[len(frame)] = {
        "image_path": os.path.join(_TMP, "不存在.png"),
        "label_name": "C128",
        "original_type": "C128",
    }
    bad_csv = os.path.join(_TMP, "bad_manifest.csv")
    frame.to_csv(bad_csv, index=False)
    return bad_csv


def case_on_fail_error() -> None:
    """坏路径 + on_fail='error' 必须立刻抛错，且错误信息带路径。"""
    bad_csv = _bad_manifest()
    try:
        ProfileDataset(bad_csv, target_length=256, verbose=False)
    except RuntimeError as exc:
        assert "不存在.png" in str(exc), "错误信息应包含失败路径"
    else:
        raise AssertionError("坏路径应抛 RuntimeError")
    print("on_fail=error: OK")


def case_on_fail_skip() -> None:
    """坏路径 + on_fail='skip' 必须丢弃并计数，不补零。"""
    bad_csv = _bad_manifest()
    ds = ProfileDataset(bad_csv, target_length=256, on_fail="skip",
                        verbose=False)
    assert len(ds) == 6, "应丢弃 1 条后剩 6 条, 实际 %d" % len(ds)
    assert ds.n_fail == 1, "失败计数应为 1, 实际 %d" % ds.n_fail
    # 丢弃策略下不存在"用零剖面占位"的情况：样本数少于清单行数
    assert len(ds) < len(ds.paths), "路径数与样本数应不同"
    print("on_fail=skip: OK")


def case_flat_image_finite() -> None:
    """全黑图不能产生 NaN/Inf。"""
    csv_path = _setup_dir()
    flat = os.path.join(_TMP, "flat.png")
    _write_image(flat, np.zeros((60, 200), np.uint8))
    frame = pd.read_csv(csv_path)
    frame.loc[len(frame)] = {"image_path": flat, "label_name": "EAN13",
                             "original_type": "EAN13"}
    flat_csv = os.path.join(_TMP, "flat_manifest.csv")
    frame.to_csv(flat_csv, index=False)
    ds = ProfileDataset(flat_csv, target_length=128, verbose=False)
    assert np.all(np.isfinite(ds.X)), "全黑图产生了 NaN/Inf"
    print("全黑图有限性: OK")


def case_pre_size_control() -> None:
    """pre_size 控制组应可用，且结果与主口径不同。"""
    csv_path = _setup_dir()
    main = ProfileDataset(csv_path, target_length=256, verbose=False)
    ctrl = ProfileDataset(csv_path, target_length=256, pre_size=(50, 20),
                          verbose=False)
    assert ctrl.X.shape == main.X.shape, "控制组形状应一致"
    assert not np.array_equal(ctrl.X, main.X), "控制组与主口径结果相同"
    # 控制组的横向细节应被压掉
    assert float(ctrl.X[:, 0].std()) < float(main.X[:, 0].std()), (
        "控制组方差未下降"
    )
    print("pre_size 控制组: OK")


def case_relative_paths() -> None:
    """相对路径 + data_root 应解析到正确位置。"""
    csv_path = _setup_dir()
    frame = pd.read_csv(csv_path)
    relative = frame.copy()
    relative["image_path"] = [os.path.basename(p)
                              for p in frame["image_path"]]
    rel_csv = os.path.join(_TMP, "rel_manifest.csv")
    relative.to_csv(rel_csv, index=False)
    ds = ProfileDataset(rel_csv, data_root=_TMP, target_length=128,
                        verbose=False)
    assert len(ds) == 6, "相对路径解析失败"
    print("相对路径 + data_root: OK")


def case_determinism() -> None:
    """两次构建必须逐位一致（前端无随机性）。"""
    csv_path = _setup_dir()
    a = ProfileDataset(csv_path, target_length=256, verbose=False)
    b = ProfileDataset(csv_path, target_length=256, verbose=False)
    assert np.array_equal(a.X, b.X), "两次构建结果不同"
    print("确定性: OK")


def main() -> int:
    try:
        case_contract()
        case_cache_roundtrip()
        case_cache_invalidated_by_params()
        case_on_fail_error()
        case_on_fail_skip()
        case_flat_image_finite()
        case_pre_size_control()
        case_relative_paths()
        case_determinism()
    finally:
        if os.path.isdir(_TMP):
            shutil.rmtree(_TMP, ignore_errors=True)
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
