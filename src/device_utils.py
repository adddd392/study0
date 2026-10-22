"""
    独立模块的设备解析
"""
import torch


def resolve_device(device_name):
    """根据设备名返回对应的 torch.device 对象。"""
    if device_name == "dml":
        import torch_directml
        if not torch_directml.is_available():
            raise ValueError("DirectML is not available.")
        return torch_directml.device()
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        return torch.device("cuda")
    raise ValueError(f"Unknown device: {device_name!r}")

