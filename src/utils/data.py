import torch


def cast_floats_by_trainer_precision(item, *, precision):
    if "64" in precision:
        return cast_floats(item, device=item.device, dtype=torch.float64)
    elif "32" in precision:
        return cast_floats(item, device=item.device, dtype=torch.float32)
    elif "bf16" in precision:
        return cast_floats(item, device=item.device, dtype=torch.bfloat16)
    elif "fp16" in precision:
        return cast_floats(item, device=item.device, dtype=torch.half)
    else:
        return item


def cast_floats(item, *, device, dtype, non_blocking=True):
    if torch.is_tensor(item):
        if torch.is_floating_point(item):
            return item.to(device=device, dtype=dtype, non_blocking=non_blocking)
        else:
            return item.to(device=device, non_blocking=non_blocking)
    elif isinstance(item, dict):
        return {
            k: cast_floats(v, device=device, dtype=dtype, non_blocking=non_blocking)
            for k, v in item.items()
        }
    elif isinstance(item, (list, tuple)):
        t = [cast_floats(v, device=device, dtype=dtype, non_blocking=non_blocking) for v in item]
        return type(item)(*t) if isinstance(item, tuple) else t
    else:
        return item


def normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    x = x.clone()
    x = (x - mean) / std
    return x


def unnormalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    x = x.clone()
    x = x * std
    x = x + mean
    return x
