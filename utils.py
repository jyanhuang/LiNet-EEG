import torch
import torch.nn as nn

def count_layer_flops(layer, x, y):
    """统计单层 FLOPs"""
    flops = 0

    # 卷积层 FLOPs = kernel_h * kernel_w * in_c * out_h * out_w * out_c / groups
    if isinstance(layer, nn.Conv2d):
        out_h, out_w = y.shape[2], y.shape[3]
        kernel_h, kernel_w = layer.kernel_size
        in_c = layer.in_channels
        out_c = layer.out_channels
        groups = layer.groups
        flops = (kernel_h * kernel_w * in_c * out_h * out_w * out_c) / groups

    # 全连接层 FLOPs = in_features * out_features
    elif isinstance(layer, nn.Linear):
        flops = layer.in_features * layer.out_features

    # BatchNorm：每个元素两次操作（乘+加）
    elif isinstance(layer, nn.BatchNorm2d):
        flops = x.numel() * 2

    # 激活函数（如ReLU）
    elif isinstance(layer, (nn.ReLU, nn.Sigmoid, nn.Tanh)):
        flops = x.numel()

    # 池化层（AvgPool / MaxPool）
    elif isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d)):
        kernel_ops = layer.kernel_size * layer.kernel_size if isinstance(layer.kernel_size, int) \
                     else layer.kernel_size[0] * layer.kernel_size[1]
        flops = y.numel() * kernel_ops

    # SE 模块内部：AdaptiveAvgPool + 全连接 + 激活
    elif isinstance(layer, nn.AdaptiveAvgPool2d):
        flops = x.numel()  # 平均操作一次算一个加法

    return flops

def calculate_model_flops(model, input_shape):
    """计算整个模型的 FLOPs"""
    model = model.eval()
    device = next(model.parameters()).device
    x = torch.randn(input_shape).to(device)

    flops = 0
    hooks = []

    def hook_fn(layer, input, output):
        nonlocal flops
        flops += count_layer_flops(layer, input[0], output)

    # 给模型中每一层注册 forward hook
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear, nn.BatchNorm2d,
                              nn.ReLU, nn.Sigmoid, nn.Tanh,
                              nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
            hooks.append(layer.register_forward_hook(hook_fn))

    # 执行一次前向传播
    with torch.no_grad():
        _ = model(x)

    # 移除 hooks
    for h in hooks:
        h.remove()

    return flops