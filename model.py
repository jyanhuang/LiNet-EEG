import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# SE Attention
# ----------------------------
class SEAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ----------------------------
# RuLCNN (Onnx-safe, No LogSoftmax)
# ----------------------------
class RuLCNN(nn.Module):
    def __init__(self, classes=2, sampleChannel=1, sampleLength=384, N1=16, d=2, kernelLength=64):
        super(RuLCNN, self).__init__()

        # Common Conv:
        # self.common_conv = nn.Conv2d(
        #     in_channels=1,
        #     out_channels=32,
        #     kernel_size=(1, 64),
        #     groups=1,
        #     bias=True
        # )

        # Pointwise conv: (1 × C × W) → (N1 × 1 × W)
        self.pointwise = nn.Conv2d(
            in_channels=1,
            out_channels=N1,
            kernel_size=(sampleChannel, 1),
            bias=True
        )

        # Depthwise conv
        self.depthwise = nn.Conv2d(
            in_channels=N1,
            out_channels=d * N1,
            kernel_size=(1, kernelLength),
            groups=N1,                     # depthwise
            bias=True
        )

        self.relu = nn.ReLU(inplace=True)

        # BatchNorm - 更适合ONNX且稳定
        self.bn = nn.BatchNorm2d(d * N1)

        # SE attention
        self.se = SEAttention(d * N1)
        # self.se = nn.Identity()

        # Global Avg Pooling 来自 W-kernelLength+1
        gap_size = sampleLength - kernelLength + 1
        self.gap = nn.AvgPool2d(kernel_size=(1, gap_size))

        # 线性层
        feature_dim = d * N1 * 1 * 1 * 30   # (batch, channels, 30, 1) after GAP
        self.fc = nn.Linear(feature_dim, classes)

        # ============ 剪枝配置（仅可剪 pointwise & fc）===========
        self.pruning_config = {
            "pointwise": {"amount": 0.0, "method": "l1"},
            "fc": {"amount": 0.0, "method": "l1"},
        }

    # ----------------------------
    # Forward
    # ----------------------------
    def forward(self, x):
        # x: (B, 1, 30, 384)
        # x = self.common_conv(x)
        x = self.pointwise(x)
        x = self.depthwise(x)
        x = self.relu(x)
        x = self.bn(x)

        x = self.se(x)

        x = self.gap(x)           # → shape (B, C, 30, 1)
        x = x.view(x.size(0), -1) # flatten

        logits = self.fc(x)       # 不做 softmax！
        return logits


    # ----------------------------
    # 可安全使用的剪枝（不能剪 depthwise）
    # ----------------------------
    def apply_pruning(self):
        import torch.nn.utils.prune as prune

        for name, config in self.pruning_config.items():
            amount = config["amount"]
            method = config["method"]

            if amount <= 0:
                continue

            module = getattr(self, name, None)
            if module is None:
                continue

            print(f"Pruning {name}: amount={amount}, method={method}")

            if method == "l1":
                prune.l1_unstructured(module, name='weight', amount=amount)
            elif method == "random":
                prune.random_unstructured(module, name='weight', amount=amount)

            prune.remove(module, 'weight')

    # ----------------------------
    # 计算稀疏度
    # ----------------------------
    def get_model_sparsity(self):
        print("sparsity....")
        total = 0
        zeros = 0
        for p in self.parameters():
            total += p.numel()
            zeros += torch.sum(p == 0).item()
        print('#######pruning ratio:{}, zero_params: {}, total_params:{}, trainable_params:{}'.format(0.2, zeros, total, total-zeros))
        return 100 * zeros / total


