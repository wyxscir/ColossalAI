import math
from typing import Optional

import loralib as lora
import torch
import torch.nn as nn
import torch.nn.functional as F

# 顾名思义 这个所有的大模型的训练都是训这个LoRA吗 好像是的


class LoraLinear(lora.LoRALayer, nn.Module):  # 看看定义的一个线性的层
    """Replace in-place ops to out-of-place ops to fit gemini. Convert a torch.nn.Linear to LoraLinear.
    """

    def __init__(
        self,
        weight: nn.Parameter,
        bias: Optional[nn.Parameter],
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False,    # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
    ):
        nn.Module.__init__(self)
        lora.LoRALayer.__init__(self,
                                r=r,
                                lora_alpha=lora_alpha,
                                lora_dropout=lora_dropout,
                                merge_weights=merge_weights)
        self.weight = weight  # 预训练模型的权重
        self.bias = bias

        out_features, in_features = weight.shape
        self.in_features = in_features
        self.out_features = out_features

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))  # 初始化LoRA层的大小
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()  # 按照LoRA论文里的方式初始化
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):  # 前面用凯明这种随机初始化,后面是0初始化
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # 前面用凯明这种随机初始化,后面是0初始化
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):  # 有点懵逼这里的train

        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Module.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

    def eval(self):

        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Module.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                delattr(self, 'lora_A')   # 直接删除实例属性
                delattr(self, 'lora_B')
            self.merged = True

    def forward(self, x: torch.Tensor):  # lora层的前向传播

        def T(w):
            return w.T if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result = result + (self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


def lora_linear_wrapper(linear: nn.Linear, lora_rank: int) -> LoraLinear:  # 把一个线性层转换成lora的线性层
    assert lora_rank <= linear.in_features, f'LoRA rank ({lora_rank}) must be less than or equal to in features ({linear.in_features})'
    lora_linear = LoraLinear(linear.weight, linear.bias, r=lora_rank, merge_weights=False)
    return lora_linear


def convert_to_lora_recursively(module: nn.Module, lora_rank: int) -> None:  # 传进来的就是一个原来的层，lora_rank这个整数参数
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, lora_linear_wrapper(child, lora_rank))
        else:
            convert_to_lora_recursively(child, lora_rank)   # 递归的处理下一个


class LoRAModule(nn.Module):  # 把这个作为模型的基你说得有多牛逼
    """A LoRA module base class. All derived classes should call `convert_to_lora()` at the bottom of `__init__()`.
    This class will convert all torch.nn.Linear layer to LoraLinear layer. 把普通的线性层转换为lora的线性层

    Args:
        lora_rank (int, optional): LoRA rank. 0 means LoRA is not applied. Defaults to 0.
        lora_train_bias (str, optional): Whether LoRA train biases.
            'none' means it doesn't train biases. 'all' means it trains all biases. 'lora_only' means it only trains biases of LoRA layers.
            Defaults to 'none'.
    """

    def __init__(self, lora_rank: int = 0, lora_train_bias: str = 'none') -> None:
        super().__init__()
        self.lora_rank = lora_rank  # 一个数字 序号？还是数量？
        self.lora_train_bias = lora_train_bias  # 这个Bias是什么意思

    def convert_to_lora(self) -> None:
        if self.lora_rank <= 0:
            return
        convert_to_lora_recursively(self, self.lora_rank)  # 这个会一步步调用前面写的lora linear 递归的转换lora
        lora.mark_only_lora_as_trainable(self, self.lora_train_bias)  # 顾名思义只训练Lora的参数
