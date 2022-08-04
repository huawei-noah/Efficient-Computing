# 2022.07.19-Implementation for building DQ model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import math
import time
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F


class Signer(Function):
    '''
    take a real value x
    output sign(x)
    '''
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def sign(input):
    return Signer.apply(input)



# quantize for weights and activations
class Quantizer(Function):
    '''
    take a real value x in alpha*[0,1] or alpha*[-1,1]
    output a discrete-valued x in alpha*{0, 1/(2^k-1), ..., (2^k-1)/(2^k-1)} or likeness
    where k is nbit
    '''
    @staticmethod
    def forward(ctx, input, nbit, alpha=None, offset=None):
        ctx.alpha = alpha
        ctx.offset = offset
        scale = (2 ** nbit - 1) if alpha is None else (2 ** nbit - 1) / alpha
        ctx.scale = scale
        return torch.round(input * scale) / scale if offset is None \
                else (torch.round(input * scale) + torch.round(offset)) / scale

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.offset is None:
            return grad_output, None, None, None
        else:
            return grad_output, None, None, torch.sum(grad_output) / ctx.scale


def quantize(input, nbit, alpha=None, offset=None):
    return Quantizer.apply(input, nbit, alpha, offset)

# sign in dorefa-net for weights
class ScaleSigner(Function):
    '''
    take a real value x
    output sign(x) * E(|x|)
    '''
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input) * torch.mean(torch.abs(input))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

    
def scale_sign(input):
    return ScaleSigner.apply(input)

def dorefa_w(w, nbit_w, *args, **kwargs):
    if nbit_w == 1:
        w = scale_sign(w)
    else:
        w = torch.tanh(w)
        w = w / (2 * torch.max(torch.abs(w))) + 0.5
        w = 2 * quantize(w, nbit_w) - 1
    return w

# dorefa quantize for activations
def dorefa_a(input, nbit_a, *args, **kwargs):
    return quantize(torch.clamp(input, 0, 1), nbit_a, *args, **kwargs)

# PACT quantize for activations
def pact_a(input, nbit_a, alpha, *args, **kwargs):
    x = 0.5*(torch.abs(input)-torch.abs(input-alpha)+alpha)
    return quantize(x, nbit_a, alpha, *args, **kwargs)


class DynamicQConv(nn.Conv2d):
    # dynamic quantization for quantized conv
    def __init__(self, in_channels, out_channels, kernel_size, quan_name_w, quan_name_a, nbit_w=32, nbit_a=32, has_offset=False, 
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(DynamicQConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nbit_w = nbit_w
        self.nbit_a = nbit_a
        name_w_dict = {'dorefa': dorefa_w, 'pact': dorefa_w}
        name_a_dict = {'dorefa': dorefa_a, 'pact': pact_a}
        self.quan_w = name_w_dict[quan_name_w]
        self.quan_a = name_a_dict[quan_name_a]
        
        if quan_name_a == 'pact':
            self.alpha_a = nn.Parameter(torch.Tensor(1), requires_grad=True)
        else:
            self.register_parameter('alpha_a', None)
        if quan_name_w == 'pact':
            self.alpha_w = nn.Parameter(torch.Tensor(1), requires_grad=True)
        else:
            self.register_parameter('alpha_w', None)
        if has_offset:
            self.offset = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('offset', None)
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_custome_parameters()
    
    def reset_custome_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if self.alpha_a is not None:
            nn.init.constant_(self.alpha_a, 10)
        if self.alpha_w is not None:
            nn.init.constant_(self.alpha_w, 10)
        if self.offset is not None:
            nn.init.constant_(self.offset, 0)
    
    def forward(self, input, mask):
        # 0-bit: identity mapping
        if self.nbit_w == 0 or self.nbit_a == 0:
            diff_channels = self.out_channels - self.in_channels
            if self.stride == 2 or self.stride == (2, 2):
                x = F.pad(input[:, :, ::2, ::2], (0, 0, 0, 0, diff_channels//2, diff_channels-diff_channels//2), 'constant', 0)
                return x
            else:
                x = F.pad(input, (0, 0, 0, 0, diff_channels//2, diff_channels-diff_channels//2), 'constant', 0)
                return x
        # w quan
        if self.nbit_w < 32:
            w0 = self.quan_w(self.weight, self.nbit_w-1, self.alpha_w, self.offset)
            w1 = self.quan_w(self.weight, self.nbit_w, self.alpha_w, self.offset)
            w2 = self.quan_w(self.weight, self.nbit_w+1, self.alpha_w, self.offset)
        else:
            w = self.weight
        # a quan
        if self.nbit_a < 32:
            x0 = self.quan_a(input, self.nbit_a-1, self.alpha_a)
            x1 = self.quan_a(input, self.nbit_a, self.alpha_a)
            x2 = self.quan_a(input, self.nbit_a+1, self.alpha_a)
        else:
            x = F.relu(input)
        
        x0 = F.conv2d(x0, w0, None, self.stride, self.padding, self.dilation, self.groups)
        x1 = F.conv2d(x1, w1, None, self.stride, self.padding, self.dilation, self.groups)
        x2 = F.conv2d(x2, w2, None, self.stride, self.padding, self.dilation, self.groups)
        x = x0*mask[:,0].unsqueeze(1).unsqueeze(2).unsqueeze(3)+ \
            x1*mask[:,1].unsqueeze(1).unsqueeze(2).unsqueeze(3)+ \
            x2*mask[:,2].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return x


class QuanConv(nn.Conv2d):
    # general quantization for quantized conv
    def __init__(self, in_channels, out_channels, kernel_size, quan_name_w, quan_name_a, nbit_w=32, nbit_a=32, 
                 stride=1, padding=0, dilation=1, groups=1, bias=True, has_offset=False):
        super(QuanConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nbit_w = nbit_w
        self.nbit_a = nbit_a
        name_w_dict = {'dorefa': dorefa_w, 'pact': dorefa_w}
        name_a_dict = {'dorefa': dorefa_a, 'pact': pact_a}
        self.quan_w = name_w_dict[quan_name_w]
        self.quan_a = name_a_dict[quan_name_a]
        
        if quan_name_a == 'pact':
            self.alpha_a = nn.Parameter(torch.Tensor(1), requires_grad=True)
        else:
            self.register_parameter('alpha_a', None)
        if quan_name_w == 'pact':
            self.alpha_w = nn.Parameter(torch.Tensor(1), requires_grad=True)
        else:
            self.register_parameter('alpha_w', None)
        if has_offset:
            self.offset = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('offset', None)

        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_custome_parameters()
    
    def reset_custome_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if self.alpha_a is not None:
            nn.init.constant_(self.alpha_a, 10)
        if self.alpha_w is not None:
            nn.init.constant_(self.alpha_w, 10)
        if self.offset is not None:
            nn.init.constant_(self.offset, 0)
    
    def forward(self, input):
        # 0-bit: identity mapping
        if self.nbit_w == 0 or self.nbit_a == 0:
            diff_channels = self.out_channels - self.in_channels
            if self.stride == 2 or self.stride == (2, 2):
                x = F.pad(input[:, :, ::2, ::2], (0, 0, 0, 0, diff_channels//2, diff_channels-diff_channels//2), 'constant', 0)
                return x
            else:
                x = F.pad(input, (0, 0, 0, 0, diff_channels//2, diff_channels-diff_channels//2), 'constant', 0)
                return x
        # w quan
        if self.nbit_w < 32:
            w = self.quan_w(self.weight, self.nbit_w, self.alpha_w, self.offset)
        else:
            w = self.weight
        # a quan
        if self.nbit_a < 32:
            x = self.quan_a(input, self.nbit_a, self.alpha_a)
        else:
            x = F.relu(input)
        
        x = F.conv2d(x, w, None, self.stride, self.padding, self.dilation, self.groups)
        return x
