import torch
import torch.nn.functional as F
from .functional.conv import conv2d
from climatestateclassifier.model.encoder import bound_pad
from climatestateclassifier import config as cfg


class Conv2d(torch.nn.Conv2d):
    def _conv_forward_explain(self, input, weight, conv2d_fn, **kwargs):
        p = kwargs.get('pattern')
        if cfg.global_padding:
            input = bound_pad(input, self._reversed_padding_repeated_twice)
            if p is not None:
                return conv2d_fn(input, weight, self.bias, self.stride,
                                 0, self.dilation, self.groups, p)
            else:
                return conv2d_fn(input, weight, self.bias, self.stride,
                                 0, self.dilation, self.groups)

        elif self.padding_mode != 'zeros':
            return conv2d_fn(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                             weight, self.bias, self.stride,
                             (2, 2), self.dilation, self.groups, **kwargs)

        if p is not None:
            return conv2d_fn(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups, p)
        else:
            return conv2d_fn(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def forward(self, input, explain=False, rule="epsilon", **kwargs):
        if cfg.global_padding:
            input = bound_pad(input, 2 * (2, 2))
        else:
            input = F.pad(input, 2 * (2, 2))
        if not explain:
            return super(Conv2d, self).forward(input)
        return self._conv_forward_explain(input, self.weight, conv2d[rule], **kwargs)

    @classmethod
    def from_torch(cls, conv):
        in_channels = conv.weight.shape[1] * conv.groups
        bias = conv.bias is not None

        module = cls(in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, conv.dilation,
                     conv.groups, bias=bias, padding_mode=conv.padding_mode)

        module.load_state_dict(conv.state_dict())

        return module
