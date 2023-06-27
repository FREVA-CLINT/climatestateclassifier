import torch
from .functional import lstm


class LSTMCell(torch.nn.LSTMCell):
    def forward(self, input, explain=False, rule="epsilon", **kwargs):
        if not explain:
            return super(LSTMCell, self).forward(input)

        p = kwargs.get('pattern')
        if p is not None:
            return lstm[rule](input, self.weight, self.bias, p)
        else:
            return lstm[rule](input, self.weight, self.bias)

    @classmethod
    def from_torch(cls, lstm):
        bias = lstm.bias is not None
        module = cls(in_features=lin.in_features, out_features=lin.out_features, bias=bias)
        module.load_state_dict(lstm.state_dict())

        return module
