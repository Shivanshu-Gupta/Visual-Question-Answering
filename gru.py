import math
import torch
import torch.nn.functional as F
from torch import matmul
from torch.nn import Parameter
from torch.nn.modules.rnn import RNNCellBase


class CustomGRUCell(RNNCellBase):
    r"""A custom gated recurrent unit (GRU) cell
    .. math::
        \begin{array}{ll}
        r = \mathrm{sigmoid}(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
        z = \mathrm{sigmoid}(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
        n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
        h' = (1 - z) * n + z * h
        \end{array}
    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        bias: If `False`, then the layer does not use bias weights `b_ih` and
            `b_hh`. Default: `True`
    Inputs: input, hidden
        - **input** (batch, input_size): tensor containing input features
        - **hidden** (batch, hidden_size): tensor containing the initial hidden
          state for each element in the batch.
    Outputs: h'
        - **h'**: (batch, hidden_size): tensor containing the next hidden state
          for each element in the batch
    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(3*hidden_size x input_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(3*hidden_size x hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(3*hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(3*hidden_size)`
    Examples::
        >>> from gru import CustomGRUCell
        >>> rnn = CustomGRUCell(10, 20)
        >>> input = Variable(torch.randn(6, 3, 10))
        >>> hx = Variable(torch.randn(3, 20))
        >>> output = []
        >>> for i in range(6):
        ...     hx = rnn(input[i], hx)
        ...     output.append(hx)
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(CustomGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(3 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        wih = self.weight_ih
        bih = self.bias_ih
        whh = self.weight_hh
        bhh = self.bias_hh
        dim_h = self.hidden_size

        bih = bih.expand(hx.size(0), 3 * dim_h)    # batch_size * input_size
        bhh = bhh.expand(hx.size(0), 3 * dim_h)    # batch_size * hidden_size
        # reset gate - batch_size * hidden_size
        # r = \mathrm{sigmoid}(W_{ir} x + b_{ir} + W_{hr} h + b_{hr})
        r = F.sigmoid(matmul(input, wih[:dim_h].t()) + bih[:, :dim_h] +
                      matmul(hx, whh[:dim_h].t()) + bhh[:, :dim_h])

        # update gate - batch_size * hidden_size
        # z = \mathrm{sigmoid}(W_{iz} x + b_{iz} + W_{hz} h + b_{hz})
        z = F.sigmoid(matmul(input, wih[dim_h:2 * dim_h].t()) + bih[:, dim_h:2 * dim_h] +
                      matmul(hx, whh[dim_h:2 * dim_h].t()) + bhh[:, dim_h:2 * dim_h])

        # n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn}))
        n = F.tanh(matmul(input, wih[2 * dim_h:].t()) + bih[:, 2 * dim_h:] +
                   matmul(r * hx, whh[2 * dim_h:].t()) + bhh[:, 2 * dim_h:])

        # h' = (1 - z) * n + z * h
        output = (1 - z) * n + z * hx

        return output
