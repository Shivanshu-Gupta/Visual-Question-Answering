# reference: https://github.com/jihunchoi/recurrent-batch-normalization-pytorch/blob/master/bnlstm.py
import torch
import torch.nn as nn
from torch import autograd


class customRNN(nn.Module):

    """A module that runs multiple steps of LSTM."""

    def __init__(self, cell_class, input_size, hidden_size,
                 use_bias=True, batch_first=False, **kwargs):
        super(customRNN, self).__init__()
        self.cell_class = cell_class
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.batch_first = batch_first

        self.cell = cell_class(input_size=input_size,
                               hidden_size=hidden_size,
                               **kwargs)
        self.cell.reset_parameters()

    # def _forward_rnn(self, cell, input_, ranges, lengths, hx):
    #     max_time, batch_size, _ = input_.size()
    #     output = [[] for i in range(batch_size)]
    #     # print(input_.size())
    #     curr = 0
    #     # print(ranges)
    #     for time in range(max_time):
    #         beg = ranges[curr][0]
    #         end = ranges[curr][1]
    #         assert(input_[time].size(0) == hx[0].size(0))
    #         hx = cell(input=input_[time], hx=hx)
    #         if isinstance(cell, nn.LSTMCell):
    #             for idx in range(beg, end):
    #                 output[idx].append(hx[0][idx - beg])
    #         else:
    #             for idx in range(beg, end):
    #                 output[idx].append(hx[idx - beg])
    #         if time == lengths[beg] - 1 and time != max_time - 1:
    #             curr += 1
    #             input_ = input_[ranges[curr][0] - beg:]
    #             if isinstance(cell, nn.LSTMCell):
    #                 h_next = hx[0][ranges[curr][0] - beg:]
    #                 c_next = hx[0][ranges[curr][0] - beg:]
    #                 hx = (h_next, c_next)
    #             else:
    #                 h_next = cell(input=input_[time], hx=hx)
    #                 h_next = h_next[ranges[curr][0] - beg:]
    #                 hx = h_next
    #     output = [torch.stack(sentence_out, 0) for sentence_out in output]
    #     output = torch.cat(output, 0)
    #     return output, hx

    # def _forward_rnn(self, cell, input_, length, hx):
    #     max_time = input_.size(0)
    #     output = []
    #     for time in range(max_time):
    #         if isinstance(cell, nn.LSTMCell):
    #             h_next, c_next = cell(input_=input_[time], hx=hx, time=time)
    #             mask = (time < length).float().unsqueeze(1).expand_as(h_next)
    #             h_next = h_next * mask + hx[0] * (1 - mask)
    #             c_next = c_next * mask + hx[1] * (1 - mask)
    #             hx_next = (h_next, c_next)
    #         else:
    #             h_next = cell(input_=input_[time], hx=hx)
    #             mask = (time < length).float().unsqueeze(1).expand_as(h_next)
    #             h_next = h_next * mask + hx[0] * (1 - mask)
    #             hx_next = h_next

    #         output.append(h_next)
    #         hx = hx_next
    #     output = torch.stack(output, 0)

    def _forward_rnn_no_mask(self, cell, input_, hx):
        max_time = input_.size(0)
        output = []
        for time in range(max_time):
            if isinstance(cell, nn.LSTMCell):
                h_next, c_next = cell(input=input_[time], hx=hx)
                hx = (h_next, c_next)
            else:
                h_next = cell(input=input_[time], hx=hx)
                hx = h_next
            output.append(h_next)
        output = torch.cat(output, 0)
        # output = torch.stack(output, 0)     # do this if want 3D tensor
        return output, hx

    def forward(self, input_, ranges, lengths, hx=None):
        if self.batch_first:
            input_ = input_.transpose(0, 1)
        max_time, batch_size, _ = input_.size()
        if hx is None:
            hx = autograd.Variable(input_.data.new(batch_size, self.hidden_size).zero_(),
                                   requires_grad=False)
            if self.cell_class == nn.LSTMCell:
                hx = (hx, hx)
        cell = self.cell
        # output, h_n = self._forward_rnn(cell=cell, input_=input_, ranges=ranges, lengths=lengths, hx=hx)
        output, h_n = self._forward_rnn_no_mask(cell=cell, input_=input_, hx=hx)
        return output, h_n
