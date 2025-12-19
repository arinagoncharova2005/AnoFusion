import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Prefer CUDA, then MPS (Apple), otherwise CPU.
def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


device = select_device()

class GTLayer(nn.Module):
    def __init__(self, in_channels, out_channels, first=True):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        if self.first == True:
            self.layer1 = GTAggregation(in_channels, out_channels)
            self.layer2 = GTAggregation(in_channels, out_channels)
        else:
            self.layer1 = GTAggregation(in_channels, out_channels)

    def forward(self, A, H_=None):
        if self.first == True:
            a = self.layer1(A)
            batch = a.shape[0]
            num_node = a.shape[-1]
            b = self.layer2(A)
            a = a.view((-1, a.shape[-2], a.shape[-1]))
            b = b.view((-1, b.shape[-2], b.shape[-1]))
            H = torch.bmm(a, b)
            H = H.view((batch, -1, num_node, num_node))
            W = [(F.softmax(self.layer1.weight, dim=2)).detach(), (F.softmax(self.layer2.weight, dim=2)).detach()]
        else:
            a = self.layer1(A)
            batch = a.shape[0]
            num_node = a.shape[-1]
            a = a.view((-1, a.shape[-2], a.shape[-1]))
            H_ = H_.view(-1, H_.shape[-2], H_.shape[-1])
            H = torch.bmm(H_, a)
            H = H.view((batch, -1, num_node, num_node))
            W = [(F.softmax(self.layer1.weight, dim=2)).detach()]
        return H, W


class GTAggregation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GTAggregation, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(1, out_channels, in_channels, 1, 1))

    def forward(self, A):
        self.weight = self.weight.to(device)
        A = torch.sum(A * F.softmax(self.weight, dim=2), dim=2)
        return A
