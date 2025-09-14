import torch
from torch import nn

class AFTFull(nn.Module):
    def __init__(self, max_len, dim, hid_dim=32):
        super().__init__()
        self.max_len = max_len
        self.dim = dim          # token 的节点数
        self.hid_dim = hid_dim  # 隐层节点数
        self.wq = nn.Linear(self.dim, self.hid_dim)
        self.wk = nn.Linear(self.dim, self.hid_dim)
        self.wv = nn.Linear(self.dim, self.hid_dim)
        self.ffnn = nn.Linear(self.hid_dim, self.dim)
        self.w = nn.Parameter(torch.Tensor(max_len, max_len))
        nn.init.xavier_uniform_(self.w)
    
    def forward(self, x):
        B, T, _ = x.shape
        Q = self.wq(x).view(B, T, self.hid_dim)
        K = self.wk(x).view(B, T, self.hid_dim)
        V = self.wv(x).view(B, T, self.hid_dim)
        Q_sig = torch.sigmoid(Q)
        temp = torch.exp(self.w) @ torch.mul(torch.exp(K), V)
        weighted = temp / (torch.exp(self.w) @ torch.exp(K))
        Yt = torch.mul(Q_sig, weighted)
        Yt = Yt.view(B, T, self.hid_dim)
        Yt = self.ffnn(Yt)
        return weighted