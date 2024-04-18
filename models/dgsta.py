from .st_att_layer import *
import torch.nn as nn
import torch



class DG_STA(nn.Module):
    def __init__(self, window_size: int, out_dim: int, dp_rate: float):
        super(DG_STA, self).__init__()

        self.proj_dim = 128

        self.input_map = nn.Sequential(
            nn.Linear(3, self.proj_dim),
            nn.ReLU(),
            LayerNorm(self.proj_dim),
            nn.Dropout(dp_rate),
        )

        h_dim = 32
        h_num = 1

        # spatial attention
        self.s_att = ST_ATT_Layer(self.proj_dim, self.proj_dim, h_num, h_dim, dp_rate, "spatial", window_size)

        # temporal attention
        self.t_att = ST_ATT_Layer(self.proj_dim, self.proj_dim, h_num, h_dim, dp_rate, "temporal", window_size)

        # linear projection
        self.cls = nn.Linear(self.proj_dim, out_dim)


    def forward(self, x):
        # input shape: [batch_size, time_len, joint_num, 3]

        

        bs, ws, feat_dim = x.shape

        x = x.reshape((bs, ws, 21, 3))

        time_len = x.shape[1]
        joint_num = x.shape[2]

        x = x.reshape(-1, time_len * joint_num, 3)

        x = self.input_map(x)
        x = self.s_att(x)
        x = self.t_att(x)

        x = x.sum(1) / x.shape[1]
        out = self.cls(x)

        return out