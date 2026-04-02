
# src/models/heads.py

import torch.nn as nn

class HeadLogistic(nn.Module):
    """
    Cabeza de regresión logística binaria: Linear -> 1 logit
    """
    def __init__(self, in_feats: int):
        super().__init__()
        self.fc = nn.Linear(in_feats, 1)

    def forward(self, x):
        return self.fc(x).squeeze(1)

class HeadMLP(nn.Module): # me gustaría que para el tuning esto se pueda tocar mas desde el init
    """
    Cabeza MLP binaria: Linear -> SiLU -> Dropout -> Linear -> 1 logit
    """
    def __init__(self, in_feats: int, hidden: int, p_drop: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feats, hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)