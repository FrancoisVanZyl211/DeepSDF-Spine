# model_multishape.py
import types
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class MultiShapeDecoder(nn.Module):
    """
    DeepSDF decoder that supports:
      • shape–conditioned forward:  forward(x, shape_id)
      • latent-vector forward:      forward_with_latent(x, z)

    `args` is optional so GUI widgets can instantiate the model without
    building an argparse.Namespace first.  Training scripts can still pass
    their full arg-struct unchanged.
    """
    def __init__(self,
                 num_shapes: int,
                 args: object | None = None,
                 latent_dim: int = 64,
                 dropout_prob: float = 0.1):
        super().__init__()

        if args is None:
            args = types.SimpleNamespace(hidden_dim=512,
                                         dropout=dropout_prob,
                                         skip=True)
        self.cfg = args                     

        # ------------------------------------------------------------------
        # layers
        # ------------------------------------------------------------------
        self.latent_codes = nn.Embedding(num_shapes, latent_dim)
        nn.init.normal_(self.latent_codes.weight, mean=0.0, std=0.01)

        h = args.hidden_dim                  
        self.dropout = nn.Dropout(args.dropout)
        self.prelu   = nn.PReLU()
        self.tanh    = nn.Tanh()

        self.fc1 = weight_norm(nn.Linear(3 + latent_dim, h))
        self.fc2 = weight_norm(nn.Linear(h, h))
        self.fc3 = weight_norm(nn.Linear(h, h))
        self.fc4 = weight_norm(nn.Linear(h, h - 3))   

        self.fc5 = weight_norm(nn.Linear(h, h))
        self.fc6 = weight_norm(nn.Linear(h, h))
        self.fc7 = weight_norm(nn.Linear(h, h))
        self.fc8 = nn.Linear(h, 1)

    # ----------------------------------------------------------------------
    # forward with shape id
    # ----------------------------------------------------------------------
    def forward(self, coords: torch.Tensor, shape_id: torch.Tensor):
        """
        coords   : [N,3]  xyz in bounding box
        shape_id : [N]    long tensor; each coord row’s shape index
        """
        z = self.latent_codes(shape_id)         
        xz = torch.cat((coords, z), dim=1)     
        x  = self._mlp(xz, coords)
        return self.tanh(x)

    # ----------------------------------------------------------------------
    # forward with explicit latent vector
    # ----------------------------------------------------------------------
    def forward_with_latent(self,
                            coords: torch.Tensor,
                            z: torch.Tensor):
        """
        coords : [N,3]
        z      : [N,latent_dim] or [1,latent_dim] – will broadcast if needed
        """
        if z.shape[0] == 1:
            z = z.expand(coords.size(0), -1)
        xz = torch.cat((coords, z), dim=1)
        x  = self._mlp(xz, coords)
        return self.tanh(x)

    # ----------------------------------------------------------------------
    # internal MLP with skip connection
    # ----------------------------------------------------------------------
    def _mlp(self, xz: torch.Tensor, coords: torch.Tensor):
        x = self.dropout(self.prelu(self.fc1(xz)))
        x = self.dropout(self.prelu(self.fc2(x)))
        x = self.dropout(self.prelu(self.fc3(x)))
        x = self.dropout(self.prelu(self.fc4(x)))
        x = torch.cat((x, coords), dim=1)       
        x = self.dropout(self.prelu(self.fc5(x)))
        x = self.dropout(self.prelu(self.fc6(x)))
        x = self.dropout(self.prelu(self.fc7(x)))
        x = self.fc8(x)
        return x