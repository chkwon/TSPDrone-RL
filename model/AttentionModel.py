import torch
import numpy as np
from torch import nn
import math
from model.graph_encoder import GraphAttentionEncoder
from typing import NamedTuple
from torch.nn import DataParallel


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class AttentionModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 n_encode_layers=3,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None):
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = "sampling"
        self.temp = 1.0
       

        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

 
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size

        
        node_dim = 2  # x, y
            
 
        self.init_embed = nn.Linear(node_dim, embedding_dim).to(device)

        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        ).to(device)

        
        
    def embed(self, static):
        # encoder 
        embeddings, _ = self.embedder(self._init_embed(static))
     #   fixed = self._precompute(embeddings)
        return embeddings
    