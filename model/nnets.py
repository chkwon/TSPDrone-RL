import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch import nn
import math
from model.graph_encoder import GraphAttentionEncoder
from typing import NamedTuple
from torch.nn import DataParallel


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

############################ Encoder ################################################

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

        
        
    def forward(self, static):
        # encoder 
        embeddings, _ = self.embedder(self._init_embed(static))
     #   fixed = self._precompute(embeddings)
        return embeddings
    
    def _init_embed(self, input):
        # compute initial embedding 
        return self.init_embed(input)

class Encoder(nn.Module):
    """Encodes the static & dynamic states using 1d Convolution."""

    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)

    def forward(self, input):
        output = self.conv(input)
        return output  # (batch, hidden_size, seq_len)

############################ Actor ###################################

class Attention(nn.Module):
    """Calculates attention over the input nodes given the current state."""

    def __init__(self, hidden_size, use_tahn=False, C = 10):
        super(Attention, self).__init__()
  
        self.use_tahn = use_tahn 
        # W processes features from static decoder elements
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))

        self.project_d = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1)
        
        self.project_ref = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1)
        self.project_query = nn.Linear(hidden_size, hidden_size)
        self.C = C

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden):
        # [b_s, hidden_dim, n_nodes]
      
        d_ex = self.project_d(dynamic_hidden)
          
        
        batch_size, hidden_size, n_nodes = static_hidden.size()
      
        # [b_s, hidden_dim, n_nodes]
        e = self.project_ref(static_hidden)
        decoder_hidden = self.project_query(decoder_hidden)
       
        # Broadcast some dimensions so we can do batch-matrix-multiply
        v = self.v.expand(batch_size, 1, hidden_size)
        q = decoder_hidden.view(batch_size, hidden_size, 1).expand(batch_size, hidden_size, n_nodes)
       
       
        u = torch.bmm(v, torch.tanh(e + q + d_ex )).squeeze(1)
       
            
        if self.use_tahn:
            logits = self.C * self.tanh(u)
        else:
            logits = u 
        # e : [b_s, hidden_dim, n_nodes]
        # logits : [b_s, n_nodes]
        return e, logits 
    
    
class Decoder(nn.Module):
    """Calculates the next state given the previous state and input embeddings."""

    def __init__(self, hidden_size, num_layers=1, dropout=0.1, n_glim=0):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, bias = False, 
                        batch_first=True,bidirectional=False, dropout=dropout if num_layers > 1 else 0)

        self.encoder_attn = Attention(hidden_size)
        if torch.cuda.is_available():
            self.lstm = self.lstm.cuda()
            self.encoder_attn = self.encoder_attn.cuda()
        self.drop_rnn = nn.Dropout(p=dropout)
        self.drop_hh = nn.Dropout(p=dropout)
        

    def forward(self, static_hidden, dynamic_hidden, decoder_input, last_hh):
        # decoder_input: [b_s, hidden_dim, 1]
        # rnn_out : [b_s, hidden_dim]
        # last_hh : [num_layers, b_s, hidden_dim]
        
        rnn_out, last_hh = self.lstm(decoder_input.transpose(2, 1), last_hh)
        rnn_out = rnn_out.squeeze(1)

        # Always apply dropout on the RNN output
        rnn_out = self.drop_rnn(rnn_out)
        if self.num_layers == 1:
            # If > 1 layer dropout is already applied
            hx = self.drop_hh(last_hh[0]) 
            cx = self.drop_hh(last_hh[1]) 
            last_hh = (hx, cx)
        #[b_s, hidden_dim]
        hy = last_hh[0].squeeze(0)
        
        # compute attention 
        _, logits = self.encoder_attn(static_hidden, dynamic_hidden, hy)
        
     
        return logits, last_hh


class Actor(nn.Module):
   
    def __init__(self, hidden_size,  
                 num_layers=1, dropout=0.1, mask_logits=True):
        super(Actor, self).__init__()
        
        
        self.mask_logits = mask_logits 
        # Define the encoder & decoder models
        # for static x, y coords 
        self.attention_encoder = AttentionModel(hidden_size, hidden_size)
        self.dynamic_d_ex = Encoder(1, hidden_size)
        self.decoder = Decoder(hidden_size, num_layers, dropout)
        if torch.cuda.is_available():
            self.attention_encoder = self.attention_encoder.cuda()
            self.dynamic_d_ex = self.dynamic_d_ex.cuda()
            self.decoder = self.decoder.cuda()
        self.logsoft = nn.LogSoftmax()
        self.Bignumber = 100000
        self.sample_mode = False

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)
                
    def emd_stat(self, static):
        
        return self.attention_encoder(static)


    def forward(self, static_hidden, dynamic, decoder_input, last_hh, terminated, avail_actions):
        
   
        dynamic_hidden = self.dynamic_d_ex(dynamic.permute(0, 2, 1))
       
        
       
       
        logits, last_hh = self.decoder(static_hidden, 
                                          dynamic_hidden,
                                          decoder_input, last_hh)
        if self.mask_logits:
            logits[avail_actions==0] = -self.Bignumber
        
        logprobs = self.logsoft(logits)
        probs = torch.exp(logprobs)
  
            
        if self.training or self.sample_mode:
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            logp = m.log_prob(action)
        else:
            prob, action = torch.max(probs, 1)  # Greedy
            logp = prob.log()

              
        logp = logp * (1. - terminated)

        return action, probs, logp, last_hh 

    def set_sample_mode(self, value):
        self.sample_mode = value 
    
############################### Critic ##########################################

class AttentionCritic(nn.Module):
    """Calculates attention over the input nodes given the current state."""

    def __init__(self, hidden_size, use_tahn=False, C = 10):
        super(AttentionCritic, self).__init__()
        self.use_tahn = use_tahn 
        # W processes features from static decoder elements
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))

        self.project_d_ex = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1)
     #   self.project_ch_l = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1)
        self.project_ref = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1)
        self.project_query = nn.Linear(hidden_size, hidden_size)
        self.C = C

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden):
        # [b_s, hidden_dim, n_nodes]
        
        batch_size, hidden_size, n_nodes = static_hidden.size()
      
        # [b_s, hidden_dim, n_nodes]
        d_ex = self.project_d_ex(dynamic_hidden)

        # [b_s, hidden_dim, n_nodes]
        e = self.project_ref(static_hidden)
        # [b_s, hidden_dim]
        decoder_hidden = self.project_query(decoder_hidden)
        
        
        
        # Broadcast some dimensions so we can do batch-matrix-multiply
        v = self.v.expand(batch_size, 1, hidden_size)
        q = decoder_hidden.view(batch_size, hidden_size, 1).expand(batch_size, hidden_size, n_nodes)
        

        u = torch.bmm(v, torch.tanh(e + q + d_ex)).squeeze(1)
        if self.use_tahn:
            logits = self.C * self.tanh(u)
        else:
            logits = u 
        # e : [b_s, hidden_dim, n_nodes]
        # logits : [b_s, n_nodes]
        
        return e, logits 

    
class Critic(nn.Module):
    """Estimates the problem complexity.
    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, hidden_size, n_processes=3, num_layers=1):
        super(Critic, self).__init__()
        
        self.hidden_size = hidden_size 
        self.num_layers = num_layers 
        self.dynamic_d_ex = Encoder(1, hidden_size)
   #     self.static_w = Encoder(1, hidden_size)
        self.static_encoder = Encoder(2, hidden_size)
        self.attention1 = AttentionCritic(hidden_size)
        self.attention2 = AttentionCritic(hidden_size)
        self.attention3 = AttentionCritic(hidden_size)
        if torch.cuda.is_available():
            self.dynamic_d_ex = self.dynamic_d_ex.cuda()
            self.static_encoder = self.static_encoder.cuda()
            self.attention1 = self.attention1.cuda()
            self.attention2 = self.attention2.cuda()
            self.attention3 = self.attention3.cuda()
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size).to(device)
        self.fc2 = nn.Linear(self.hidden_size, 1).to(device)
        

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, d_ex):
        # x, y coords 
        static_hidden = self.static_encoder(static)
        
        batch_size, _, n_nodes = static_hidden.size()
      
        dynamic_hidden = self.dynamic_d_ex(d_ex.permute(0, 2, 1))
  
        hx = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        hy = hx.squeeze(0)
        if torch.cuda.is_available():
            hy = hy.cuda()
        
        e, logits = self.attention1(static_hidden, dynamic_hidden, hy)
        probs = torch.softmax(logits, dim=1)
        
        # [b_s, hidden_dim] = [b_s, 1, n_nodes] * [b_s, n_nodes, hidden_dims]
        hy = torch.matmul(probs.unsqueeze(1), e.permute(0,2,1)).squeeze(1)
        e, logits = self.attention2(static_hidden, dynamic_hidden, hy)
        probs = torch.softmax(logits, dim=1)
        
        hy = torch.matmul(probs.unsqueeze(1), e.permute(0,2,1)).squeeze(1)
        e, logits = self.attention3(static_hidden, dynamic_hidden, hy)
        probs = torch.softmax(logits, dim=1)
        hy = torch.matmul(probs.unsqueeze(1), e.permute(0,2,1)).squeeze(1)
        
        out = F.relu(self.fc1(hy))
        out = self.fc2(out)
        
        return out 
    