import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev = input_dim
        
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers) 
    
    def forward(self, x):
        return self.net(x)
    
    
# def train_model(X, Y, output_dim, 
#                 hidden_dims=[128,64], epochs=1500, lr=1e-3,
#                 loss_fn=None, plot=True, use_deepsets=False):
#     # Split
#     Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=0)
    
#     # Convert to torch
#     Xt = torch.tensor(Xtr, dtype=torch.float32)
#     yt = torch.tensor(Ytr, dtype=torch.long)
    
    # Call model
    # model = MLP(
    #     input_dim = X.shape[1],
    #     hidden_dims = hidden_dims,
    #     output_dim = output_dim
    # )


class TransformerEnc(nn.Module):
    def __init__(self, F, output_dim, max_T,
                 d_model=64, n_head=4, n_layers=2, d_ff=64, dropout=0.0):
        super().__init__()
        self.goal_token = nn.Parameter(torch.zeros(1,1,F))
        self.in_proj = nn.Linear(F, d_model)
        self.pos_emb = nn.Embedding(max_T, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_ff, 
                                               dropout=dropout, batch_first=True)

        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out = nn.Linear(d_model, output_dim)
        
    def forward(self, x):  # x is shape [B, T, F]
        # x: (B, T, F). For one example in batch, it looks like: 
        # [ onehot(A), onehot(A->B), onehot(B->C), onehot(C) ]  (goal is last)
        B, T, F = x.shape
        
        # marks padded rows; real rows like onehot(A) are nonzero
        pad_mask = (x.abs().sum(dim=-1) == 0)  # (B, T) 
        
        # learned vector that will represent the "<goal>" token
        goal_marker = self.goal_token.expand(B, 1, F)  # (B, 1, F)
        
        # insert "<goal>" right before the last token, so:
        # [A, A->B, B->C, C]  ->  [A, A->B, B->C, <goal>, C]
        x = torch.cat([x[:, :-1, :], goal_marker, x[:, -1:, :]], dim=1)  # (B, T+1, F)
        
        # update padding mask to match x; "<goal>" is never padding
        pad_mask = torch.cat(  
            [pad_mask[:, :-1],                                        # padding flags for [A, A->B, B->C]
             torch.zeros(B, 1, device=x.device, dtype=torch.bool),    # padding flag for "<goal>" = False
             pad_mask[:, -1:]],                                       # padding flag for [C]
            dim=1
        ) # (B, T+1) 
        # print(f'x.size(1) is .... {x.size(1)}')
        # project each token (A, A->B, B->C, <goal>, C) from F-dim to d_model-dim ... add pos enc...
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        h = self.in_proj(x) + self.pos_emb(pos)   # (B, T+1, d_model) 
        
        # self-attention lets tokens interact (e.g. A attends to A->B, B->C, goal C, etc.)
        h = self.encoder(h, src_key_padding_mask=pad_mask) 
        goal_h = h[:, -2, :] # <goal> 
        
        return self.out(goal_h) 


class TransformerEnc2(nn.Module):
    def __init__(self, F, output_dim, max_T,
                 d_model=64, n_head=4, n_layers=2, d_ff=64, dropout=0.0):
        super().__init__()
        self.in_proj = nn.Linear(F, d_model)
        self.pos_emb = nn.Embedding(max_T, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_ff, 
                                               dropout=dropout, batch_first=True)

        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out = nn.Linear(d_model, output_dim)
        
    def forward(self, x):  # x is shape [B, T, F]
        # x: (B, T, F). For one example in batch, it looks like: 
        # [ onehot(A), onehot(A->B), onehot(B->C), onehot(C) ]  (goal is last)
        B, T, F = x.shape
        
        # marks padded rows; real rows like onehot(A) are nonzero
        pad_mask = (x.abs().sum(dim=-1) == 0)  # (B, T) 

        # project each token (A, A->B, B->C, C) from F-dim to d_model-dim ... add pos enc...
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)      # (1, T)
        pos_emb = self.pos_emb(pos).clone()                              # (1, T, d_model)
        pos_emb[:, :-1, :] = 0
        h = self.in_proj(x) + pos_emb                                    # (B, T, d_model) 
        
        # self-attention lets tokens interact (e.g. A attends to A->B, B->C, goal C, etc.)
        h = self.encoder(h, src_key_padding_mask=pad_mask) 
        goal_h = h[:, -1, :]   
        
        return self.out(goal_h) 








    
def train_model(X, Y, output_dim, batch_size=512,
                hidden_dims=[128,64], epochs=1500, lr=1e-3,
                loss_fn=None, plot=True, use_deepsets=False, use_transformer=False,
               printevery=200):
    # Split
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=0)
    # Take from np arrays, to not copy data unnecessarily
    Xtr = np.asarray(Xtr, dtype=np.float32)
    Ytr = np.asarray(Ytr, dtype=np.int64)
    Xt  = torch.from_numpy(Xtr)   # no copy
    yt  = torch.from_numpy(Ytr)

    
    
    if use_deepsets:
        model = DeepSetReasoner(
            n_formulas=output_dim,
            d_embed=64,
            hidden_dims=hidden_dims,
        )
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        
    elif use_transformer: 
        F = X.shape[-1]  # Here, X should be  (N, T, F)
        model = TransformerEnc2(F=F, output_dim=output_dim, max_T=12)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
        
    else:
        model = MLP(
            input_dim=X.shape[1],
            hidden_dims=hidden_dims,
            output_dim=output_dim,
        )
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Train
    losses = []; batch_size=batch_size;
    for epoch in range(epochs):
        perm = torch.randperm(len(Xt))
        total, n = 0.0, 0
        for s in range(0, len(Xt), batch_size):
            idx = perm[s:s+batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            n += len(idx)

        epoch_loss = total / n
        if epoch % printevery == 0:
            print(epoch, epoch_loss)
        losses.append(epoch_loss)
    if plot==True: plt.plot(losses)
        
    return model, (Xte, Yte)
            
        
        
        
class DeepSetReasoner(nn.Module):
    def __init__(self, n_formulas, d_embed=64, hidden_dims=[128,64]):
        super().__init__()
        self.n_formulas = n_formulas
        
        # φ: formula embeddings 
        self.formula_emb = nn.Embedding(n_formulas, d_embed)

        # ψ: goal embedding from one-hot
        self.goal_proj = nn.Linear(n_formulas, d_embed, bias=False)

        
        layers = []
        in_dim = 2 * d_embed
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_formulas))
        self.mlp = nn.Sequential(*layers) 
        
        

    def forward(self, X):
        
        # X = [x_state, x_goal]  with shape (N_tr, 2*n_formulas)
        x_state = X[:, :self.n_formulas]      # (N_tr, n_formulas)
        x_goal  = X[:, self.n_formulas:]      # (N_tr, n_formulas)
        
        # formula_emb.weight: (n_formulas, d_embed)
        E = self.formula_emb.weight              # embeddings for all formulas

        # permutation-invariant sum over formulas in the state
        state_emb = x_state @ E               # (N_tr, d_embed)
        goal_emb = self.goal_proj(x_goal)     # (N_tr, d_embed)
        
        # combine and predict next rule
        h = torch.cat([state_emb, goal_emb], dim=-1)  # (N_tr, 2*d_embed)
        logits = self.mlp(h)                     # (N_tr, n_formulas)
        return logits

        
        
        
        
        
        
        
        
        
        
        
        
        