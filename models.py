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
    
    
def train_model(X, Y, output_dim, 
                hidden_dims=[128,64], epochs=1500, lr=1e-3,
                loss_fn=None, plot=True, use_deepsets=False):
    # Split
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=0)
    
    # Convert to torch
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(Ytr, dtype=torch.long)
    
    # Call model
#     model = MLP(
#         input_dim = X.shape[1],
#         hidden_dims = hidden_dims,
#         output_dim = output_dim
#     )
    if use_deepsets:
        model = DeepSetReasoner(
            n_formulas=output_dim,
            d_embed=64,
            hidden_dims=hidden_dims,
        )
    else:
        model = MLP(
            input_dim=X.shape[1],
            hidden_dims=hidden_dims,
            output_dim=output_dim,
        )
    
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Train
    losses = []
    for epoch in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()
        
        if epoch % 200 == 0:
            print(epoch, loss.item())
        losses.append(loss.item())
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
        
        

    def forward(self, x_state, x_goal):
        
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

        
        
        
        
        
        
        
        
        
        
        
        
        