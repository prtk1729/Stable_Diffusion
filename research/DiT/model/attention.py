import torch
import torch.nn as nn
from einops import rearrange


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_size = config["hidden_size"]        
        self.num_heads = config["num_heads"] 
        self.head_dim = config["head_dim"] 

        self.attn_dim = num_heads * head_dim

        self.q_proj = nn.Linear( self.hidden_size, self.attn_dim, bias = True )
        self.k_proj = nn.Linear( self.hidden_size, self.attn_dim, bias = True )
        self.v_proj = nn.Linear( self.hidden_size, self.attn_dim, bias = True )
        self.o_proj = nn.Linear( self.attn_dim, self.hidden_size, bias = True )


        #### Init for faster training, prevent vanishing or exploding grad issues
        #### due to random small init, previous garbage values on gpu when uninit
        # zero outr all bias => training doesn't start with weird biases rather learn them
        nn.init.constant_(self.q_proj.bias, val = 0)
        nn.init.constant_(self.k_proj.bias, val = 0)
        nn.init.constant_(self.v_proj.bias, val = 0)
        nn.init.constant_(self.o_proj.bias, val = 0)

        # Weights init -> Use Glorot init
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)        


    def forward(self, x):
        # x: patch_embs + pos_embs
        batch, num_patches, hidden_size = x.shape

        # 1st project them to attn_dim
        # (bsz, num_patches, hijdden_size) -> (bsz, num_patches, attn_sim)
        key, query, value = self.k_proj(x), self.q_proj(x), self.v_proj(x)

        # Operate on a fixed p[ortion of attn_dim parallely, by splitting into heads
        key = rearrange(key, pattern = "b n (n_heads head_dim) -> b n_heads n head_dim",
                  n_heads = self.num_heads,
                  head_dim = self.head_dim)
        query = rearrange(query, pattern = "b n (n_heads head_dim) -> b n_heads n head_dim",
                  n_heads = self.num_heads,
                  head_dim = self.head_dim)
        value = rearrange(value, pattern = "b n (n_heads head_dim) -> b n_heads n head_dim",
                  n_heads = self.num_heads,
                  head_dim = self.head_dim)


        #### Compute Attention Weights
        # (bsz, num_heads, num_patches, head_dim) @ (bsz, num_heads, head_dim, num_patches)
        attn_weights = torch.matmul( query, key.transpose(-2, -1) ) / (self.head_dim ** (-0.5))
        # (bsz, n_heads, num_patches, num_patches)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        ##############################

        ######## Weighted Value Computation 
        # How much weight should be assoc to the values, to predict the next token
        # (bsz, n_heads, num_pathces, num_patches) @ (bsz, n_heads, num_pathces, head_dim) 
        # ->  (bsz, n_heads, num_pathces, head_dim) 
        attn_score = torch.matmul(attn_weights, value)
        ##############################

        #### Mix all the independent head_learnings to overall learning
        # Prepare for o_proj
        attn_score = rearrange(attn_score, pattern = "b n_heads n h_dim -> b n (h_dim n_heads)" )
        out = self.o_proj(attn_score)
        ######################################

        # return out, attn_weights # Sometimes returned to visualise the attn_weights in segmentation problems
        return out


if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    bsz, num_patches, hidden_size = 2, 196, 768
    x = torch.randn( (bsz, num_patches, hidden_size) ).to(device)

    num_heads, head_dim = 4, 1024
    config = dict()
    config["num_heads"] = num_heads
    config["head_dim"] = head_dim
    config["hidden_size"] = hidden_size

    model = Attention(config).to(device)
    out = model(x)
    print(out.shape)