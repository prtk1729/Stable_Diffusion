import torch
import torch.nn as nn
from einops import rearrange

class Attention(nn.Module):
    def __init__(self, config):
        '''
            Unlike CLIP / SigLIP vision-towers, here:-
                - attn_dim = num_heads * head_dim is the projected space dim
        '''
        super().__init__()
        self.emb_dim = config['emb_dim']
        self.head_dim = config["head_dim"]
        self.n_heads = config["n_heads"]
        self.att_dim = self.n_heads * self.head_dim
        # during inference => dropout is 0 (no reg.), else during training use this
        self.drop_prob = config["drop_prob"] if "drop_prob" in config else 0.0

        # projection matrices
        # self.k_proj = nn.Linear(self.emb_dim, self.head_dim, bias = False)
        # self.q_proj = nn.Linear(self.emb_dim, self.head_dim, bias = False)
        # self.v_proj = nn.Linear(self.emb_dim, self.head_dim, bias = False)
        self.kqv_proj = nn.Linear(self.emb_dim, 3 * self.att_dim)

        self.out_proj = nn.Linear(self.att_dim, self.emb_dim)
        self.attn_drop = nn.Dropout(self.drop_prob)
        self.scale = self.head_dim ** (-0.5)

    def forward(self, x):

        # Converting to Attention Dim -> Head Splitting
        ####################################################
        # x: is coming after Patch Embedding (sometimes cls is removed)
        # (B, num_patches, emb_dim) -> (B, num_patches, 3 * att_dim)
        x = self.kqv_proj(x) 
        # (B, num_patches, 3 * att_dim) -> (B, num_patches, att_dim)
        k, q, v = torch.split( x, 
                              split_size_or_sections= self.att_dim, 
                              dim = -1 ) # means we will have (-1, att_dim) but 3 of these as `3` was the multiplier
        print(k.shape)


        # q: Input Repr
        # k: Context Repr of how much other tokens are relevant to input
        # v: Context Repr of ..

        ########## Before cosine similarity calc, reshape so that heads can be attended parallely ####
        # (B, num_patches, att_dim) -> (B, n_heads num_patches head_dim)
        k = rearrange(k, 
                      pattern = "b n (h hd) -> b h n hd",
                      h = self.n_heads,
                      hd = self.head_dim)
        q = rearrange(q, 
                      pattern = "b n (h hd) -> b h n hd",
                      h = self.n_heads,
                      hd = self.head_dim)
        v = rearrange(v, 
                      pattern = "b n (h hd) -> b h n hd",
                      h = self.n_heads,
                      hd = self.head_dim)
        ####################################################
        


        # Attention Weight Computation
        ####################################################
        # np: num_patches; nh: n_heads; hd: head_dim
        # (B, nh, np, hd) @ (B, nh, hd, np) -> (B, nh, np, np)
        att_weights = torch.matmul(q, k.transpose(-2, -1)) 
        att_weights *= self.scale

        # sum in every row / col as 1
        att_weights = torch.softmax( att_weights, dim = -1 )
        att_weights = self.attn_drop(att_weights)
        ####################################################


        # Weighted Value Computation
        ####################################################
        # (B, nh, np, np) @ (B, nh, np, hd) -> (B, nh, np, hd)
        att_outputs = torch.matmul(att_weights, v)
        ####################################################


        # Project back to Transformer Dimension
        ##############################################
        # (B, nh, np, hd) => n_heads have info independently attended
        # We want to aggregate the info and project it back, needs rehaping
        att_outputs = rearrange(att_outputs, 
                                pattern = "b nh np hd -> b np (nh hd)",
                                nh = self.n_heads,
                                hd = self.head_dim
                                )
        # (B, np, att_dim) -> (B, np, embed_dim)
        att_outputs = self.out_proj(att_outputs)
        ##############################################

        # return the contextualised embeddings
        return att_outputs 


if __name__ == "__main__":
    bsz, num_patches, emb_dim = 2, 16, 256

    config = dict()
    config["n_heads"] = 8
    config["head_dim"] = 128
    config["att_dim"] = 1024
    config["drop_prob"] = 0.2
    config["emb_dim"] = emb_dim

    x = torch.randn((bsz, num_patches, emb_dim))
    model = Attention(config)
    print( model(x).shape )