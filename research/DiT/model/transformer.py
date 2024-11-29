import torch
import torch.nn as nn
from einops import rearrange

from patch_embed import PatchEmbedding
from transformer_layer import TransformerLayer

def get_time_embedding(time_steps: torch.Tensor, temb_dim: int):
    r"""
    Convert time steps tensor into an embedding using the
    sinusoidal time embedding formula
    :param time_steps: 1D tensor of length batch size
    :param temb_dim: Dimension of the embedding
    :return: BxD embedding representation of B time steps
    """
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"

    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0,
        end=temb_dim // 2,
        dtype=torch.float32,
        device=time_steps.device) / (temb_dim // 2))
    )

    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb

class DiT(nn.Module):
    '''
        Takes the latent input (out of VAE's Encoder )
        Spits out the predicted `latent noise`

        1. Patchify ( Image -> Patches )
        2. Adds conditioning signal i.e timestep_emb
        3. Goes through the Transformer i.e Nx of TransformerLayer
        4. Unpatchify Block (mostly implement, others I already have)
            - AdaptiveLN preds scale, shift params from conditioning_time_signal
            - MLP Layer -> Projects from `hidden_dim` => `c * ph * pw`
            - Later rearrange to get b, c, h, w in latent space
    '''

    def __init__(self, im_size: int=224, im_channels: int=3, config = None):
        super().__init__()
        self.im_channels = im_channels
        self.image_height = im_size
        self.image_width = im_size

        self.hidden_size = config["hidden_size"]
        self.patch_height = config["patch_size"]
        self.patch_width = config["patch_size"]
        self.patch_height = config["patch_height"]

        # Number of patches along height / width ( i.e grid of patches shape)
        self.nh = self.image_height // self.patch_height
        self.nw = self.image_width // self.patch_width

        # Recall the diag
        self.patch_embedding_layer = PatchEmbedding( self.image_height, \
                        self.image_width, 
                        self.im_channels, 
                        self.patch_height, 
                        self.patch_width,
                        self.hidden_size )
        

        # First the incoming "timestep" input, first encode the position info
        # Then this is living in the timestep_emb_dim space
        # We project it to hidden_size space
        self.timestep_emb_dim = config["timestep_emb_dim"]
        self.timestep_proj = nn.Sequential( \
                                            nn.Linear(self.timestep_emb_dim, self.hidden_size),
                                            nn.SiLU(),
                                            nn.Linear(self.hidden_size, self.hidden_size)
                                          )

        # Nx of TransformerLayer
        self.num_layers = config["num_layers"]
        self.layers = nn.ModuleList( [TransformerLayer(config) \
                                    for _ in range(self.num_layers)] )


        # Unpatchify Block
        self.unpatchify_layernorm = nn.LayerNorm(self.hidden_size, \
                     elementwise_affine=False,
                     eps = 1e-6) # alpha and beta ot ffrom here, rather from the conditioning signal

        # Learns the scale and shift for the above norm layer
        # 2*self.hidden_size => scale, shift params to mult and add to the above norm o/p
        self.adaptive_layernorm = nn.Linear( self.hidden_size, 2*self.hidden_size )

        # Recall: This is inverse of what we do in the PatchEmb where we move from 
        # (bsz, im_ch, im_h, im_w) => (bsz im_ch (pat_h nh) (pat_w nw) ) => (bsz (nh nw) (im_ch im_h im_w) )
        # (bsz (nh nw) (im_ch im_h im_w) ) => (bsz num_patches hidden_size) [PATCH EMBEDDING OPERATION]
        self.out_proj = nn.Linear( self.hidden_size, \
                                  self.im_channels * self.patch_height * self.patch_width )


        ###### INIT #################
        nn.init.normal_(self.timestep_proj[0].weight, std=0.02)
        nn.init.normal_(self.timestep_proj[2].weight, std=0.02)

        nn.init.constant_(self.adaptive_layernorm[-1].weight, 0)
        nn.init.constant_(self.adaptive_layernorm[-1].bias, 0)

        nn.init.constant_(self.out_proj.weight, 0)
        nn.init.constant_(self.out_proj.bias, 0)
        #############################

    
    def forward(self, x, t):
        # (b c, h, w) => (b, num_patches, hidden_size)
        out = self.patch_embedding_layer(x)

        # for layer in self.layers:
        #     out = layer(out, t_emb) # But we can't do this as we need to make t_emb in same dim

        # We have to fuse the timestep emb info into the patches
        # prep t as torch_tensor
        t = torch.as_tensor(t).long() # typecast as torch.Tensor
        # B, temb_dim
        t_emb = get_time_embedding(t, self.timestep_emb_dim)
        # (B, hidden_size)
        t_emb = self.timestep_proj(t_emb)
        # Now we can fuse timestep info as conditioning signal to patch info
        # (b, num_patches, hidden_size) => (b, num_patches, hidden_size)
        for layer in self.layers:
            out = layer(out, t_emb)      

        # Apply adaptive layernorm on conditiuoning info
        # predict the shift and scale
        norm_scale, norm_shift = self.adaptive_layernorm(t_emb).chunk(2, dim = 1)
        # (b, num_patches, hidden_size) -> (b, num_patches, hidden_size)
        out = self.unpatchify_layernorm(out)
        # norm_scale and shift are init 0s before training
        # Hence, we add 1 + .. => So that it behaves as Identity
        out = out*(1 + norm_scale.unsqueeze(1)) + norm_shift.unsqueeze(1)

        # Final proj and get the predicted noise in latent space
        # (b, num_patches, hidden_size) -> (b, num_patches, im_ch * pat_h * pat_w)
        # Equivalent to: (b (nh nw) (c * ph pw) )
        out = self.out_proj(out)    

        # rearrange and get back to latent image dim
        rearrange(out, 
                  pattern = "b (nh nw) (c ph pw) -> b c (nh ph) (nw pw) ",
                  ph = self.patch_height,
                  pw = self.patch_width,
                  c = self.im_channels,
                  nw = self.nw,
                  nh = self.nh )
        # (B, C, H, W)
        return out # predicted noise in latent space