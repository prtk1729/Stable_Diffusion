import torch
import torch.nn as nn

from einops import rearrange, repeat

class PatchEmbedding(nn.Module):
    def __init__(self, config):
        '''
            3 Responsibilities:-
                1. Project patch_feats to transformer dim
                2. Add CLS tokens to patch sequences
                3. Add the positional info.
        '''
        super().__init__()
        # treat config asd a dict / some use csaes we can use it as a class
        self.im_channels = config["im_channels"]
        self.image_height = config["image_height"]
        self.image_width = config["image_width"]

        self.patch_height = config["patch_height"]
        self.patch_width = config["patch_width"]

        # for regularisation
        self.patch_emb_dropout = nn.Dropout( config["patch_embd_drop"] ) # dropout rate

        # For embeddings
        self.embed_dim = config["embed_dim"]

        # For positional info, we need to know number of patches    
        self.num_patches = (self.image_height // self.patch_height) * (self.image_width // self.patch_width)

        self.patch_dim = self.im_channels * self.patch_width * self.patch_height
        # (B, num_patches, patch_dim ) -> (B, num_patches, embed_dim)
        self.patch_embeds = nn.Sequential( nn.Linear( self.patch_dim, self.embed_dim ) )

        self.patch_embeds2 = nn.Conv2d( in_channels = self.im_channels,
                                        out_channels = self.embed_dim,
                                        kernel_size = self.patch_height,
                                        stride = self.patch_height,
                                        padding = 0
                                      )

        # CLS token to be appended to carry a repr. of the image / This will come in handy during classification
        self.pos_embed = nn.Parameter( torch.zeros(1, 1 + self.num_patches, self.embed_dim) )
        self.cls_token = nn.Parameter( torch.zeros( self.embed_dim ) ) # we will broadcast it later to introduce batch

        

    def forward(self, x):
        # (b, c, h, w)
        batch_size, c, h, w = x.shape

        # intend to make it go from "b c (ph nh) (pw nw) -> b (nh nw) (c ph pw) "
        # (b, c, h, w) -> ( b, num_patches=nh*nw, patch_dim=c*ph*pw )
        out = rearrange( 
                  x, 
                  pattern = "b c (ph nh) (pw nw) -> b (nh nw) (c ph pw)",
                  ph = self.patch_height,
                  pw = self.patch_width
                  )
        
        # Can do the above using nn.Conv2d(). Can u do later?
        # project to the embed_dim and these are learnables
        # ( b, num_patches, patch_dim ) -> (b, num_patches, embed_dim)
        image_token_embeds = self.patch_embeds(out)
        # print( image_token_embeds.shape )

        # # (b, c, h, w) -> (b, embed_dim, n_patch_h, n_patch_w)
        # image_token_embeds = self.patch_embeds2(x)
        # b, d, nh, nw = image_token_embeds.shape
        # # (b, embed_dim, n_patch_h, n_patch_w)
        # image_token_embeds = rearrange(image_token_embeds, 
        #                                pattern = "b d nh nw -> b (nh nw) d"  
        #                                )
        # print( image_token_embeds.shape )

        # add embed info
        # Add cls info to each batch item i.e each patch
        cls_tokens = repeat( self.cls_token, 
               pattern = "d -> b 1 d",
               b = batch_size )
        # append it ahead of cls [ img_tok_emb1, img_tok_emb2, ... ] , ...
        out = torch.cat( [cls_tokens, image_token_embeds], dim = 1 ) # num_patches dim

        # Viz => Now we append the pos info
        assert out.shape == (batch_size, 1 + self.num_patches, self.embed_dim), f"Expected out.shape was {(batch_size, 1 + self.num_patches, self.embed_dim)} instead got {out.shape}"

        out += self.pos_embed
        return out


if __name__ == "__main__":
    config = dict()
    config["im_channels"] = 3
    config["image_height"] = 224
    config["image_width"] = 224
    config["patch_height"] = 16
    config["patch_width"] = 16
    config["patch_embd_drop"] = 0.2
    config["embed_dim"] = 1024

    model = PatchEmbedding(config)
    x = torch.randn( 2, 3, 224, 224 )
    out = model(x)
    print( out.shape ) # torch.Size([2, 197, 1024]) # 1 + 196(= num_patches = 14 * 14). 14 = 224 / 16
