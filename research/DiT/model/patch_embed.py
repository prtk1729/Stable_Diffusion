import torch
import torch.nn as nn
from einops import rearrange, repeat

def get_patch_pos_emb( hidden_size, grid_size, device ):
    nh, nw = grid_size # (num_patches along ht, num_pathces along width)
    # For each patch
        # In 1D (say time-emb):
            # Each timestep: [ sin(pos_row) cos(pos_row) ]
            # Since, timestep in 1 dimensional vector, we only have row-dim
            # to uniquely identify a timestep position
            # Hence, "i" i.e hidden_dim % 2 == 0
        # In 2D (say patch_pos emb)
            # Each patch-pos can be uniquely defined by row_pos, col_pos
            # Each patch_pos: [ sin(pos_row), cos(pos_col), sin(pos_col), cos(pos_col) ]
            # Hence, "i" i.e hidden_dim % 4 == 0
    assert hidden_size % 4 == 0, "For 2D pos-emb, the hidden_dim should be divisible by 4"
    # create coords: for 2c, 2-> [ (0, 0), [0, 1], (1, 0), (1, 1) ] => [ pos_emb(0, 0), .... ]

    # Can use meshgrid to achieve this
    grid_height = torch.arange( nh, dtype = torch.float32, device = device ) # all coords along height_dim i.e rows
    grid_width = torch.arange( nw, dtype = torch.float32, device = device )

    # First grid_height repeats this along "col" (this row vector) i.e grid[0]
        # [0, 1] ->   [ | | ] where each | is [0, 1] vertically
    # Then grid_width repeats this along "row" (this col vector) i.e grid[1]
        # [0, 1] -> [ -- , 
        #             -- ]  where -- i.e row is [0, 1] horiszontally
    # The returned grid: grid[0], grid[1] 
        # grid[0]: Only fetch the y-coords/row-nums i.e [ [0, 0], [1, 1] ] in RMO
        # grid[1]: Only fetch the x-coords/col-nums i.e [ [0, 1], [0, 1] ] in RMO
        # If I flatten both then see, they are in RMO coords



    grid = torch.meshgrid( grid_height, grid_width, indexing = "ij" )
    grid_h_positions = grid[0].reshape(-1)
    grid_w_positions = grid[1].reshape(-1)
    # print( grid_h_positions ) # tensor([0, 0, 1, 1])
    # print( grid_w_positions ) # tensor([0, 1, 0, 1])


    # sinosuids
    # 10000 ^ (4i / d), i = {0, 1, 2...., d/4}
    factor = 10000.0 ** (torch.arange( start = 0, end = hidden_size, step = 4, 
                                      dtype = torch.float32, device=device ) / hidden_size)

    # arg: pos / factor
    # [ sin(arg), cos(arg) ] -> For each 1D
    # [ sin(arg_row_pos), cos(arg_row_pos), sin(arg_col_pos), cos(arg_col_pos) ] -> For each 1D

    grid_h_emb = grid_h_positions[:, None].repeat(1, hidden_size // 4) / factor
    grid_h_emb = torch.cat([torch.sin(grid_h_emb), torch.cos(grid_h_emb)], dim=-1)
    # grid_h_emb -> (Number of patch tokens, hidden_size // 4)

    grid_w_emb = grid_w_positions[:, None].repeat(1, hidden_size // 4) / factor
    grid_w_emb = torch.cat([torch.sin(grid_w_emb), torch.cos(grid_w_emb)], dim=-1)
    pos_emb = torch.cat([grid_h_emb, grid_w_emb], dim=-1)

    # pos_emb -> (Number of patch tokens, pos_emb_dim)
    return pos_emb




class PatchEmbedding(nn.Module):
    def __init__(self,
                 image_height,
                 image_width, 
                 im_channels, 
                 patch_height,
                 patch_width,
                 hidden_size):
        self.image_height = image_height
        self.image_width = image_width
        self.im_channels = im_channels
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.hidden_size = hidden_size


        super().__init__()


        # 2 Responsibilities
            # - Patchify
            # - Add positionl info to each patch_repr

        self.patch_dim = self.patch_height * self.patch_width * self.im_channels
        # Projection layer from (ph * pw * C) -> (hidden_dim)
        self.patch_emb = nn.Sequential( \
                                       nn.Linear(self.patch_dim, self.hidden_size) )
        
        ### DiT inits ################\
        # Preventing the model's training to face vanishing and exploding grads problems
        # Also, not introducing biases from start (if not done either uninit bias OR small random bias) can result in eratic training at the beginning
        nn.init.xavier_uniform_( tensor = self.patch_emb[0].weight )
        nn.init.constant_( tensor = self.patch_emb[0].bias, val = 0 ) # init with 0
        ##############################
        

    def forward(self, x):
        # 1st project
        # (b, c, h, w) -> (can't do patch_dim Need to rearrange) 
        # out = self.patch_emb(x)

        # (b, c, h, w) -> (b, num_patches, patch_dim) 
        out = rearrange( x, 
                        pattern = "b c (ph nh) (pw nw) -> b (nh nw) (c ph pw)",
                        ph = self.patch_height,
                        pw = self.patch_width )
        # (b, num_patches, patch_dim) -> (b, num_patches, hidden_size)
        out = self.patch_emb(out)

        # Gets a 2D pos-em repr for each patch
        # Transformer, unlike CNNs are position invariant i.e [t1 t2 t3] == [t2 t1 t3] interms of contextual emb
        # So, we expliciltly need to integrate the positional info
        # nh: number of patches along height, nw: Similarly
        nh, nw = (self.image_height // self.patch_height), (self.image_width // self.patch_width)
        pos_emb = get_patch_pos_emb(self.hidden_size, grid_size = (nh, nw), 
                          device = x.device) # gets the 3d patch-pos emb
        out = out + pos_emb
        return out



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_height, image_width, im_channels = 128, 128, 3
    patch_height, patch_width, hidden_size = 2, 2, 768
    x = torch.randn((2, im_channels, image_height, image_width)).to(device)
    model = PatchEmbedding( image_height, image_width, im_channels, patch_height, patch_width, hidden_size ).to(device)
    out = model(x)
    print(out.shape) # torch.Size([2, 4096, 768]) -> (bsz, num_patches, hidden_size)

    # nh, nw = (image_height // patch_height), (image_width // patch_width)
    # nh, nw = 2, 2
    # out = get_patch_pos_emb(hidden_size, grid_size = (nh, nw), device = x.device )
    # print( out )
    # print( out.shape )
