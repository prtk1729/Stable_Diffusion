import torch 
import torch.nn as nn
from einops import rearrange, repeat
from tqdm import tqdm


def get_time_embs(time_steps, time_emb_dim: int):
    # We implement the sinosuidal pos emb
    # time_steps: [ 0, 1, 2 ]
    # -> Convert these by projecting them to the time_emb_space

    # factor: 1000 ** ( 2i/dim )
    # dim = 6 => [1000.0^0, 1000.0^1/3, 1000.0^2/3 ] = [1, 10.0, 100.0]
    # [ [0 0 0], [1/1, 0.1, 0.01], [2/1, 0.2, 0.02] ]
    assert time_emb_dim % 2 == 0, "time_emb_dim should be even"

    factor = 10000.0 ** (torch.arange( start = 0, 
                             end = time_emb_dim, 
                             step = 2, dtype = torch.float32 ) / time_emb_dim)
    
    # create a new dim to interact with factor
    # (B) ---[:, None]--> (B, 1) --expand-> (B, dim//2) 
    time_steps = time_steps[:, None].expand(-1, time_emb_dim//2) # (time_emb_dim/2, 1)
    term = time_steps / factor 
    # print( term.shape, time_steps.shape, factor.shape )

    # now we can do [ sin(term) cos(term) ] for each time
    embs = torch.cat( [torch.sin(term), torch.cos(term) ], dim = -1)
    return embs




def print_gpus():
    # Get the number of GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # List each GPU's name
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Check if the current process is using GPUs
    current_device = torch.cuda.current_device()
    print(f"Currently using GPU {current_device}: {torch.cuda.get_device_name(current_device)}")
    return

class Down(nn.Module):
    def __init__(self, 
                 in_channels: int, out_channels: int, 
                 num_heads: int, num_layers: int, 
                 attn: bool, # Whether we want SA Block? 
                 norm_channels: int,
                 t_emb_dim: int,
                 down_sample: bool,
                 context_dim = None,
                 cross_attn: bool = False):
        super().__init__()

        self.num_heads = num_heads
        self.num_layers = num_layers 
        self.attn = attn # Whther to use SA?
        self.cross_attn = cross_attn # Whether to use cross_attn 
        self.down_sample = down_sample # Whether to down_sample?
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.t_emb_dim = t_emb_dim
        self.context_dim = context_dim

        # Recall: Viz of Down Block
        # (resnet_conv1) -- time_emb -- (resnet_conv2) -- skip --- Norm -- SA
        # GrpNorm: Since, we are working with images. Why?

        ######## resnet_conv1 ###############
        self.resnet_conv1 = nn.ModuleList( [
                                                nn.Sequential(
                                                    nn.GroupNorm(num_groups = 1 if i == 0 else norm_channels, 
                                                          num_channels = in_channels if i == 0 else out_channels
                                                          ), # layer 1, 2, , ... will take output of 1st layer
                                                    nn.SiLU(),
                                                    nn.Conv2d(in_channels = in_channels if i == 0 else out_channels,
                                                            out_channels = out_channels,
                                                            kernel_size = 3,
                                                            padding = 1,
                                                            stride = 1
                                                            ) # same conv
                                                )
                                                for i in range(num_layers) 
                                           ]
                                           
                                         )
        ######################################

        ######## time_emb as a conditioning signal, which we concat after resnet_conv1 ###############
        if t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList( 
                                        [
                                            nn.Sequential( nn.SiLU(), \
                                                            nn.Linear(t_emb_dim, out_channels)
                                                            ) # projects t_emb_dim to out_channels to "add"
                                            for i in range(num_layers) 
                                        ]
                                        )
        ######################################

        ##### resnet_conv2 ##################
        self.resnet_conv2 = nn.ModuleList( [
                                                nn.Sequential(
                                                    nn.GroupNorm(num_groups = norm_channels, 
                                                          num_channels = out_channels
                                                          ), 
                                                    nn.SiLU(),
                                                    nn.Conv2d(in_channels = out_channels,
                                                            out_channels = out_channels,
                                                            kernel_size = 3,
                                                            padding = 1,
                                                            stride = 1
                                                            ) # same conv
                                                )
                                                for i in range(num_layers) 
                                           ]
                                         )       
        ######################################


        ####### Norm + SA ##########################
        if self.attn:
            self.attn_norm = nn.ModuleList( [ nn.GroupNorm(norm_channels, out_channels) \
                                                 for i in range(num_layers) ] )
            
            # init -> requires n_heads, dim? But forward requires k, v, q
            # Can u deduce what's the embed_dim? What's the input in the block diag?
            # nn.MHA expects input in the shape (seq_len, bsz, emb_dim)
            # But, we want to pass data in the shape: [bsz, seq_len, emb_dim]
            # seq_len in case of images = num_patches / num_image_tokens 
            self.self_attn = nn.ModuleList( [   
                                                nn.MultiheadAttention( num_heads = self.num_heads, \
                                                   embed_dim = out_channels,
                                                   batch_first = True,    
                                                  ) 
                                                for _ in range(num_layers)
                                            ] 
                                            ) 
        ######################################


        ############## Cross Attention (Prompt or Text) in cond LDMs
        if self.cross_attn:
            assert context_dim is not None, "Context is necessary for Cross-Attention in this architecture"
            # CA's puropse is to attend to the "text / prompt" as the conditioning signal
            # NormC, CA, 
            self.cross_attention_norms = nn.ModuleList( [nn.GroupNorm(norm_channels, out_channels)  \
                                                         for _ in range(num_layers)] )
            self.cross_attentions = nn.ModuleList(  [ nn.MultiheadAttention(out_channels, 
                                                                           num_heads, 
                                                                           batch_first=True)  
                                                     for _ in range(num_layers) 
                                                    ] 
                                           )
            # Project text_embedding from context_emb to out_channels to add
            # We need to add this Context_Emb in the feature map, as additional info
            # about the image we want to train on, Mostly used in <text-image> training
            self.context_proj = nn.ModuleList( 
                                                [
                                                nn.Linear(context_dim, out_channels)
                                                for _ in range(num_layers)
                                                ]  
                                              )
        ##############################################


        ##### Resnet Skip Conv (1st skip after the 2 conv layers) ############ 
        # input to this can either be "x" tensor or out from last_layer i.e i-1 layer of DownBlock
        self.resnet_input_conv = nn.ModuleList( [  nn.Conv2d( in_channels = in_channels if i == 0 else out_channels, 
                                                              out_channels= out_channels,
                                                              kernel_size=1) 
                                                    for i in range(num_layers)
                                                ]
                                              )
        ##############################################
        
        ##### DownSampleConv ############ 
        # This Downsamples the spatial dim
        # Either MaxPool or other pooling are used OR
            # conv layer can be used
            # Pros: Why to alwyas, pick the max faeture val?
            # Maybe we can learn to pick the best features inorder to Downsample
            # Pool -> Fast, non-learnable params, best feats needn't be aggregations 
            # When working in Latent Space, for reconstruction, if we can learn them better
            # reconstruction/ feat encoding happens   
            # ( out_channels + 2*pad - kernel_sz)/stride + 1 ~ B, C_out, H/2, W/2
        self.down_sample_conv = nn.Conv2d( out_channels, 
                                           out_channels, 
                                           kernel_size=4, padding=1, 
                                           stride=2 ) if self.down_sample else nn.Identity()
                                                
        ##############################################


    def forward(self, x, context: None, t_emb: None):
        b, in_channels, height, width = x.shape

        out = x
        for i in tqdm(range(self.num_layers)):
            print_gpus()

            skip_inp = out

            out = self.resnet_conv1[i](out)

            # Fuse the timestep info
            if self.t_emb_dim is not None:
                # (B, t_emb_dim) -> (B, out_channels)
                t_proj = self.t_emb_layers[i](t_emb)
                assert t_proj.shape == (b, self.out_channels), "Line 176"
                # Need to expand dim
                # (B, out_ch, H, W) + (B, out_ch, 1, 1) => can broadcsat now
                out = out + t_proj[:, :, None, None] 
            
            out = self.resnet_conv2[i](out)

            # recall the skip connection (1st skip after 2 resnet_conv_blocks)
            # if i == 0, => (B, in_ch, H, W) -> (B, out_ch, H, W)
            # if i >= 1, => (B, out_ch, H, W) -> (B, out_ch, H, W)
            # Either way, we can now add to out (as the channels match)
            # out.shape: (B, out_ch, H, W) + updated_inp_to_add
            inp_to_add = self.resnet_input_conv[i](skip_inp)
            out = out + inp_to_add

            if self.attn:
                # out = self.attn_norm(out) # can we do this? What does this expect?
                # Norm expects last dim to have actual feat dim
                # (B, out_ch, H, W) -> (B, out_ch, H*W) 
                out = out.flatten(2)
                out = self.attn_norm[i](out)
                # Prepare for attn: (B, out_ch, H*W) -> (B, H*W, out_ch)
                out = out.transpose(-1, -2)
                # (B, H*W, out_ch) -> (B, H*W, out_ch) as SA expects that
                out, _ = self.self_attn[i](query = out, key = out, value = out) # attn_out, attn_weights
                # undo the reshape
                out = rearrange(out, 
                                pattern = "b (h w) oc -> b oc h w",
                                h = height)
            
            if self.cross_attn:
                skip_inp = out
                # (B, out_ch, H, W) -> (B, out_ch, H*W) 
                out = out.flatten(2)
                out = self.cross_attention_norms[i](out)

                # context info before integrating to features
                #Prepare context for cross_attn
                # (B, seq_len, context_dim) -> (B, seq_len, out_ch)
                assert context.shape[0] == x.shape[0] and context.shape[-1] == self.context_dim, "Context Shape doesn't match Expected Shape"
                context_proj = self.context_proj[i](context)

                # Prepare for cross_attn: (B, out_ch, H*W) -> (B, H*W, out_ch)
                out = out.transpose(-1, -2)
                # (B, H*W, out_ch) -> (B, H*W, out_ch) as SA expects that
                out, _ = self.cross_attentions[i](query = out, key = context_proj, value = context_proj) # attn_out, attn_weights
                # undo the reshape
                out = rearrange(out, 
                          pattern = "b (h w) oc -> b oc h w",
                          h = height
                          )
                out = out + skip_inp

        # Downsample
        out = self.down_sample_conv(out)
        return out



class Mid(nn.Module):
    def __init__(self, config):
        pass

    def forward(self):
        pass


class Up(nn.Module):
    def __init__(self, config):
        pass

    def forward(self):
        pass





if __name__ == "__main__":
    time_steps = torch.tensor([0, 1, 2])
    time_dim_emb = 6

    time_embs = get_time_embs(time_steps, time_dim_emb)
    print( time_embs.shape )

    batch_size, height, width = 2, 128, 128
    context_dim = 768
    t_emb_dim = 768

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Down( in_channels=3,
                 out_channels=128,
                 num_heads = 4,
                 num_layers = 3,
                 attn = True,
                 norm_channels = 32,
                 t_emb_dim = t_emb_dim,
                 down_sample = True,
                 context_dim = context_dim,
                 cross_attn = True 
                 ).to(device)
    model = torch.nn.DataParallel(model)

    x = torch.randn( (batch_size, 3, height, width) ).to(device)
    context = torch.randn( (batch_size, 12, context_dim) ).to(device)
    t_emb = torch.randn( (batch_size, t_emb_dim) ).to(device)

    out = model(x, context, t_emb)  
    print( out.shape ) # torch.Size([2, 128, 64, 64])