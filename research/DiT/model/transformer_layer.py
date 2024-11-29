import torch
import torch.nn as nn
from attention import Attention

class TransformerLayer(nn.Module):
    def __init__(self, config):
        '''
            Recall the viz ( Refer the design docs )
                # Disable LayerNorm's elementwise_affine
                # Let the model predict these scaling and shifting params
                # from conditioning signal

            1. x -> LayerNorm -> out = pre_attn_scale*LN + pre_attn_shift
            2. y = Attention(out)*post_attn_scale
            3. Residual: out = x + y

            4. out -> LayerNorm -> out = pre_mlp_scale*LN + pre_mlp_shift
            5. y = MLP(out) * post_mlp_scale
            6. Residual: out = x + y
        '''
        super().__init__()
        self.hidden_size = config["hidden_size"]
        ## Adaptive Layer => To predict the 6 params namely
        # pre_attn_scale, pre_attn_shift, post_attn_scale
        # pre_mlp_scale, pre_mlp_shift, post_mlp_scale

        # Let's Not learn scale and shift for these, rathger predict based on c-signal
        # ( (x - mu) / (std + epsilon) ) * scale + shift = LN
        self.pre_attn_layernorm = nn.LayerNorm( self.hidden_size, 
                                                elementwise_affine=False,
                                                eps = 1e-6 )
        self.attn = Attention(config)

        # Apply post_scale param -> residual recall

        self.pre_mlp_layernorm = nn.LayerNorm( self.hidden_size, 
                                               elementwise_affine=False,
                                               eps = 1e-6 )
        
        # up_proj -> act -> down_proj
        self.interm_size = 4 * self.hidden_size # similar to LLaMa
        self.mlp = nn.Sequential( 
                                    nn.Linear(self.hidden_size, self.interm_size), \
                                    nn.GELU("tanh"),
                                    nn.Linear(self.interm_size, self.hidden_size)
                                  )

        self.adapative_layer = nn.Sequential( \
                                                nn.SiLU(),
                                                nn.Linear( self.hidden_size, 6*self.hidden_size, bias = True )
                                            )

        ### INIT ############
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.constant_(self.mlp[0].bias, 0)
        nn.init.xavier_uniform_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, 0)

        nn.init.constant_(self.adapative_layer[-1].weight, 0)
        nn.init.constant_(self.adapative_layer[-1].bias, 0)
        ##################


    def forward(self, x, conditioning_signal):

        # 1st we predict the adative params
        scale_shift_params = self.adapative_layer(conditioning_signal).chunk(6, dim=1)
        (pre_attn_shift, pre_attn_scale, post_attn_scale,
         pre_mlp_shift, pre_mlp_scale, post_mlp_scale) = scale_shift_params
        
        residual_inp = x
        out = self.pre_attn_layernorm(x)
        # Explcitly use the para,s
        # NOTE: Initially due to initialisation all will be 0
        # Hence, the scale will make it 0, (1 + ..) will make the transformation Identity
        out = out * (1 + pre_attn_scale.unsqueeze(1)) + pre_attn_shift.unsqueeze(1)
        out = self.attn(out)
        # Use post_attn i.e alpha recall diagram
        out = out * post_attn_scale.unsqueeze(1)
        out = residual_inp + out

        # (Norm -> MLP)
        residual_inp = out
        out = self.pre_mlp_layernorm(out)
        out = out * (1 + pre_mlp_scale.unsqueeze(1)) + pre_mlp_shift.unsqueeze(1)
        out = self.mlp(out)
        # Use post_mlp i.e alpha recall diagram
        out = out * post_mlp_scale.unsqueeze(1)
        out = residual_inp + out

        return out


