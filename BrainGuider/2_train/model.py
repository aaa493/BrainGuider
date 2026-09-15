import torch
import torch.nn as nn

    

class PatchEmbedding(nn.Module):

    def __init__(self, patch_size, brain_region, patch_embed, Time_len):
        super(PatchEmbedding, self).__init__()

        self.patch_size = patch_size
        self.brain_region = brain_region
        self.patch_embed = patch_embed
        self.Time_len = Time_len

        assert Time_len % patch_size == 0, "Time length must be divisible by patch size"
        self.patch_number = (self.Time_len-self.patch_size) // self.patch_size + 1

        self.proj = nn.Conv1d(in_channels=self.brain_region, 
                              out_channels=self.brain_region * self.patch_embed, 
                              kernel_size=self.patch_size, 
                              stride=self.patch_size, 
                              groups=self.brain_region)
        
        self.pos_encoder = PositionalEncoding(d_model=self.patch_embed)

    def forward(self, x):

        Batch_size, brain_region, _ = x.shape
        out = self.proj(x)              # [B, brain_region * patch_embed, patch_number]
        
        out = out.reshape(Batch_size, brain_region, self.patch_embed, self.patch_number).permute(0, 1, 3, 2)

        pos = torch.arange(self.patch_number, device=x.device)
        pos_encoding = self.pos_encoder(pos).unsqueeze(0).unsqueeze(0)  # (1, 1, self.patch_number, F_latent)
        out = out + pos_encoding

        out = out.reshape(Batch_size, brain_region * self.patch_number, self.patch_embed)

        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model):

        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

    def forward(self, pos):

        device = pos.device
        T = pos.size(0)
        d_model = self.d_model

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32, device=device) *
            (-torch.log(torch.tensor(10000.0, device=device)) / d_model)
        )  # (D/2,)

        pe = torch.zeros(T, d_model, device=device)  # (T, D)
        pe[:, 0::2] = torch.sin(pos.unsqueeze(1) * div_term)  # broadcasting: (T, 1) * (D/2)
        pe[:, 1::2] = torch.cos(pos.unsqueeze(1) * div_term)

        return pe  # (T, D)


class TemporalTransformerLayer(nn.Module):
    def __init__(self, F_latent, num_layers, num_heads, dropout):
        super(TemporalTransformerLayer, self).__init__()
        self.F_latent = F_latent

        self.pos_encoder = PositionalEncoding(d_model=F_latent)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=F_latent,
            nhead=num_heads,
            dim_feedforward=4 * F_latent,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, X):
        B, T, F_latent = X.shape

        pos = torch.arange(T, device=X.device)
        pos_encoding = self.pos_encoder(pos).unsqueeze(0)  # (1, T, F_latent)
        X = X + pos_encoding

        X_out = self.transformer(X) # [B, T, F_latent]
        return X_out



class encoderModel(nn.Module):
    def __init__(self, config_en):
        super().__init__()

        self.F_latent = config_en["F_latent"]
        self.N = config_en["n_regions"]
        self.T_in = config_en["input_time"]
        self.patch_size = config_en["patch_size"]

        self.patch_embeddings = PatchEmbedding(patch_size=self.patch_size, brain_region=self.N, patch_embed=self.F_latent, Time_len=self.T_in)
        self.TemporalLayer = TemporalTransformerLayer(self.F_latent, num_layers=4, num_heads=4, dropout=0.1)

        self.output_norm = nn.LayerNorm(self.F_latent)

    def forward(self, x):

        x = x.permute(0, 2, 1) # [B, N, T]
        out_layers = []

        embeddings = self.patch_embeddings(x) # (batch_size, num_patches, hidden_size)
        out_layers.append(embeddings)

        h1 = self.TemporalLayer(embeddings)   # (batch_size, num_patches, hidden_size)
        out_layers.append(h1)

        h1 = self.output_norm(h1)

        return h1, out_layers
    

class decoderModel(nn.Module):
    def __init__(self, config_de):
        super().__init__()
        self.N = config_de["n_regions"]
        self.F_latent = config_de["F_latent"]
        self.L_in = config_de["output_time"] // config_de["patch_size"]
        self.patch_size = config_de["patch_size"]

        self.DecoderLayer = TemporalTransformerLayer(
            self.F_latent, num_layers=2, num_heads=4, dropout=0.1
        )
        self.token_to_time = nn.Linear(self.F_latent, self.patch_size)

    def forward(self, z, pre_tokens=None, alpha: float = 0.0):
        B, L_keep, F_latent = z.shape
        x_dec_1 = z
        x_dec_3 = self.DecoderLayer(x_dec_1)                       # [B, L_full, F]
        x_dec = x_dec_3.view(B, self.N, self.L_in, F_latent)         # [B, N, L_in, F]

        x_time = self.token_to_time(x_dec).reshape(B, self.N, self.L_in *self.patch_size)      # [B, N, L_in, self.T_out]

        out = x_time.transpose(1, 2)                        # [B, T_out(=L_in), N]

        return out, x_dec_1


class Predictor(nn.Module):

    def __init__(self, config_pred):
        super().__init__()

        self.F_latent = config_pred["F_latent"]
        self.layer = config_pred["layers"]
        self.n_regions = config_pred["n_regions"]
        self.TemporalLayer = TemporalTransformerLayer(self.F_latent, num_layers=self.layer, num_heads=4, dropout=0.1)

        self.L_in = config_pred["output_time"] // config_pred["patch_size"]
        self.token_number = self.n_regions *self.L_in
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.F_latent))

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(nn.init, "trunc_normal_"):
            nn.init.trunc_normal_(self.mask_token, mean=0.0, std=0.02, a=-0.04, b=0.04)
        else:
            nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

    def forward(self, x):
        B, L, F = x.shape
        mask = self.mask_token.expand(B, self.token_number, -1)      # [B, L_in, F]

        x_dec = torch.cat([x, mask], dim=1)           # [B, L_in, F]
        out = self.TemporalLayer(x_dec)               # [B, 2*L_in, F]
        out = out[:, -self.token_number:, :]

        return out
    

class ProjectionHead(nn.Module):
    def __init__(self, config_proj):
        super(ProjectionHead, self).__init__()

        self.F_latent = config_proj["F_latent"]
        self.proj_T = nn.Linear(self.F_latent, self.F_latent)

    def forward(self, x_T_tok, x_S_tok):
        x_T_proj = self.proj_T(x_T_tok)   # [B, M, D']
        return x_T_proj, x_S_tok

