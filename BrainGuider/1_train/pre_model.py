
import torch
import torch.nn as nn


def random_masking_tokens(x, mask_ratio, N, patch_size):
    B, L, D = x.shape
    patch_number = L // N

    if mask_ratio <= 0.0:
        x_keep = x                                    # [B, L, D]

        mask = torch.zeros(B, L, device=x.device)     # [B, L]
        ids_restore = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)

        mask_nt = mask.view(B, N, patch_number)            # [B, N, P]
        mask_tn = mask_nt.permute(0, 2, 1).contiguous()    # [B, P, N]
        if patch_size > 1:
            mask_btn = mask_tn.repeat_interleave(patch_size, dim=1)
        else:
            mask_btn = mask_tn                              # [B, T, N]

        return x_keep, mask, ids_restore, mask_btn

    len_keep = int(L * (1 - mask_ratio))
    noise = torch.rand(B, L, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]
    x_keep = torch.gather(
        x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
    )

    mask = torch.ones(B, L, device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    mask_nt = mask.view(B, N, patch_number)
    mask_tn = mask_nt.permute(0, 2, 1).contiguous()
    if patch_size > 1:
        mask_btn = mask_tn.repeat_interleave(patch_size, dim=1)
    else:
        mask_btn = mask_tn

    return x_keep, mask, ids_restore, mask_btn


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
        
        # [Batch_size, brain_region, self.patch_embed, self.patch_number] --> [Batch_size, brain_region, self.patch_number, self.patch_embed]
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


class pre_encoderModel(nn.Module):

    def __init__(self, config_en):
        super().__init__()

        self.F_latent = config_en["F_latent"]
        self.N = config_en["n_regions"]
        self.T_in = config_en["input_time"]
        self.patch_size = config_en["patch_size"]

        self.patch_embeddings = PatchEmbedding(
            patch_size=self.patch_size,
            brain_region=self.N,
            patch_embed=self.F_latent,
            Time_len=self.T_in,
        )

        self.TemporalLayer = TemporalTransformerLayer(
            self.F_latent, num_layers=4, num_heads=4, dropout=0.1
        )

    def forward(self, x, mask_ratio):

        x = x.permute(0, 2, 1)

        tokens = self.patch_embeddings(x)
        tokens_keep, mask, ids_restore, mask_btn = random_masking_tokens(
            tokens, mask_ratio=mask_ratio, N=self.N, patch_size=self.patch_size,
        )  # tokens_keep: [B, L_keep, F]

        z = self.TemporalLayer(tokens_keep)  # [B, L_keep, F_latent]

        aux = {
            "tokens": tokens,
            "mask": mask,
            "ids_restore": ids_restore,
            "mask_btn": mask_btn,
        }

        return z, aux


class pre_decoderModel(nn.Module):
    def __init__(self, config_de):
        super().__init__()

        self.N = config_de["n_regions"]
        self.F_latent = config_de["F_latent"]
        self.L_in = config_de["input_time"] // config_de["patch_size"]
        self.patch_size = config_de["patch_size"]

        self.DecoderLayer = TemporalTransformerLayer(
            self.F_latent, num_layers=2, num_heads=4, dropout=0.1
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.F_latent))
        self.token_to_time = nn.Linear(self.F_latent, self.patch_size)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.xavier_uniform_(self.token_to_time.weight)

    def forward(self, z, aux):
 
        mask = aux["mask"]
        ids_restore = aux["ids_restore"]

        B, L_keep, F_latent = z.shape
        L_full = mask.size(1)
        L_mask = L_full - L_keep

        mask_tokens = self.mask_token.expand(B, L_mask, F_latent)

        x_ = torch.cat([z, mask_tokens], dim=1)
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, F_latent)
        )

        x_dec_0 = self.DecoderLayer(x_)
        x_dec = x_dec_0.view(B, self.N, self.L_in, F_latent)         # [B, N, L_in, F]

        x_time = self.token_to_time(x_dec).reshape(B, self.N, self.L_in *self.patch_size)      # [B, N, L_in, self.T_out]
        out = x_time.transpose(1, 2)                        # [B, T_out(=L_in), N]

        return out, x_dec_0

