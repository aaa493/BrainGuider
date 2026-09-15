
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


def compute_metrics(y_true, y_pred):

    y_true = y_true.float()
    y_pred = y_pred.float()

    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)

    mae = torch.mean(torch.abs(yt - yp))
    mse = torch.mean((yt - yp) ** 2)

    var = torch.var(yt, unbiased=False)
    if var.item() == 0:
        r2 = torch.tensor(0.0)
    else:
        r2 = 1 - mse / var

    eps = 1e-8
    den = torch.clamp(torch.abs(yt), min=eps)
    mape = torch.mean(torch.abs(yt - yp) / den) * 100.0

    yt_mean = torch.mean(yt)
    yp_mean = torch.mean(yp)

    yt_center = yt - yt_mean
    yp_center = yp - yp_mean

    denom = torch.sqrt(torch.sum(yt_center**2)) * torch.sqrt(torch.sum(yp_center**2))
    if denom.item() == 0:
        corr = torch.tensor(0.0)
    else:
        corr = torch.sum(yt_center * yp_center) / denom

    return {
        "MAE": mae.item(),
        "MSE": mse.item(),
        "MAPE": mape.item(),
        "R2": r2.item(),
        "CORR": corr.item()
    }


def evaluate(
    encoder_re,
    decoder_re,
    val_dataloader,
    device,
):

    metrics_sums = {
        "MAE": 0.0,
        "MSE": 0.0,
        "MAPE": 0.0,
        "R2": 0.0,
        "CORR": 0.0
    }

    encoder_re.eval()
    decoder_re.eval()
    criterion = nn.MSELoss(reduction='none')

    total_val_loss = 0.0
    with torch.no_grad():
        for X in tqdm(val_dataloader, desc="Batches val", position=2, leave=False):

            X = X.to(device)

            z, aux = encoder_re(X, mask_ratio=0.5)
            recon_n, _ = decoder_re(z, aux)

            inv_mask = aux["mask_btn"]
            diff = criterion(recon_n, X)
            masked_diff = diff * inv_mask
            loss_val = masked_diff.sum() / (inv_mask.sum() + 1e-5)
            m = compute_metrics(y_true=X, y_pred=recon_n)

            total_val_loss += loss_val
            for key in metrics_sums.keys():
                metrics_sums[key] += m[key]

    num_batches = len(val_dataloader)
    total_metrics = {k: v / num_batches for k, v in metrics_sums.items()}
    total_metrics["loss_re"] = total_val_loss / num_batches

    return total_metrics


def pre_train_model(
        encoderModel,
        decoderModel,
        config,
        task_name,
        train_dataloader,
        val_dataloader,
):

    device = config['train']['device']
    encoder_re = encoderModel(config['encoder_re']).to(device)
    decoder_re = decoderModel(config['decoder_re']).to(device)

    opt_re = optim.Adam(
        list(encoder_re.parameters()) +
        list(decoder_re.parameters()),
        lr=config['train']['lr'],
        weight_decay=1e-6
    )

    criterion = nn.MSELoss(reduction='none')

    patience  = config['train']["early_stop_patience"]
    min_delta = config['train']["early_stop_min_delta"]

    best_val_mse = None
    best_model = None
    bad_epochs = 0

    num_epochs = config['train']['epochs']
    for epoch in tqdm(range(num_epochs), desc="Epochs", position=0, leave=False):

        encoder_re.train()
        decoder_re.train()
        epoch_loss_re = 0.0
        
        for X in tqdm(train_dataloader, desc="Batches", position=1, leave=False):
            X = X.to(device)
            opt_re.zero_grad()

            z, aux = encoder_re(X, mask_ratio=0.5)
            recon_n, _ = decoder_re(z, aux)

            inv_mask = aux["mask_btn"]
            diff = criterion(recon_n, X)
            masked_diff = diff * inv_mask
            masked_loss = masked_diff.sum() / (inv_mask.sum() + 1e-5)

            full_loss = criterion(recon_n, X).mean()

            a = 1.0
            b = 0.1
            total_loss = (a * masked_loss) + (b * full_loss)

            total_loss.backward()
            opt_re.step()
            epoch_loss_re += total_loss.item()

        with torch.no_grad():
            total_metrics = evaluate(
                encoder_re,
                decoder_re,
                val_dataloader,
                device,
            )

        val_mse  = total_metrics["MSE"]
        if best_val_mse is None or (best_val_mse - val_mse) > min_delta:
            best_val_mse = val_mse
            bad_epochs = 0
            best_model = {
                'epoch': epoch + 1,
                'encoder_re_state_dict': encoder_re.state_dict(),
                'decoder_re_state_dict': decoder_re.state_dict(),
            }
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            break

    return best_model


def test_pre_model(
        encoderModel,
        decoderModel,
        config,
        best_model,
        test_dataloader,
        ):

    device=config['train']['device']

    encoder_re = encoderModel(config['encoder_re']).to(device)
    decoder_re = decoderModel(config['decoder_re']).to(device)
    encoder_re.load_state_dict(best_model['encoder_re_state_dict'])
    decoder_re.load_state_dict(best_model['decoder_re_state_dict'])

    total_metrics = evaluate(encoder_re, 
                             decoder_re, 
                             test_dataloader, 
                             device)
    
    return total_metrics
