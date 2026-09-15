
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
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
    encoder_pred,
    decoder_pred,
    encoder_re,
    decoder_re,
    proj_head,
    pred_head,
    val_dataloader,
    input_time_len,
    output_time_len,
    device,
):

    mae = mse = mape = r2 = corr = 0.0
    batch_loss_align_z = 0.0
    batch_loss_pred_X_t1 = 0.0

    encoder_pred.eval()
    decoder_pred.eval()
    encoder_re.eval()
    decoder_re.eval()
    proj_head.eval()
    pred_head.eval()

    criterion = nn.MSELoss()

    with torch.no_grad():
        for X in tqdm(val_dataloader, desc="Batches val", position=2, leave=False):
            X = X.to(device)

            X_t, X_t1 = X[:, :input_time_len], X[:, -output_time_len:]
            
            z_re, aux = encoder_re(X_t1, mask_ratio=0.0)

            z_pred_1, _= encoder_pred(X_t)
            z_pred_2 = pred_head(z_pred_1)

            loss_align_z = criterion(z_pred_2, z_re)

            pred_full_x, _ = decoder_pred(z_pred_2)
            loss_pred_X_t1 = criterion(pred_full_x, X_t1)

            m = compute_metrics(y_true=X_t1, y_pred=pred_full_x)

            mae  += m["MAE"]
            mse  += m["MSE"]
            mape += m["MAPE"]
            r2   += m["R2"]
            corr += m["CORR"]

            batch_loss_pred_X_t1 += loss_pred_X_t1.item()
            batch_loss_align_z += loss_align_z.item()


    num_batches = len(val_dataloader)

    total_metrics = {
        "MAE":  mae  / num_batches,
        "MSE":  mse  / num_batches,
        "MAPE": mape / num_batches,
        "R2":   r2   / num_batches,
        "CORR": corr / num_batches,
        "loss_align_z": batch_loss_align_z / num_batches,
        "loss_pred_X_t1": batch_loss_pred_X_t1 / num_batches,
    }

    return total_metrics


def test_model(
        encoderModel_pred,
        decoderModel_pred,
        encoderModel_re,
        decoderModel_re,
        Projector,
        Predictor,
        config,
        best_model,
        test_dataloader,
        ):

    device=config['train']['device']

    encoder_pred = encoderModel_pred(config['encoder']).to(device)
    decoder_pred = decoderModel_pred(config['decoder']).to(device)
    proj_head = Projector(config['ProjectorHead']).to(device)
    pred_head = Predictor(config['PredictorHead']).to(device)

    encoder_re = encoderModel_re(config['encoder_re']).to(device)
    decoder_re = decoderModel_re(config['decoder_re']).to(device)

    encoder_pred.load_state_dict(best_model['encoder_pred_state_dict'])
    decoder_pred.load_state_dict(best_model['decoder_pred_state_dict'])
    encoder_re.load_state_dict(best_model['encoder_re_state_dict'])
    decoder_re.load_state_dict(best_model['decoder_re_state_dict'])
    proj_head.load_state_dict(best_model['proj_head_state_dict'])
    pred_head.load_state_dict(best_model['pred_head_state_dict'])

    input_time_len = config['decoder']['input_time']
    output_time_len = config['decoder']['output_time']

    total_metrics_h2 = evaluate(encoder_pred,
                                decoder_pred,
                                encoder_re, 
                                decoder_re,
                                proj_head,
                                pred_head,
                                test_dataloader, 
                                input_time_len,
                                output_time_len,
                                device,
                                )
    
    return total_metrics_h2


def calculate_feature_kl_loss(student_feature, teacher_feature, tau=0.5, dim=1):

    with torch.no_grad():
        t_prob = nn.functional.softmax(teacher_feature / tau, dim=dim)

    s_log_prob = nn.functional.log_softmax(student_feature / tau, dim=dim)

    kl = nn.functional.kl_div(s_log_prob, t_prob, reduction='none')  # (B,P,D)
    kl = kl.sum(dim=dim)

    loss = kl.mean() * (tau ** 2)
    return loss



def train_H(
        encoderModel_pred,
        decoderModel_pred,
        encoderModel_re,
        decoderModel_re,
        Projector,
        Predictor,
        config,
        task_name,
        train_dataloader,
        val_dataloader,
        pre_model_path,
):

    run_dir = f"anonymous/exp1_{task_name}/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=run_dir)
    device = config['train']['device']

    encoder_pred = encoderModel_pred(config['encoder']).to(device)
    decoder_pred = decoderModel_pred(config['decoder']).to(device)

    proj_head = Projector(config['ProjectorHead']).to(device)
    pred_head = Predictor(config['PredictorHead']).to(device)

    encoder_re = encoderModel_re(config['encoder_re']).to(device)
    decoder_re = decoderModel_re(config['decoder_re']).to(device)

    checkpoint = torch.load(pre_model_path, map_location=device, weights_only=True)

    encoder_re.load_state_dict(checkpoint['encoder_re_state_dict'], strict=False)
    decoder_re.load_state_dict(checkpoint['decoder_re_state_dict'], strict=False)

    encoder_re.eval()
    decoder_re.eval()

    for p in encoder_re.parameters():
        p.requires_grad = False
    for p in decoder_re.parameters():
        p.requires_grad = False


    pred_head.load_state_dict(checkpoint['pred_head_state_dict'], strict=False)
    pred_head.eval()
    for p in pred_head.parameters():
        p.requires_grad = False

    encoder_pred.load_state_dict(checkpoint['encoder_re_state_dict'], strict=False)
    encoder_pred.eval()

    for p in encoder_pred.parameters():
        p.requires_grad = False

    opt_pred = optim.Adam(
        list(decoder_pred.parameters()),
        lr=config['train']['lr'],
        weight_decay=1e-6
    )

    criterion = nn.MSELoss()

    patience = config['train']["early_stop_patience"]
    min_delta = config['train']["early_stop_min_delta"]

    best_val_mse = None
    best_model = None
    bad_epochs = 0

    input_time_len = config['decoder']['input_time']
    output_time_len = config['decoder']['output_time']
    num_epochs = config['train']['epochs']

    for epoch in tqdm(range(num_epochs), desc="Epochs", position=0, leave=False):

        decoder_pred.train()

        epoch_loss_pred_X_t1 = 0.0
        epoch_loss_align_z = 0.0

        alpha = 0.0

        for X in tqdm(train_dataloader, desc="Batches", position=1, leave=False):
            X = X.to(device)

            X_t  = X[:, :input_time_len]
            X_t1 = X[:, input_time_len: input_time_len + output_time_len]

            with torch.no_grad():
                z_re, aux = encoder_re(X_t1, mask_ratio=0.0)

            opt_pred.zero_grad()

            z_pred_0, _ = encoder_pred(X_t)
            z_pred_1 = pred_head(z_pred_0)
            pred_full_x, x_dec_student = decoder_pred(
                z_pred_1,
                pre_tokens=z_re,
                alpha=alpha
            )

            loss_align_dec = calculate_feature_kl_loss(z_pred_1, z_re, tau=0.5, dim=1)
            loss_pred_X_t1 = criterion(pred_full_x, X_t1)

            total_pred_loss = loss_pred_X_t1
            total_pred_loss.backward()
            opt_pred.step()

            epoch_loss_pred_X_t1 += loss_pred_X_t1.item()
            epoch_loss_align_z += loss_align_dec.item()

        num_batches = len(train_dataloader)
        avg_epoch_pred_X_t1 = epoch_loss_pred_X_t1 / num_batches
        avg_epoch_loss_align_z = epoch_loss_align_z / num_batches

        writer.add_scalar("Train/loss_pred_X_t1_MSE", avg_epoch_pred_X_t1, epoch)
        writer.add_scalar("Train/loss_align_z_MSE", avg_epoch_loss_align_z, epoch)

        with torch.no_grad():
            total_metrics_h2 = evaluate(
                encoder_pred,
                decoder_pred,
                encoder_re,
                decoder_re,
                proj_head,
                pred_head,
                val_dataloader,
                input_time_len,
                output_time_len,
                device,
                alpha=alpha,
            )

        val_mae  = total_metrics_h2["MAE"]
        val_mse  = total_metrics_h2["MSE"]
        val_mape = total_metrics_h2["MAPE"]
        val_r2   = total_metrics_h2["R2"]
        val_corr = total_metrics_h2["CORR"]
        val_align= total_metrics_h2["loss_align_z"]
        val_pred_X_t1= total_metrics_h2["loss_pred_X_t1"]


        writer.add_scalar("Val/X_t1_MAE",   val_mae,  epoch)
        writer.add_scalar("Val/X_t1_MSE",   val_mse,  epoch)
        writer.add_scalar("Val/X_t1_MAPE",  val_mape, epoch)
        writer.add_scalar("Val/X_t1_R2",    val_r2,   epoch)
        writer.add_scalar("Val/X_t1_CORR",  val_corr, epoch)
        writer.add_scalar("Val/align_z",     val_align, epoch)
        writer.add_scalar("Val/val_pred_X_t1",     val_pred_X_t1, epoch)
        
        if best_val_mse is None or (best_val_mse - val_mse) > min_delta:
            best_val_mse = val_mse
            bad_epochs = 0

            best_model = {
                'epoch': epoch + 1,
                'encoder_pred_state_dict': encoder_pred.state_dict(),
                'decoder_pred_state_dict': decoder_pred.state_dict(),
                'encoder_re_state_dict': encoder_re.state_dict(),
                'decoder_re_state_dict': decoder_re.state_dict(),
                'proj_head_state_dict': proj_head.state_dict(),
                'pred_head_state_dict': pred_head.state_dict(),
            }
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            break

    writer.close()
    return best_model
