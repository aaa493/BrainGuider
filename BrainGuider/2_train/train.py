
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim


def calculate_feature_kl_loss(student_feature, teacher_feature, tau=0.5, dim=1):
    t_prob = nn.functional.softmax(teacher_feature / tau, dim=dim)
    s_log_prob = nn.functional.log_softmax(student_feature / tau, dim=dim)

    kl = nn.functional.kl_div(s_log_prob, t_prob, reduction='none')  # (B,P,D)
    kl = kl.sum(dim=dim)
    loss = kl.mean() * (tau ** 2)
    return loss


def train_H(
        encoderModel_pred,
        encoderModel_re,
        decoderModel_re,
        Predictor,
        config,
        task_name,
        train_dataloader,
        val_dataloader,
        pre_model_path,
):

    device = config['train']['device']

    encoder_pred = encoderModel_pred(config['encoder']).to(device)
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

    encoder_pred.load_state_dict(checkpoint['encoder_re_state_dict'], strict=False)
    encoder_pred.eval()

    for p in encoder_pred.parameters():
        p.requires_grad = False

    opt_pred = optim.Adam(
        list(pred_head.parameters()),
        lr=config['train']['lr'],
        weight_decay=1e-6
    )

    criterion_mse = nn.MSELoss()
    patience = config['train']["early_stop_patience"]
    min_delta = config['train']["early_stop_min_delta"]

    best_val_mse = None
    best_model = None
    bad_epochs = 0

    input_time_len = config['decoder']['input_time']
    output_time_len = config['decoder']['output_time']
    num_epochs = config['train']['epochs']

    for epoch in tqdm(range(num_epochs), desc="Epochs", position=0, leave=False):

        pred_head.train()
        epoch_loss_align_z = 0.0
        epoch_loss_kl = 0.0

        for X in tqdm(train_dataloader, desc="Batches", position=1, leave=False):
            X = X.to(device)

            X_t  = X[:, :input_time_len]
            X_t1 = X[:, input_time_len: input_time_len + output_time_len]

            with torch.no_grad():
                z_re, aux = encoder_re(X_t1, mask_ratio=0.0)

            opt_pred.zero_grad()
            z_pred_0, _ = encoder_pred(X_t)
            z_pred_1 = pred_head(z_pred_0)

            loss_align_kl = calculate_feature_kl_loss(z_pred_1, z_re, tau=0.5, dim=1)
            loss_align_dec = criterion_mse(z_pred_1, z_re)
            total_pred_loss = loss_align_dec + loss_align_kl

            total_pred_loss.backward()
            opt_pred.step()

            epoch_loss_align_z += loss_align_dec.item()
            epoch_loss_kl += loss_align_kl.item()

        with torch.no_grad():
            total_metrics_h2 = evaluate(
                encoder_pred,
                encoder_re,
                pred_head,
                val_dataloader,
                input_time_len,
                output_time_len,
                device,
            )

        val_align_mse= total_metrics_h2["loss_align_mse"]
        val_align_kl= total_metrics_h2["loss_align_kl"]
        
        if best_val_mse is None or (best_val_mse - (val_align_mse + val_align_kl)) > min_delta:
            best_val_mse = val_align_mse + val_align_kl
            bad_epochs = 0

            best_model = {
                'epoch': epoch + 1,
                'encoder_pred_state_dict': encoder_pred.state_dict(),
                'encoder_re_state_dict': encoder_re.state_dict(),
                'decoder_re_state_dict': decoder_re.state_dict(),
                'pred_head_state_dict': pred_head.state_dict(),
            }

            print(f"[✔] New best model at epoch {epoch+1}, Val_MSE={best_val_mse:.4f}")
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break
    return best_model


def evaluate(
    encoder_pred,
    encoder_re,
    pred_head,
    val_dataloader,
    input_time_len,
    output_time_len,
    device,
):

    encoder_pred.eval()
    encoder_re.eval()
    pred_head.eval()

    criterion_mse = nn.MSELoss()

    batch_loss_align_kl = 0.0
    batch_loss_align_mse = 0.0

    with torch.no_grad():
        for X in tqdm(val_dataloader, desc="Batches val", position=2, leave=False):
            X = X.to(device)
        
            X_t, X_t1 = X[:, :input_time_len], X[:, input_time_len: input_time_len+output_time_len]

            z_re, _ = encoder_re(X_t1, mask_ratio=0.0)

            z_pred_0, _= encoder_pred(X_t)
            z_pred_1 = pred_head(z_pred_0)

            loss_align_kl = calculate_feature_kl_loss(z_pred_1, z_re, tau=0.5, dim=1)
            loss_align_mse = criterion_mse(z_pred_1, z_re)

            batch_loss_align_kl += loss_align_kl.item()
            batch_loss_align_mse += loss_align_mse.item()


    num_batches = len(val_dataloader)

    total_metrics = {
        "loss_align_kl": batch_loss_align_kl / num_batches,
        "loss_align_mse": batch_loss_align_mse / num_batches,   
    }

    return total_metrics


def test_model(
        encoderModel_pred,
        encoderModel_re,
        Predictor,
        config,
        best_model,
        test_dataloader,
        ):

    device=config['train']['device']

    encoder_pred = encoderModel_pred(config['encoder']).to(device)
    pred_head = Predictor(config['PredictorHead']).to(device)
    encoder_re = encoderModel_re(config['encoder_re']).to(device)

    encoder_pred.load_state_dict(best_model['encoder_pred_state_dict'])
    pred_head.load_state_dict(best_model['pred_head_state_dict'])
    encoder_re.load_state_dict(best_model['encoder_re_state_dict'])

    input_time_len = config['decoder']['input_time']
    output_time_len = config['decoder']['output_time']

    total_metrics_h2 = evaluate(
                                encoder_pred,
                                encoder_re,
                                pred_head,
                                test_dataloader,
                                input_time_len,
                                output_time_len,
                                device,
                                )
    
    return total_metrics_h2

