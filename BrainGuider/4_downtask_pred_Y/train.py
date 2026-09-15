
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def compute_metrics(y_true, y_logits, average="macro"):

    B, T_out, C = y_logits.shape

    logits_flat = y_logits.reshape(B * T_out, C)
    y_flat      = y_true.reshape(B * T_out)

    y_prob = nn.functional.softmax(logits_flat, dim=-1).detach().cpu().numpy()
    y_pred = y_prob.argmax(axis=-1)
    y_true_np = y_flat.detach().cpu().numpy()

    acc  = accuracy_score(y_true_np, y_pred)
    prec = precision_score(y_true_np, y_pred, average=average, zero_division=0)
    rec  = recall_score(y_true_np, y_pred, average=average, zero_division=0)
    f1   = f1_score(y_true_np, y_pred, average=average, zero_division=0)

    try:
        auroc = roc_auc_score(
            y_true_np,
            y_prob,
            multi_class="ovr",
            average=average
        )
    except:
        auroc = float("nan")

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "AUROC": auroc
    }


def evaluate(encoder_pred, pred_head, decoder_classify, val_dataloader, time_len, device):

    encoder_pred.eval()
    pred_head.eval()
    decoder_classify.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for X, Y in tqdm(val_dataloader, desc="Batches val", position=2, leave=False):
            X, Y = X.to(device), Y.to(device)
            Xt   = X[:, :time_len]  
            Yt_1 = Y[:, time_len:] 

            z_pred_0, _ = encoder_pred(Xt)
            z_pred_1 = pred_head(z_pred_0)

            logits = decoder_classify(z_pred_1) 

            all_logits.append(logits.detach().cpu())
            all_labels.append(Yt_1.detach().cpu())

    y_logits_full = torch.cat(all_logits, dim=0)  
    y_true_full   = torch.cat(all_labels, dim=0)

    metrics = compute_metrics(y_true=y_true_full, y_logits=y_logits_full)

    return metrics


def train_downtask(
        encoderModel,
        predictor,
        classifyDecoder,
        config,
        task_name,
        pre_model_dir,
        train_dataloader,
        val_dataloader,
):
    device=config['train']['device']
    encoder_pred = encoderModel(config['encoder']).to(device)
    pred_head = predictor(config['PredictorHead']).to(device)
    decoder_classify = classifyDecoder(config['decoder']).to(device)

    ckpt_path = os.path.join(pre_model_dir, "train_3_best_model_1.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    encoder_pred.load_state_dict(ckpt['encoder_pred_state_dict'])
    encoder_pred.eval()

    for p in encoder_pred.parameters():
        p.requires_grad = False
    
    pred_head.load_state_dict(ckpt['pred_head_state_dict'])
    pred_head.eval()

    for p in pred_head.parameters():
        p.requires_grad = False

    optimizer = optim.Adam(
        list(decoder_classify.parameters()),
        lr=config['train']['lr'],
        weight_decay=1e-6
    )

    time_len = config['decoder']['input_time']
    patience = config['train']["early_stop_patience"]
    min_delta = config['train']["early_stop_min_delta"]

    best_val_acc = None
    best_model = None
    bad_epochs = 0

    for epoch in tqdm(range(config['train']['epochs']), desc="Epochs", position=0, leave=False):

        decoder_classify.train()
        epoch_loss = 0.0

        for X, Y in tqdm(train_dataloader, desc="Batches", position=1, leave=False):
 
            X, Y = X.to(device), Y.to(device)
            Xt   = X[:, :time_len]     
            Yt_1 = Y[:, time_len:]   

            with torch.no_grad():
                z_pred_0, _ = encoder_pred(Xt)
                z_pred_1 = pred_head(z_pred_0)

            logits = decoder_classify(z_pred_1)         
            B, T_out, C = logits.shape
            logits_flat = logits.reshape(B * T_out, C)
            y_flat      = Yt_1.reshape(B * T_out)

            loss_cls = nn.functional.cross_entropy(logits_flat, y_flat)

            optimizer.zero_grad()
            loss_cls.backward()
            optimizer.step()
            epoch_loss += loss_cls.item()

        metrics = evaluate(encoder_pred, pred_head, decoder_classify, val_dataloader, time_len, device)
        val_acc = metrics["Accuracy"]
        if best_val_acc is None or (best_val_acc - val_acc) > min_delta:
            best_val_acc = val_acc
            bad_epochs = 0
            best_model = {
                "epoch": epoch + 1,
                "encoder_pred_state_dict": encoder_pred.state_dict(),
                'pred_head_state_dict': pred_head.state_dict(),
                "decoder_classify_state_dict": decoder_classify.state_dict(),
            }
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            break
    return best_model


def test_model(
        encoderModel,
        predictor,
        classifyDecoder,
        config,
        best_model,
        test_dataloader,
        ):

    device=config['train']['device']
    time_len = config['decoder']['input_time']

    encoder_pred = encoderModel(config['encoder']).to(device)
    pred_head = predictor(config['PredictorHead']).to(device)
    decoder_classify = classifyDecoder(config['decoder']).to(device)

    encoder_pred.load_state_dict(best_model['encoder_pred_state_dict'])
    pred_head.load_state_dict(best_model['pred_head_state_dict'])
    decoder_classify.load_state_dict(best_model['decoder_classify_state_dict'])

    metrics = evaluate(encoder_pred, pred_head, decoder_classify, test_dataloader, time_len, device)

    return metrics
