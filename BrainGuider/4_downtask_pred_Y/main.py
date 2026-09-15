import os
import yaml
import argparse
from model import encoderModel, Predictor, ClassifyDecoder
from train import train_downtask, test_model
from dataset import get_dataloader
import random
import numpy as np
import pandas as pd
import torch


def set_seed(seed: int = 42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def result_print_and_save(metrics_list, save_dir):

    acc_list  = [m['Accuracy']  for m in metrics_list]
    prec_list = [m['Precision'] for m in metrics_list]
    rec_list  = [m['Recall']    for m in metrics_list]
    f1_list   = [m['F1']        for m in metrics_list]
    auc_list  = [m['AUROC']     for m in metrics_list]

    acc_mean,  acc_std  = np.mean(acc_list),  np.std(acc_list)
    prec_mean, prec_std = np.mean(prec_list), np.std(prec_list)
    rec_mean,  rec_std  = np.mean(rec_list),  np.std(rec_list)
    f1_mean,   f1_std   = np.mean(f1_list),   np.std(f1_list)
    auc_mean,  auc_std  = np.mean(auc_list),  np.std(auc_list)

    rows = []
    for i, m in enumerate(metrics_list):
        rows.append({
            "Run":       i + 1,
            "Accuracy":  m["Accuracy"],
            "Precision": m["Precision"],
            "Recall":    m["Recall"],
            "F1":        m["F1"],
            "AUROC":     m["AUROC"],
        })

    rows.append({
        "Run":       "Mean",
        "Accuracy":  acc_mean,
        "Precision": prec_mean,
        "Recall":    rec_mean,
        "F1":        f1_mean,
        "AUROC":     auc_mean,
    })

    rows.append({
        "Run":       "Std",
        "Accuracy":  acc_std,
        "Precision": prec_std,
        "Recall":    rec_std,
        "F1":        f1_std,
        "AUROC":     auc_std,
    })

    df = pd.DataFrame(rows)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "metrics_results_cls.csv")
    df.to_csv(save_path, index=False)

parser = argparse.ArgumentParser()
task_list = ["Emotion", "Gambling", "Language", "Motor", "Nback", "Relation", "Social"]
task_num_classes= [3, 3, 3, 4, 3, 3, 3]

task_id = 0
task_name = task_list[task_id]
num_classes = task_num_classes[task_id]
time_chunk_length = 16

data_dir = "anonymous"
parser.add_argument('--data_dir', type=str, default="anonymous")
parser.add_argument('--pre_model_dir', type=str, default="anonymous")
parser.add_argument('--model_save_dir', type=str, default="anonymous")

args = parser.parse_args()
os.makedirs(args.model_save_dir, exist_ok=True)

project_root = os.path.dirname(os.path.dirname(__file__))
config_path = os.path.join(project_root, "4_downtask_pred_Y", "config", "base.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)
    config['decoder']['num_classes'] = num_classes
    keys = ["train", "encoder", "decoder", "PredictorHead"]
    output = "\n".join(f"{key}: {config.get(key, '')}" for key in keys)

save_config_path = os.path.join(args.model_save_dir, "config_used.yaml")
with open(save_config_path, "w") as f:
    yaml.dump(config, f)

metrics_list = []
for i in range(5):
    seed = 2024 + i
    dataloader_train, dataloader_val, dataloader_test = get_dataloader(data_dir=args.data_dir, batch_size=config['train']['batch_size'], seed=seed)

    set_seed(seed=seed)
    best_model = train_downtask(
        encoderModel=encoderModel,
        predictor = Predictor,
        classifyDecoder=ClassifyDecoder,
        config=config,
        task_name=task_name,
        pre_model_dir=args.pre_model_dir,
        train_dataloader = dataloader_train,
        val_dataloader = dataloader_val,
    )

    metrics = test_model(
        encoderModel=encoderModel,
        predictor = Predictor,
        classifyDecoder=ClassifyDecoder,
        config=config,
        best_model = best_model,
        test_dataloader = dataloader_test,
    )

    metrics_list.append(metrics)
    save_path = os.path.join(args.model_save_dir, f"pred_y_best_model_%s.pt" % (i+1))
    torch.save(best_model, save_path)

result_print_and_save(metrics_list, args.model_save_dir)
