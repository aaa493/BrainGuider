import os
import yaml
import argparse
from model import encoderModel, decoderModel, ProjectionHead, Predictor
from pre_model import pre_encoderModel, pre_decoderModel
from train import train_H, test_model
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


def result_print_and_save(metrics_list, save_dir, name):
    metric_names = list(metrics_list[0].keys())
    stats_mean = {}
    stats_std  = {}
    for key in metric_names:
        values = [m[key] for m in metrics_list]
        stats_mean[key] = float(np.mean(values))
        stats_std[key]  = float(np.std(values))
    rows = []
    for i, m in enumerate(metrics_list):
        row = {"Run": i + 1}
        for key in metric_names:
            row[key] = m[key]
        rows.append(row)
    row_mean = {"Run": "Mean"}
    for key in metric_names:
        row_mean[key] = stats_mean[key]
    rows.append(row_mean)
    row_std = {"Run": "Std"}
    for key in metric_names:
        row_std[key] = stats_std[key]
    rows.append(row_std)
    df = pd.DataFrame(rows)
    save_path = os.path.join(save_dir, f"metrics_results_{name}.csv")
    df.to_csv(save_path, index=False)



parser = argparse.ArgumentParser()
task_list = ["REST1", "Emotion", "Gambling", "Language", "Motor", "Nback", "Relation", "Social"]

task_id = 0
task_name = task_list[task_id]
time_chunk_length = 16

data_dir_s100 = "anonymous"
parser.add_argument('--data_dir_s100', type=str, default="anonymous")
parser.add_argument('--model_save_dir', type=str, default="anonymous")
parser.add_argument('--pre_model_save_dir', type=str, default="anonymous")
args = parser.parse_args()

os.makedirs(args.model_save_dir, exist_ok=True)
project_root = os.path.dirname(os.path.dirname(__file__))  # 适用于模块化项目
config_path = os.path.join(project_root, "3_train", "config", "base.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)
    keys = ["train", "encoder", "decoder", "PredictorHead"]
    output = "\n".join(f"{key}: {config.get(key, '')}" for key in keys)

save_config_path = os.path.join(args.model_save_dir, "config_used.yaml")
with open(save_config_path, "w") as f:
    yaml.dump(config, f)

metrics_list = []
for i in range(5):
    seed = 2024 + i
    dataloader_train, dataloader_val, dataloader_test = get_dataloader(data_dir_s100=args.data_dir_s100, batch_size=config['train']['batch_size'], seed=seed)
                                                                       
    set_seed(seed=seed)
    best_model = train_H(
        encoderModel_pred=encoderModel,
        decoderModel_pred=decoderModel,
        encoderModel_re=pre_encoderModel,
        decoderModel_re=pre_decoderModel,
        Projector = ProjectionHead,
        Predictor = Predictor,
        config=config,
        task_name=task_name,
        train_dataloader = dataloader_train,
        val_dataloader = dataloader_val,
        pre_model_path = args.pre_model_save_dir,
    )

    total_metrics = test_model(
        encoderModel_pred=encoderModel,
        decoderModel_pred=decoderModel,
        encoderModel_re=pre_encoderModel,
        decoderModel_re=pre_decoderModel,
        Projector = ProjectionHead,
        Predictor = Predictor,
        config=config,
        best_model = best_model,
        test_dataloader = dataloader_test,
    )

    metrics_list.append(total_metrics)
    save_path = os.path.join(args.model_save_dir, f"train_3_best_model_%s.pt" % (i+1))
    torch.save(best_model, save_path)

result_print_and_save(metrics_list, args.model_save_dir, "s100_h2")
