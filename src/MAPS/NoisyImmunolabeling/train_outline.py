"""subJob --env light2 --gpu_type 'a100-pcie-80gb' --long --memory 120 train_outline.py"""

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from data.Data3D import BasicDataset
from data.Data3D import build_data as build_data, FileRecord, read_data_config
from torch.utils.data import DataLoader
from models.BaseModel3D import OutlineModel as UNet3D

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--train_dataset_config",
    type=str,
    help="""JSON file specifying the training dataset. Example in data/Data3D.py looking like
    [
        {
            "file_name": "WT1_D6_60X.tif",
            "path_base": "/<ourPath>/MouseKidney_April2024/",
            "ab_threshold": 115,
            "nhs_lower": 110,
            "nhs_upper": 0.995
        }
]""",
)
argparser.add_argument("--prediction_dataset_config", type=str, help="JSON file specifying the prediction dataset")
argparser.add_argument("--path_output", type=str, help="Path where to store the predictions")
argparser.add_argument("--name", type=str, help="Name of the current training run")
argparser.add_argument("--n_epochs", type=int, default=6)
argparser.add_argument("--id_device", type=int, default=0)
argparser.add_argument("--batch_size", type=int, default=8)
argparser.add_argument("--debug", type=bool, default=False)
argparser.add_argument("--compile", action="store_true")
argparser.add_argument("--label_smoothing", type=float, default=0.05)
argparser.add_argument("--lr", type=float, default=1e-3)
argparser.add_argument("--loss", type=str, default="BCEWithLogitsLoss")
argparser.add_argument("--l2", type=float, default=0)
argparser.add_argument("--thres_upper", type=float, default=0.25)
argparser.add_argument("--thres_lower", type=float, default=0.25)
argparser.add_argument("--pos_weight", type=float, default=5)

args = argparser.parse_args()


print("Building data...")
train_dataset = build_data(args.train_dataset_config)
train_loader = DataLoader(train_dataset, num_workers=8, batch_size=args.batch_size, shuffle=True)
print(
    "Building data finished! Length of train loader: ",
    len(train_loader),
)


model = UNet3D(
    train_step_max_thres=int(len(train_loader) * args.n_epochs * 0.8),
    train_steps_warm_up=int(len(train_loader) * args.n_epochs * 0.2),
    threshold_upper=args.thres_upper,
    threshold_lower=args.thres_lower,
    label_smoothing=args.label_smoothing,
    loss=args.loss,
    lr=args.lr,
    L2=args.l2,
    pos_weight=args.pos_weight,
)

if args.compile:
    model = torch.compile(model)

logger = loggers.TensorBoardLogger(save_dir=args.path_output, name=args.name)
trainer = pl.Trainer(
    accelerator="gpu",
    devices=[args.id_device],
    max_epochs=args.n_epochs,
    fast_dev_run=args.debug,
    enable_progress_bar=False,
    logger=logger,
    callbacks=[
        ModelCheckpoint(
            every_n_epochs=1,
            save_top_k=-1,
        )
    ],
)
trainer.fit(model=model, train_dataloaders=train_loader)

with open(os.path.join(logger.log_dir, "params.txt"), "w") as f:
    f.write(f"N_EPOCHS: {args.n_epochs}\n")
    f.write(f"ID_DEVICE: {args.id_device}\n")
    f.write(f"BATCH_SIZE: {args.batch_size}\n")
    f.write(f"LOSS: {args.loss}\n")
    f.write(f"LR: {args.lr}\n")
    f.write(f"LABEL_SMOOTHING: {args.label_smoothing}\n")
    f.write(f"L2: {args.l2}\n")
    f.write(f"THRES_UPPER: {args.thres_upper}\n")
    f.write(f"THRES_LOWER: {args.thres_lower}\n")
    f.write(f"POS_WEIGHT: {args.pos_weight}\n")


# Plot some the training progress for quick overview
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].plot(np.arange(len(model.losses)), model.losses)
ax[0].set_title("Loss")
ax[1].plot(np.arange(len(model.n_added_pix)), model.n_added_pix)
ax[1].set_title("Added pixels")
ax[2].plot(np.arange(len(model.n_ignored_antibodies)), model.n_ignored_antibodies)
ax[2].set_title("Ignored antibodies")
plt.savefig(os.path.join(logger.log_dir, "losses.png"))
train_loader = None

# Make a prediction for the prediction data and save several predicted slices
prediction_data_config = read_data_config(args.prediction_dataset_config)

for record in prediction_data_config:
    data = BasicDataset(
        img_records=[record],
        training_size=(16, 256, 256),  # Check your GPU memory
        data_stride=(8, 128, 128),
        mode="test",
    )
    model.make_full_pred(
        data.img_list[0], file_name=record.file_name.replace(".tif", "") + "pred", path=args.path_output
    )
