"""
subJob --env light2 --gpu_type 'a100-pcie-80gb' --long --memory 250 main_init_model.py
"""

import argparse
import os
from dataclasses import dataclass

import pytorch_lightning as pl
from data.InitalAdoptDataset import CellData
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from MAPS.NoisyImmunolabeling.models import FineTuneModel as Model


@dataclass
class Sample:
    file: str
    path_nhs: str
    path_pred_outline: str
    path_pred_inner: str


def main(train_files, test_files):
    args = argparse.ArgumentParser()
    args.add_argument("--run_name", type=str)
    args.add_argument("--path_result", type=str)
    args.add_argument("--lr", type=float, default=1e-4)
    args.add_argument("--weight_decay", type=float, default=1e-4)
    # args.add_argument('--early_stopping', action='store_true')
    args.add_argument("--device", type=int, default=0)
    args.add_argument("--max_epochs", type=int, default=20)
    args.add_argument("--batch_size", type=int, default=4)
    args = args.parse_args()

    pl.seed_everything(123456)
    data = CellData(
        train_files=train_files,
        test_files=test_files,
        training_size=(16, 512, 512),
        data_stride=(8, 256, 256),
        extract_channel=1,
        batch_size=args.batch_size,
        outline_dilation=0,
    )
    print(f"Length training data: {len(data.train_data)}")

    model = Model(lr=args.lr, weight_decay=args.weight_decay)
    logger = pl.loggers.TensorBoardLogger(args.path_result, name=args.run_name)
    # if args.early_stopping:
    #     early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=5)
    checkpoint_callback = ModelCheckpoint(every_n_epochs=2, save_top_k=-1)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=[args.device],
        logger=logger,
        enable_progress_bar=False,
        callbacks=[
            # early_stop,
            checkpoint_callback,
            LearningRateMonitor(logging_interval="epoch"),
        ],
    )

    trainer.fit(model, data)

    with open(os.path.join(logger.log_dir, "params.txt"), "w") as f:
        f.write(f"Files: {train_files}\n")


if __name__ == "__main__":
    path_nhs_T1 = "/well/rittscher/projects/PanVision/data/FullStacks/Originals/MouseKidney_April2024"
    path_nhs_T2 = "/well/rittscher/projects/PanVision/data/FullStacks/Originals/MouseKidney_August2024_T2"
    path_nhs_T3 = "/well/rittscher/projects/PanVision/data/FullStacks/Originals/MouseKidney_August2024"

    path_pred_outline_WT1 = "/well/rittscher/users/jyo949/AntiBodySegKidney/results/combined/"
    path_pred_outline_WT2 = "/well/rittscher/users/jyo949/AntiBodySegKidney/logs/Model_August_WT_L2_0_Weight_5_T2Data_AllFiles_Thres123/version_0/"
    path_pred_outline_WT3 = "/well/rittscher/users/jyo949/AntiBodySegKidney/logs/Model_August_WT_L2_0_Weight_5_AugustData_AllFiles_Thres123/version_0/"
    path_pred_inner_WT1 = "/well/rittscher/users/jyo949/tmp/KidneyTest/MajorityVote/"
    path_pred_inner_WT2 = "/well/rittscher/users/jyo949/tmp/KidneyTest_AugustT2/MajorityVote"
    path_pred_inner_WT3 = "/well/rittscher/users/jyo949/tmp/KidneyTest_AugustT3/MajorityVote/"

    path_pred_outline_AKI1 = "/well/rittscher/users/jyo949/AntiBodySegKidney/results/combined/"
    path_pred_outline_AKI2 = "/well/rittscher/users/jyo949/AntiBodySegKidney/logs/Model_August_AKI_L2_0_Weight_5_T2Data_AllFiles_Thres133/version_0/"
    path_pred_outline_AKI3 = "/well/rittscher/users/jyo949/AntiBodySegKidney/logs/Model_August_AKI_L2_0_Weight_5_AugustData_AllFiles_Thres134/version_1/"
    path_pred_inner_AKI1 = "/well/rittscher/users/jyo949/tmp/KidneyTest/MajorityVote/"
    path_pred_inner_AKI2 = "/well/rittscher/users/jyo949/tmp/KidneyTest_AugustT2/MajorityVote"
    path_pred_inner_AKI3 = "/well/rittscher/users/jyo949/tmp/KidneyTest_AugustT3/MajorityVote/"

    files = [
        Sample("AKI1_D6_60X.tif", path_nhs_T1, path_pred_outline_AKI1, path_pred_inner_AKI1),
        Sample("WT2_60X_0.tif", path_nhs_T2, path_pred_outline_WT2, path_pred_inner_WT2),
        Sample("WT3_Mar_60X_0.tif", path_nhs_T3, path_pred_outline_WT3, path_pred_inner_WT3),
    ]

    test_files = [
        Sample("AKI2_D6_60X.tif", path_nhs_T1, path_pred_outline_AKI1, path_pred_inner_AKI1),
        Sample("WT3_60X_4.tif", path_nhs_T2, path_pred_outline_WT2, path_pred_inner_WT2),
        Sample("WT2_Mar_60X_6.tif", path_nhs_T3, path_pred_outline_WT3, path_pred_inner_WT3),
    ]

    main(files, test_files)
