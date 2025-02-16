import argparse
import json

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from MAPS.NoisyImmunolabeling.data import FineTuneCellData, Sample
from MAPS.NoisyImmunolabeling.models import FineTuneModel as Model
from MAPS.utils import SegmentatorNetwork, UNet


def main():
    pl.seed_everything(123456)
    args = argparse.ArgumentParser()
    args.add_argument("--run_name", type=str, default="V1")
    args.add_argument("--train_json", type=str, default="")
    args.add_argument("--test_json", type=str, default="")
    args.add_argument("--path_result", type=str, default="/well/rittscher/users/jyo949/MAPS_test/FineTuned_model")
    args.add_argument(
        "--path_init_model",
        type=str,
        default="/well/rittscher/users/jyo949/AntiBodySegKidney/MitoClustering/init_model/FirstModel/version_0/checkpoints/epoch=1-step=6272.ckpt",
    )
    args.add_argument("--lr", type=float, default=1e-4)
    args.add_argument("--weight_decay", type=float, default=1e-4)
    args.add_argument("--device", type=int, default=0)
    args.add_argument("--n_epochs", type=int, default=25)

    args.add_argument("--batch_size", type=int, default=4)
    args = args.parse_args()

    with open(args.train_json, "r") as f:
        train_files = json.load(f)
        train_files = [Sample(e["file"], e["path_nhs"], e["path_pred"]) for e in train_files]

    if args.test_json:
        with open(args.test_json, "r") as f:
            test_files = json.load(f)
            test_files = [Sample(e["file"], e["path_nhs"], e["path_pred"]) for e in train_files]
    else:
        test_files = []

    data = FineTuneCellData(
        train_files=train_files,
        test_files=test_files,
        training_size=(16, 512, 512),
        data_stride=(2, 256, 256),
        extract_channel=1,
        batch_size=args.batch_size,
    )
    print(f"Length training data: {len(data.train_data)}")

    model = Model.load_from_checkpoint(args.path_init_model, lr=args.lr, weight_decay=args.weight_decay, n_classes=5)
    model.segmentation_head = SegmentatorNetwork(n_classes=5, in_classes=32)  # New seg head
    model.embedding_network.decoder = UNet(
        encoder_channels=[1, 64, 128, 256, 512],  # You could experiment with larger models if necessary
        decoder_channels=[512, 256, 128, 64, 32, 1],
        residual=True,
        type="3D",
    ).decoder

    logger = pl.loggers.TensorBoardLogger(args.path_result, name=args.run_name)
    # early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=5)
    checkpoint_callback = ModelCheckpoint(every_n_epochs=5, save_top_k=-1)

    trainer = pl.Trainer(
        max_epochs=args.n_epochs,
        accelerator="gpu",
        devices=[0],
        logger=logger,
        enable_progress_bar=False,
        callbacks=[
            # early_stop,
            checkpoint_callback,
            LearningRateMonitor(logging_interval="epoch"),
        ],
    )

    trainer.fit(model, data)


if __name__ == "__main__":
    main()
