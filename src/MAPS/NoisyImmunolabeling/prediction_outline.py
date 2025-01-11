import os
import torch
import argparse
from data.Data3D import BasicDataset, read_data_config
from models.BaseModel3D import UNet as UNet3D


argparser = argparse.ArgumentParser()
argparser.add_argument('--path_model_checkpoint', type=str)
argparser.add_argument('--path_target', type=str)
argparser.add_argument('--prediction_dataset_config', type=str, help='JSON file specifying the prediction dataset')
argparser.add_argument('--compile', type=bool, default=False)
argparser.add_argument('--cube_width', type=int, default=512)
argparser.add_argument('--cube_depth', type=int, default=32)
argparser.add_argument('--stride_width', type=int, default=256)
argparser.add_argument('--stride_depth', type=int, default=16)
args = argparser.parse_args()

model = UNet3D()
model = model.load_from_checkpoint(args.path_model_checkpoint)

if args.compile:
    model = torch.compile(model)


# Make a prediction for the prediction data and save several predicted slices
prediction_data_config = read_data_config(args.prediction_dataset_config)

if not os.path.exists(args.path_target):
    os.makedirs(args.path_target)

for record in prediction_data_config:
    data = BasicDataset(
        img_records=[record],
        training_size=(args.cube_depth, args.cube_width, args.cube_width),  # Check your GPU memory
        data_stride=(args.stride_depth, args.stride_width, args.stride_width),
        mode='test',
    )
    model.make_full_pred(
        data.img_list[0], file_name=record.file_name.replace('.tif', '') + 'pred', path=args.path_target
    )
