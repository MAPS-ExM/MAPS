import torch
from torch.nn import DataParallel


class CustomDataParallel(DataParallel):
    def __getattr__(self, name):
        try:
            return super(CustomDataParallel, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def build_model(args):
    # Late import to avoid cirulcar imports
    from .BasicUNets import (
        BasicSmallRegressionUNet,
        BasicSmallUNet,
        BasicUNet3D,
        ConditionalGNUNet,
        GNSmallUNet,
        SmallUNeXt,
    )
    from .DoubleModel import DoubleModel
    
    if args.model_name == "BasicUNet3D":
        model = BasicUNet3D(n_classes=args.num_classes, loss_fn=args.selected_loss, device=args.device)
    elif args.model_name == "SmallUNet3D":
        model = BasicSmallUNet(
            n_classes=args.num_classes,
            input_channels=getattr(args, "input_channels", 1),
            loss_fn=args.selected_loss,
            residual=getattr(args, "residualUNet", False),
            device=args.device,
        )
    elif args.model_name == "UNetGN":
        model = GNSmallUNet(
            n_classes=args.num_classes,
            loss_fn=args.selected_loss,
            residual=getattr(args, "residualUNet", False),
            device=args.device,
        )
    elif args.model_name == "ConditionalGNUnet":
        model = ConditionalGNUNet(
            n_classes=args.num_classes,
            loss_fn=args.selected_loss,
            residual=getattr(args, "residualUNet", False),
            cat_emb_dim=getattr(args, "cat_emb_dim", 8),
            device=args.device,
        )
    elif args.model_name == "UNeXt":
        model = SmallUNeXt(
            n_classes=args.num_classes,
            loss_fn=args.selected_loss,
            residual=getattr(args, "residualUNet", False),
            device=args.device,
        )
    elif args.model_name == "SmallMaskedUnet":
        raise NotImplementedError()
    elif args.model_name == "DoubleModel":
        model = DoubleModel(
            n_classes=args.num_classes,
            loss_fn=args.selected_loss,
            input_channels=getattr(args, "input_channels", 1),
            mask_threshold=0.5,
            loss_weight=args.loss_weights,
            pretrain=getattr(args, "pretrain", ""),
            cat_emb_dim=getattr(args, "cat_emb_dim", 8),
            device=args.device,
        )
    elif args.model_name == "UNet_MaskedImageNet2D":
        from .UNetImageNet import MaskedUNet2DImageNet

        model = MaskedUNet2DImageNet(
            n_classes=args.num_classes,
            loss_fn=args.selected_loss,
            mask_threshold=0.5,
            loss_weight=args.loss_weights,
            device=args.device,
        )
    elif args.model_name == "SwinTransformer":
        from .SwinTransformer import SwinTransformer

        model = SwinTransformer(
            n_classes=args.num_classes,
            img_size=args.training_size,
            feature_size=args.feature_size,
            num_heads=getattr(args, "num_heads", ""),
            depth=getattr(args, "depth", ""),
            loss_fn=args.selected_loss,
            mask_threshold=0.5,
            device=args.device,
        )
    elif args.model_name == "SwinTransformerMasked":
        from .SwinTransformer import SwinTransformerMasked

        model = SwinTransformerMasked(
            n_classes=args.num_classes,
            img_size=args.training_size,
            feature_size=args.feature_size,
            loss_fn=args.selected_loss,
            pretrain=getattr(args, "pretrain", ""),
            mask_threshold=0.5,
            device=args.device,
        )
    elif args.model_name == "RegressionUNet3D":
        model = BasicSmallRegressionUNet(n_classes=args.num_classes, loss_fn=args.selected_loss, device=args.device)
    else:
        raise NotImplementedError(f"Unknown model name: {args.model_name}")

    if args.device_id == "all":
        model = CustomDataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])

    model.to(args.device)
    return model
