import torch.jit
import matplotlib.pyplot as plt
from pl_model import ActivationSys, QPIDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ProgressBar
from argparse import ArgumentParser


def main(args):
    """
    This function trains the pytorch lightning model found in pl_model file
    """
    qpi_dm = QPIDataModule()
    model = ActivationSys()
    model.load_state_dict(torch.load('trained_model_norm.pt'))

    # trainer = pl.Trainer.from_argparse_args(args)
    trainer = pl.Trainer(
                        auto_lr_find=True, auto_select_gpus=True, gpus=-1, accelerator='auto'
                        )
    trainer.tune(model=model, datamodule=qpi_dm)
    trainer.fit(model=model, datamodule=qpi_dm)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
