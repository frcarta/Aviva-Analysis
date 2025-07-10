import argparse
import gc
import os
from datetime import datetime
import matplotlib.pyplot as plt


import numpy as np
import scipy.io as io
import torch
from torch import nn

from tqdm import tqdm

from network import *
from loss import MySpectralLoss, StructuralLoss

from tools.spectral_tools import gen_mtf, normalize_prisma, denormalize_prisma

from dataset import open_mat
from config_dict import config
from tools.cross_correlation import local_corr_mask


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g., '20250612_142735'

# TODO put this in a yml file please


class Config:
    def __init__(self, config):
        for key, value in config.items():
            setattr(self, key, value)


class EarlyStopping:
    def __init__(self, patience, delta=0, relative=False):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False
        self.relative = relative

    def check_early_stop(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
            self.no_improvement_count = 0
            return

        if not self.relative:
            improvement = loss < self.best_loss - self.delta
        else:
            improvement = (self.best_loss - loss) > self.delta * self.best_loss

        if improvement:
            self.best_loss = loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True


def test_r_pnn(args):

    # Paths and env configuration
    basepath = args["input"]
    method = "R-PNN"
    out_dir = os.path.join(args["out_dir"], method, timestamp)

    gpu_number = args["gpu_number"]
    use_cpu = args["use_cpu"]

    # Training hyperparameters
    learning_rate = args["learning_rate"]

    # Satellite configuration
    sensor = config["satellite"]
    ratio = args["ratio"]

    # Environment Configuration
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)

    # Devices definition
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and not use_cpu else "cpu"
    )

    if sensor == "PRISMA":
        normalize = normalize_prisma
        denormalize = denormalize_prisma
    else:
        raise "Satellite not supported"

    # Open the image

    pan, ms_lr, ms, _, wl = open_mat(basepath)

    # move to device
    ms_lr = ms_lr.to(device)
    ms = ms.to(device)
    pan = pan.to(device)

    net_scope = args["net_scope"]
    pad = nn.ReflectionPad2d(net_scope)

    # Torch configuration
    net = R_PNN_model3x3_res(scope=net_scope).to(device)

    if args["pretrained"]:
        weight_path = os.path.join("weights", "R-PNN_" + sensor + ".tar")
        if os.path.exists(weight_path):
            net.load_state_dict(torch.load(weight_path, map_location=device))
            print("Pretrained weights loaded.")
        else:
            raise FileNotFoundError(f"Pretrained weights not found at {weight_path}")
    else:
        print("Skipping weight loading — training from scratch.")

    # criterion_spec = SpectralLoss(
    # gen_mtf(ratio, sensor, kernel_size=61, nbands=1), ratio, device
    #    gen_mtf(ratio, sensor, kernel_size=args["mtf_kernel_size"], nbands=1),
    #    ratio,
    #    device,
    # ).to(device)
    criterion_spec = MySpectralLoss(ratio, device).to(device)
    criterion_struct = StructuralLoss(ratio).to(device)

    optim = torch.optim.Adam(net.parameters(), lr=learning_rate)

    history_loss_spec = []
    history_loss_struct = []

    # alpha = config["alpha_1"]
    alpha = args["alpha"]

    fused = []

    for band_number in range(ms.shape[1]):

        # Reinitialize early stopping for each band
        early_stopping = EarlyStopping(
            patience=config["patience"],
            delta=config["delta"],
            relative=config["relative"],
        )

        band = ms[:, band_number : band_number + 1, :, :].to(
            device
        )  # selects band_number band but keeps the channel dimension
        band_lr = ms_lr[:, band_number : band_number + 1, :, :].to(device)

        # Aux data generation
        inp = torch.cat([band, pan], dim=1)
        inp = pad(inp)

        threshold = local_corr_mask(inp, ratio, sensor, device, config["semi_width"])

        ft_epochs = int(args["epochs"])

        min_loss = torch.inf
        print("Band {} / {}".format(band_number + 1, ms.shape[1]))
        pbar = tqdm(range(ft_epochs))

        for epoch in pbar:

            pbar.set_description("Epoch %d/%d" % (epoch + 1, ft_epochs))

            net.train()
            optim.zero_grad()

            outputs = net(inp)

            loss_spec, _ = criterion_spec(outputs, band_lr)
            loss_struct, loss_struct_without_threshold = criterion_struct(
                outputs,
                pan,
                threshold[:, :, net_scope:-net_scope, net_scope:-net_scope],
            )

            loss = loss_spec + alpha * loss_struct

            loss.backward()
            optim.step()

            running_loss_spec = loss_spec.item()
            running_loss_struct = loss_struct_without_threshold

            history_loss_spec.append(running_loss_spec)
            history_loss_struct.append(running_loss_struct)

            if loss.item() < min_loss:
                min_loss = loss.item()
                if not os.path.exists("temp"):
                    os.makedirs(os.path.join("temp"))
                torch.save(
                    net.state_dict(), os.path.join("temp", "R-PNN_best_model.tar")
                )

            pbar.set_postfix(
                {"Spec Loss": running_loss_spec, "Struct Loss": running_loss_struct}
            )
            if epoch + 1 >= config["min_epochs"]:
                early_stopping.check_early_stop(loss.item())
                if early_stopping.stop_training:
                    break

        # DEBUGGING
        # if band_number == 0:
        #     a1 = threshold.detach().cpu().squeeze()
        #     a2 = inp.detach().cpu().squeeze()
        #     a3 = a3.detach().cpu().squeeze()
        #     a4 = outputs.detach().cpu().squeeze()

        #     fig, axs = plt.subplots(1,5, figsize = (10,5))
        #     axs[0].imshow(a1)
        #     axs[0].set_title("Threshold")
        #     axs[1].imshow(a2[0,:,:])
        #     axs[1].set_title("lr_band")
        #     axs[2].imshow(a2[1,:,:])
        #     axs[2].set_title("panchromatic")
        #     axs[3].imshow(a3)
        #     axs[3].set_title("outputs before training ")
        #     axs[4].imshow(a4)
        #     axs[4].set_title("outputs after training")

        #     fig1, axs1 = plt.subplots(1,2, figsize = (10,5))
        #     axs1[0].plot(history_loss_struct)
        #     axs1[0].set_title("history_loss_struct")
        #     axs1[1].plot(history_loss_spec)
        #     axs1[1].set_title("history_loss_spec")
        #     plt.tight_layout()
        #     plt.show()

        net.load_state_dict(torch.load(os.path.join("temp", "R-PNN_best_model.tar")))

        net.eval()
        fused.append(net(inp).detach().cpu())

    fused = torch.cat(fused, 1)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    save_path = os.path.join(
        out_dir, basepath.split(os.sep)[-1].split(".")[0] + "_R-PNN.mat"
    )
    io.savemat(save_path, {"I_MS": fused})
    history = {"loss_spec": history_loss_spec, "loss_struct": history_loss_struct}
    io.savemat(
        os.path.join(
            out_dir, basepath.split(os.sep)[-1].split(".")[0] + "_R-PNN_stats.mat"
        ),
        history,
    )

    torch.cuda.empty_cache()
    gc.collect()

    return


test_r_pnn(config)
