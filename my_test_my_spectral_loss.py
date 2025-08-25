import argparse
import gc
import os
from datetime import datetime
import matplotlib.pyplot as plt


import numpy as np
import scipy.io as io
import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm

from network import *
from loss import SpectralLoss, MySpectralLoss, StructuralLoss

from tools.spectral_tools import gen_mtf, normalize_prisma, denormalize_prisma
from tools import pytorch_degradation
from tools.early_stopping import EarlyStopping

from dataset import open_mat
from config_dict import config
from tools.cross_correlation import local_corr_mask

import h5py


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g., '20250612_142735'

# TODO put this in a yml file please


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
    ms = ms.to(device) if ms is not None else None
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

    criterion_spec = MySpectralLoss(ratio, device).to(device)

    criterion_struct = StructuralLoss(ratio).to(device)

    optim = torch.optim.Adam(net.parameters(), lr=learning_rate)

    history_loss_spec = []
    history_loss_struct = []

    alpha = args["alpha"]

    fused = []

    for band_number in range(ms_lr.shape[1]):

        # Reinitialize early stopping for each band
        early_stopping = EarlyStopping(
            patience=config["patience"],
            delta=config["delta"],
            relative=config["relative"],
        )

        band_lr = ms_lr[:, band_number : band_number + 1, :, :].to(device)
        if config["interpolation"] is None:
            band = ms[:, band_number : band_number + 1, :, :].to(
                device
            )  # selects band_number band but keeps the channel dimension
        else:
            band = F.interpolate(
                band_lr,
                size=(pan.shape[2], pan.shape[3]),
                mode=config["interpolation"],
                # align_corners=False,
            ).to(device)
        # scale_factor=config["ratio"],

        # Aux data generation
        inp = torch.cat([band, pan], dim=1)
        inp = pad(inp)

        threshold = local_corr_mask(inp, ratio, sensor, device, config["semi_width"])

        ft_epochs = int(args["epochs"])

        min_loss = torch.inf
        print("Band {} / {}".format(band_number + 1, ms_lr.shape[1]))
        pbar = tqdm(range(ft_epochs))

        for epoch in pbar:

            pbar.set_description("Epoch %d/%d" % (epoch + 1, ft_epochs))

            net.train()
            optim.zero_grad()

            outputs = net(inp)  # input is padded, net removes the padding

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

        net.load_state_dict(torch.load(os.path.join("temp", "R-PNN_best_model.tar")))

        net.eval()
        fused.append(net(inp).detach().cpu())

    fused = torch.cat(fused, 1).squeeze().numpy().transpose(1, 2, 0)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    wl = wl.cpu().numpy().squeeze()

    try:
        save_path = os.path.join(
            out_dir, basepath.split(os.sep)[-1].split(".")[0] + "_R-PNN.mat"
        )
        io.savemat(save_path, {"I_MS": fused, "Wavelengths": wl})
    except Exception as e:
        save_path = os.path.join(
            out_dir, basepath.split(os.sep)[-1].split(".")[0] + "_R-PNN.h5"
        )
        with h5py.File(save_path, "w") as f:
            f.create_dataset("I_MS", data=fused)
            f.create_dataset("Wavelengths", data=wl)

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
