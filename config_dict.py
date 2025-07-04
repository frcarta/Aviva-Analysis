# epochsconfig = {
# TODO implement this as yaml
config_OLD = {
    # Training Configuration
    "validation": True,
    # Satellite configuration
    "satellite": "PRISMA",
    "ratio": 6,
    "nbits": 16,
    # Training settings
    "save_weights": True,
    "save_weights_path": "weights",
    "save_training_stats": False,
    # Training hyperparameters
    "learning_rate": 0.00001,
    "beta_1": 0.9,
    "beta_2": 0.999,
    "epochs": 200,
    "batch_size": 4,  # TODO implement batches in our network to speed things up
    # 'semi_width': 18,
    "semi_width": 6,
    "alpha_1": 0.5,
    "alpha_2": 0.25,
    "first_iter": 20,
    "epoch_nm": 15,
    "sat_val": 80,
    "net_scope": 6,
    "ms_scope": 1,
}

config = {
    # Training Configuration
    "validation": True,
    # Satellite configuration
    "satellite": "PRISMA",
    # "ratio": 6,
    "nbits": 16,
    # Training settings
    "save_weights": True,
    "save_weights_path": "weights",
    "save_training_stats": False,
    # Training hyperparameters
    "learning_rate": 0.00001,
    "beta_1": 0.9,
    "beta_2": 0.999,
    "epochs": 200,
    "batch_size": 4,  # TODO implement batches in our network to speed things up
    # 'semi_width': 18,
    "semi_width": 6,
    "alpha_1": 0.5,
    "alpha_2": 0.25,
    "first_iter": 20,
    "epoch_nm": 15,
    "sat_val": 80,
    "net_scope": 6,
    "ms_scope": 1,
    # ---------------------------------
    "input": "Datasets/aviva_eye_aligned_r6_rwl.mat",
    "ratio": 6,
    "out_dir": "Outputs",
    "gpu_number": 0,
    "use_cpu": False,
    # "learning_rate": -1.0,
    "pretrained": False,
    "epochs": 2000,
    "min_epochs": 100,
    "patience": 50,
    "delta": 0.5e-2,
    "relative": True,  # early stopping criterion uses realtive or absolute delta
    "alpha": 5,  # weight of the struct loss wrt spectral loss
    "learning_rate": 0.0001,
    "net_scope": 3,  # np.floor(half of receptive field), depends on convolutional layers
    # "save_path": "",
    "mtf_kernel_size": 15,  # must be greater than 2*ratio and odd
}
