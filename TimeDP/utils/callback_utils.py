# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import numpy as np
import time
import torch
import pytorch_lightning as pl

from packaging import version
from omegaconf import OmegaConf
import wandb
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

from ldm.util import instantiate_from_config
import matplotlib.pyplot as plt

class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass

def plot_naming(k, bi, ni):  # the ni-th row, bi-th column
    name = ''
    if k == 'diffusion_row':
        name = f'sample{bi} diffstep {ni*200}'
    return name


class TSLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=False, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step


    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx,
                  key_list, dm, logger=None):
        root = os.path.join(save_dir, "images", split)
        image_dict = {}
        for k in images:  # assume inverse normalization has been applied
            grid = images[k]
            
            grid = grid.numpy()  # shape: num_samples, channels, window
            if len(grid.shape) == 3:
                b, c, w = grid.shape  # batchsize, channels, window
                for i in range(b):  
                    grid[i] = dm.inverse_transform(grid[i], data_name=key_list[i]) 
                fig, axs = plt.subplots(c, b, figsize=(b * 4, c * 4))  # c rows, b columns 
                for bi in range(b):  # transposed plotting
                    if c == 1:  # typically 1 x 8
                        axs[bi].plot(grid[bi, 0])
                    else:
                        for ci in range(c):
                            axs[ci, bi].plot(grid[bi, ci])
            elif len(grid.shape) == 4:  # compare across rows, so batchsize as num of columns
                n, b, c, w = grid.shape
                for i in range(b):  
                    grid[:,i] = dm.inverse_transform(grid[:,i], data_name=key_list[i]) 
                fig, axs = plt.subplots(n, b, figsize=(b * 4, n * 4))  # n rows, b columns
                for bi in range(b):
                    if n == 1:
                        for ci in range(c):
                            axs[bi].plot(grid[0, bi, ci])
                            axs[bi].set_title(plot_naming(k, bi, n))
                    else:
                        for ni in range(n):
                            for ci in range(c):
                                axs[ni, bi].plot(grid[ni, bi, ci])
                                axs[ni, bi].set_title(plot_naming(k, bi, ni))
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            plt.suptitle(filename)
            image_dict[k] = wandb.Image(fig)
            plt.close()
        logger.experiment.log(image_dict, step=global_step)

    def log_img(self, pl_module, batch, batch_idx, split="train", n_row=8):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        # self.log_steps = [1000, check_idx]
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, n_row=n_row, split=split, **self.log_images_kwargs)
                key_list = pl_module.trainer.datamodule.key_list
                batch_key_list = []
                for i in range(n_row):
                    batch_key_list.append(key_list[batch['data_key'][i].detach().cpu().numpy()])

            for k in images:
                if k != "samples_swapping" and k != "samples_swapping_partial":  # TODO: change to swapping intercept
                    N = min(images[k].shape[0], self.max_images)
                    images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)  # should clamp to [0,1]? or modify data loader to [-1,1]
                    else:
                        images[k] = torch.clamp(images[k], -2., 2.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx,
                           batch_key_list, pl_module.trainer.datamodule,logger=pl_module.logger)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        self.start_time = time.time()
        # Reset the memory use counter (only for CUDA)
        if torch.cuda.is_available():
            try:
                device_idx = trainer.strategy.root_device.index
                if device_idx is not None:
                    torch.cuda.reset_peak_memory_stats(device_idx)
                    torch.cuda.synchronize(device_idx)
            except (AttributeError, RuntimeError):
                pass

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.start_time
        max_memory = 0

        if torch.cuda.is_available():
            try:
                device_idx = trainer.strategy.root_device.index
                if device_idx is not None:
                    torch.cuda.synchronize(device_idx)
                    max_memory = torch.cuda.max_memory_allocated(device_idx) / 2 ** 20
            except (AttributeError, RuntimeError):
                pass

        try:
            rank_zero_info(f"Epoch time: {epoch_time:.2f} seconds")
            if max_memory > 0:
                rank_zero_info(f"Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass
        
def prepare_trainer_configs(nowname, logdir, opt, lightning_config, ckptdir, model, now, cfgdir, config, trainer_opt):
    trainer_kwargs = dict()

    # default logger configs
    default_logger_cfgs = {
        "wandb": {
            "target": "pytorch_lightning.loggers.WandbLogger",
            "params": {
                "name": f"{nowname}_{now}",
                "save_dir": logdir,
                "offline": opt.debug,
                "id": f"{nowname}_{now}",
                "project": "TimeDP",
            }
        }
    }
    default_logger_cfg = default_logger_cfgs["wandb"]
    if "logger" in lightning_config:
        logger_cfg = lightning_config.logger
    else:
        logger_cfg = OmegaConf.create()
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)
    # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
    # specify which metric is used to determine best models
    default_modelckpt_cfg = {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {
            "dirpath": ckptdir,
            "filename": "{epoch:06}-{val/loss_simple_ema:.4f}",
            "verbose": True,
            "save_last": True,
            "auto_insert_metric_name": False
        }
    }
    if hasattr(model, "monitor"):
        print(f"Monitoring {model.monitor} as checkpoint metric.")
        default_modelckpt_cfg["params"]["monitor"] = model.monitor
        default_modelckpt_cfg["params"]["save_top_k"] = 3
        default_modelckpt_cfg["params"]["mode"] = "min"
    if default_modelckpt_cfg["params"]["monitor"] == "train/step_num":
        default_modelckpt_cfg["params"]["every_n_train_steps"] = 2000
        default_modelckpt_cfg["params"]["every_n_epochs"] = None
        default_modelckpt_cfg["params"]["filename"] = "{step:09}"
        default_modelckpt_cfg["params"]["mode"] = "max"


    if "modelcheckpoint" in lightning_config:
        modelckpt_cfg = lightning_config.modelcheckpoint
    else:
        modelckpt_cfg =  OmegaConf.create()
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
    print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
    if version.parse(pl.__version__) < version.parse('1.4.0'):
        trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

    # add callback which sets up log directory
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "utils.callback_utils.SetupCallback",
            "params": {
                "resume": opt.resume,
                "now": now,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": config,
                "lightning_config": lightning_config,
            }
        },
        "learning_rate_logger": {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
            }
        },
        "cuda_callback": {
            "target": "utils.callback_utils.CUDACallback"
        },
    }
    if version.parse(pl.__version__) >= version.parse('1.4.0'):
        default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()

    if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
        print(
            'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
        default_metrics_over_trainsteps_ckpt_dict = {
            'metrics_over_trainsteps_checkpoint':
                {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                    'params': {
                        "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                        "filename": "{epoch:06}-{step:09}",
                        "verbose": True,
                        'save_top_k': -1,
                        'every_n_train_steps': 10000,
                        'save_weights_only': True
                    }
                    }
        }
        default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
    if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
        callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
    elif 'ignore_keys_callback' in callbacks_cfg:
        del callbacks_cfg['ignore_keys_callback']

    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
    return trainer_kwargs
