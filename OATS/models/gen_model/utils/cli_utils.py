# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from pytorch_lightning import Trainer

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-n","--name",type=str,const=True,default="",nargs="?",help="postfix for logdir")
    parser.add_argument("-b","--base",nargs="*",metavar="base_config.yaml",help="paths to base configs. Loaded from left-to-right.", default=list(),)
    parser.add_argument("-t","--train",type=str2bool,const=True,default=True,nargs="?",help="train",)
    parser.add_argument("-r","--resume",type=str2bool,const=True,default=False,nargs="?",help="resume and test",)
    parser.add_argument("--no-test",type=str2bool,const=True,default=False,nargs="?",help="disable test",)
    parser.add_argument("-d","--debug",type=str2bool,nargs="?",const=True,default=False,help="debug mode",)
    parser.add_argument("-s","--seed",type=int,default=23,help="seed for seed_everything",)
    parser.add_argument("-f","--postfix",type=str,default="",help="post-postfix for default name",)
    parser.add_argument("-l","--logdir",type=str,default="./logs",help="directory for logging dat shit",)
    parser.add_argument("--scale_lr",type=str2bool,nargs="?",const=True,default=False,help="scale base-lr by ngpu * batch_size * n_accumulate",)
    parser.add_argument("--ckpt_name",type=str,default="last",help="ckpt name to resume",)
    parser.add_argument("-sl","--seq_len", type=int, const=True, default=24,nargs="?", help="sequence length")
    parser.add_argument("-uc","--uncond", action='store_true', help="unconditional generation")
    parser.add_argument("-up","--use_pam", action='store_true', help="use prototype")
    parser.add_argument("-bs","--batch_size", type=int, const=True, default=128,nargs="?", help="batch_size")
    parser.add_argument("-nl","--num_latents", type=int, const=True, default=16,nargs="?", help="number of prototypes")
    parser.add_argument("-lr","--overwrite_learning_rate", type=float, const=True, default=None, nargs="?", help="learning rate")
    
    return parser

def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    # parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))
