# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os, sys
from pytorch_lightning.trainer import Trainer
from utils.cli_utils import get_parser
from utils.init_utils import init_model_data_trainer
from utils.test_utils import test_model_with_dp, test_model_uncond, test_model_unseen


if __name__ == "__main__":
    
    data_root = os.environ['DATA_ROOT']

    parser = get_parser()
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--benchmark", type=bool, default=True)
    # parser = Trainer.add_argparse_args(parser)

    model, data, trainer, opt, logdir, melk = init_model_data_trainer(parser)

    # run
    if opt.train:
        try:
            trainer.logger.experiment.config.update(opt)
            trainer.fit(model, data)
        except Exception:
            melk()
            raise
    if not opt.no_test and not trainer.interrupted:
        if opt.uncond:
            test_model_uncond(model, data, trainer, opt, logdir)
        else:
            test_model_with_dp(model, data, trainer, opt, logdir)
            test_model_unseen(model, data, trainer, opt, logdir)

