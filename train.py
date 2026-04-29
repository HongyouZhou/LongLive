# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
from datetime import datetime
from omegaconf import OmegaConf
import wandb

from trainer import ScoreDistillationTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="", help="Path to the directory to save logs")
    parser.add_argument("--wandb-save-dir", type=str, default="", help="Path to the directory to save wandb logs")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--auto-resume", action="store_true", help="Resume from latest checkpoint in logdir (default: start fresh)")
    parser.add_argument("--no-one-logger", action="store_true", help="Disable One Logger (enabled by default)")

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize

    # Wandb run name: config filename + start-time stamp (yymmdd_HHMM).
    # The suffix only labels FRESH starts so concurrent runs of the same
    # config are visually distinct in the wandb dashboard. On --auto-resume
    # the trainer reads logdir/wandb_run_id.txt and skips passing `name`
    # to wandb.init, so the run keeps its original name across resubmits.
    config_basename = os.path.basename(args.config_path).split(".")[0]
    config.config_name = f"{config_basename}_{datetime.now().strftime('%y%m%d_%H%M')}"
    config.logdir = args.logdir
    config.wandb_save_dir = args.wandb_save_dir
    config.disable_wandb = args.disable_wandb
    config.auto_resume = args.auto_resume  # Default to False; pass --auto-resume to continue from logdir
    config.use_one_logger = not args.no_one_logger

    if config.trainer == "score_distillation":
        trainer = ScoreDistillationTrainer(config)
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()
