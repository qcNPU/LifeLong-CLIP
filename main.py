# import torch
import argparse
import json
import os

from configuration import config
from methods.adapter_clip import AdapterCLIP
from methods.Trainer_ProtoCLIP import Trainer_ProtoCLIP
from methods.clib import CLIB
from methods.continual_clip import ContinualCLIP
from methods.er_baseline import ER
from methods.ewcpp import EWCpp
from methods.finetuning import FT
from methods.lwf import LwF
from methods.maple import MaPLe
from methods.mvp import MVP
from methods.mvp_clip import CLIP_MVP
from methods.rainbow_memory import RM

# os.environ["CUDA_VISIBLE_DEVICES"]='1,2,3'
import warnings
warnings.filterwarnings("ignore")

# torch.backends.cudnn.enabled = False
methods = {
    "er": ER,
    "clib": CLIB,
    "rm": RM,
    "lwf": LwF,
    "Finetuning": FT,
    "ewc++": EWCpp,
    "mvp": MVP,
    "continual-clip": ContinualCLIP,
    "mvp-clip": CLIP_MVP,
    "maple": MaPLe,
    "adapter-clip": AdapterCLIP,
    "adapter-clip-proto_prompt": Trainer_ProtoCLIP,
    "template": Trainer_ProtoCLIP,
    "lora-clip": AdapterCLIP
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='Prompt Learning for CLIP.')
    """
    Please specify the corresponding JSON file!!!!
    """
    parser.add_argument('--config', type=str, default='./config/adapter_cifar224_config.json',
                        help='Json file of settings.')
    return parser.parse_args()

def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param

def merge_configs(args, config):
    merged_config = vars(args)
    merged_config.update(config)
    return merged_config


def main():
    # args = parse_arguments()
    # config = load_json(args.config)
    # args_dict = merge_configs(args, config)
    # args = argparse.Namespace(**args_dict)

    # Get Configurations
    args = config.base_parser()
    args.note = f'{args.method}_{args.visible_classes}_{args.peft_encoder}_{args.seed}'
    trainer = methods[args.method](**vars(args))

    trainer.run()

if __name__ == "__main__":
    main()
