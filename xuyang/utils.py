# from config import *
import argparse

import json
import os
import pprint as pp
import random
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import optim as optim

def setup_train(args):
    args = get_device(args)

    export_root = create_experiment_export_folder(args)
    export_experiments_config_as_json(args, export_root)

    pp.pprint({k: v for k, v in args.__dict__.items() if v is not None}, width=1)
    return export_root, args


def reset_args(args):
    # reset llm model
    if args.llm_name == 'llama-7b':
        args.model_name_or_path = "/ssd/public_datasets/llama/llama_to_hf/llama-7b"
        args.tokenizer_name_or_path = "/ssd/public_datasets/llama/llama_to_hf/llama-7b"
    elif args.llm_name == 'vicuna-7b':
        args.model_name_or_path = "/ssd/public_datasets/llama/vicuna-7b"
        args.tokenizer_name_or_path = "/ssd/public_datasets/llama/vicuna-7b"
    elif args.llm_name == 'llama-v2-13b':
        args.model_name_or_path = "/ssd/public_datasets/llama/llama_v2/Llama-2-13b-chat-hf"
        args.tokenizer_name_or_path = "/ssd/public_datasets/llama/llama_v2/Llama-2-13b-chat-hf"
    else:
        args.model_name_or_path = "gpt2"
        args.tokenizer_name_or_path = "gpt2"
    # reset path
    args.peft_model_id = f"{args.llm_name}_{args.peft_type}_{args.task_type}"
    args.checkpoint_name = f"{args.dataset_name}_{args.model_name_or_path}_{args.peft_type}_{args.task_type}_v1.pt".replace("/", "_")
    if args.fixed_prompt:
        args.experiment_description = 'v1_pointwise_without_prompt_example_{}_{}_{}_{}_{}_{}_fixed_prompt_contractive_hard_10_val_loss'.format(args.dataset_name, args.llm_name, args.peft_model_id, args.num_virtual_tokens, args.few_shot_num, args.prompt_num)
    else:
        args.experiment_description = 'v1_pointwise_{}_{}_{}_{}_{}_{}'.format(args.dataset_name, args.llm_name, args.peft_model_id, args.num_virtual_tokens, args.few_shot_num, args.prompt_num)


    return args


def create_experiment_export_folder(args):
    experiment_dir = args.experiment_dir
    experiment_description = args.experiment_description
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    experiment_path = get_name_of_experiment_path(experiment_dir, experiment_description)
    os.mkdir(experiment_path)
    print('Folder created: ' + os.path.abspath(experiment_path))
    return experiment_path


def get_name_of_experiment_path(experiment_dir, experiment_description):
    experiment_path = os.path.join(experiment_dir, (experiment_description + "_" + str(date.today())))
    idx = _get_experiment_index(experiment_path)
    experiment_path = experiment_path + "_" + str(idx)
    return experiment_path


def _get_experiment_index(experiment_path):
    idx = 0
    while os.path.exists(experiment_path + "_" + str(idx)):
        idx += 1
    return idx


def load_weights(model, path):
    pass


def save_test_result(export_root, result):
    filepath = Path(export_root).joinpath('test_result.txt')
    with filepath.open('w') as f:
        json.dump(result, f, indent=2)


def export_experiments_config_as_json(args, experiment_path):
    with open(os.path.join(experiment_path, 'config.json'), 'w') as outfile:
        json.dump(args.__dict__, outfile, indent=2)


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


# def set_up_gpu(args):
#     os.environ['CUDA_VISIBLE_DEVICES'] = args['device_idx']
#     # args['num_gpu'] = len(args['device_idx'].split(","))
#     get_device(args)

def get_device(args):
    if torch.cuda.is_available():
        device = "cuda:{}".format(args.device_idx)
    else:
        device = "cpu"
    args.device = device
    print("use device", device)
    return args

# def load_pretrained_weights(model, path):
#     chk_dict = torch.load(os.path.abspath(path))
#     model_state_dict = chk_dict[STATE_DICT_KEY] if STATE_DICT_KEY in chk_dict else chk_dict['state_dict']
#     model.load_state_dict(model_state_dict)


# def setup_to_resume(args, model, optimizer):
#     chk_dict = torch.load(os.path.join(os.path.abspath(args.resume_training), 'models/checkpoint-recent.pth'))
#     model.load_state_dict(chk_dict[STATE_DICT_KEY])
#     optimizer.load_state_dict(chk_dict[OPTIMIZER_STATE_DICT_KEY])


# def create_optimizer(model, args):
#     if args.optimizer == 'Adam':
#         return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

#     return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count