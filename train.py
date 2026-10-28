import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from importlib import import_module

from sam_lora_image_encoder import LoRA_Sam
from segment_anything import sam_model_registry

from trainer import trainer_synapse, trainer_myops
from icecream import ic

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/content/MyoPS380/Processed_data', help='root dir for data')
parser.add_argument('--output', type=str, default='./outputs/sam/results')
parser.add_argument('--dataset', type=str,
                    default='MyoPS380', choices=['Synapse', 'MyoPS380'], help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_MyoPS380', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=300, help='maximum epoch number to train')
parser.add_argument('--stop_epoch', type=int,
                    default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=8, help='batch_size per gpu')
parser.add_argument('--effective_batch_size', type=int,
                    default=16, help='target effective batch size using accumulation')
parser.add_argument('--num_workers', type=int,
                    default=2, help='number of dataloader workers')
parser.add_argument('--cpu_threads', type=int,
                    default=2, help='torch cpu threads')
parser.add_argument('--weight_decay', type=float,
                    default=0.0001, help='AdamW weight decay')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.005,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=512, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--vit_name', type=str,
                    default='vit_b', help='select one vit model')
parser.add_argument('--ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth',
                    help='Pretrained checkpoint')
parser.add_argument('--lora_ckpt', type=str, default=None, help='Finetuned lora checkpoint')
parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
parser.add_argument('--warmup', action='store_true', default=True, help='If activated, warp up the learning from a lower lr to the base_lr')
parser.add_argument('--no_warmup', dest='warmup', action='store_false', help='Disable warmup')
parser.add_argument('--warmup_period', type=int, default=250,
                    help='Warp up iterations, only valid when warmup is activated')
parser.add_argument('--AdamW', action='store_true', default=True, help='If activated, use AdamW to finetune SAM model')
parser.add_argument('--SGD', dest='AdamW', action='store_false', help='Use SGD instead of AdamW')
parser.add_argument('--module', type=str, default='sam_lora_image_encoder')
parser.add_argument('--dice_param', type=float, default=0.8)
args = parser.parse_args()

if __name__ == "__main__":
    torch.set_num_threads(args.cpu_threads)
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': args.root_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
        },
        'MyoPS380': {
            'root_path': args.root_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
        }
    }
    args.is_pretrain = True
    args.exp = dataset_name + '_' + str(args.img_size)
    snapshot_path = os.path.join(args.output, f"SAMed_{dataset_name}_Seed_{args.seed}")

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](
        image_size=args.img_size,
        num_classes=args.num_classes,
        checkpoint=args.ckpt,
        pixel_mean=[0, 0, 0],
        pixel_std=[1, 1, 1]
    )

    pkg = import_module(args.module)
    net = pkg.LoRA_Sam(sam, args.rank).cuda()

    if args.lora_ckpt is not None:
        net.load_lora_parameters(args.lora_ckpt)

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    low_res = img_embedding_size * 4

    config_file = os.path.join(snapshot_path, 'config.txt')
    config_items = []
    for key, value in args.__dict__.items():
        config_items.append(f'{key}: {value}\n')

    with open(config_file, 'w') as f:
        f.writelines(config_items)

    trainer = {
        'Synapse': trainer_synapse,
        'MyoPS380': trainer_myops,
    }
    trainer[dataset_name](args, net, snapshot_path, multimask_output, low_res)
