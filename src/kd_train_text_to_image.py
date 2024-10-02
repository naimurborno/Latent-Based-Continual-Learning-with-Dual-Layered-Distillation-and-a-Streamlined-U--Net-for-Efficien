# ------------------------------------------------------------------------------------
# Copyright 2023. Nota Inc. All Rights Reserved.
# Code modified from https://github.com/huggingface/diffusers/blob/v0.15.0/examples/text_to_image/train_text_to_image.py
# ------------------------------------------------------------------------------------
#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# device
import argparse
import logging
import math
import os
import random
from pathlib import Path
from typing import Optional

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate
from diffusers.utils.import_utils import is_xformers_available

import csv
import time
import copy

# try to import wandb
try:
    import wandb
    has_wandb = True
except:
    has_wandb = False

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.15.0")

logger = get_logger(__name__, log_level="INFO")
#distillation loss function
#Run this
def distillation_loss(student_logits, teacher_logits, true_labels, T, alpha):
    # Soft target
  soft_targets = F.softmax(teacher_logits / T, dim=1)
  student_softmax = F.log_softmax(student_logits / T, dim=1)
  distillation_loss = F.kl_div(student_softmax, soft_targets, reduction='batchmean') * (T * T)

    # Hard targets
  hard_loss = F.cross_entropy(student_logits, true_labels)

    # Combined loss
  return alpha * distillation_loss + (1 - alpha) * hard_loss
flowers = {
    "1": "pink primrose",
    "2": "hard-leaved pocket orchid",
    "3": "canterbury bells",
    "4": "sweet pea",
    "5": "english marigold",
    "6": "tiger lily",
    "7": "moon orchid",
    "8": "bird of paradise",
    "9": "monkshood",
    "10": "globe thistle",
    "11": "snapdragon",
    "12": "colt's foot",
    "13": "king protea",
    "14": "spear thistle",
    "15": "yellow iris",
    "16": "globe-flower",
    "17": "purple coneflower",
    "18": "peruvian lily",
    "19": "balloon flower",
    "20": "giant white arum lily",
    "21": "fire lily",
    "22": "pincushion flower",
    "23": "fritillary",
    "24": "red ginger",
    "25": "grape hyacinth",
    "26": "corn poppy",
    "27": "prince of wales feathers",
    "28": "stemless gentian",
    "29": "artichoke",
    "30": "sweet william",
    "31": "carnation",
    "32": "garden phlox",
    "33": "love in the mist",
    "34": "mexican aster",
    "35": "alpine sea holly",
    "36": "ruby-lipped cattleya",
    "37": "cape flower",
    "38": "great masterwort",
    "39": "siam tulip",
    "40": "lenten rose",
    "41": "barbeton daisy",
    "42": "daffodil",
    "43": "sword lily",
    "44": "poinsettia",
    "45": "bolero deep blue",
    "46": "wallflower",
    "47": "marigold",
    "48": "buttercup",
    "49": "oxeye daisy",
    "50": "common dandelion",
    "51": "petunia",
    "52": "wild pansy",
    "53": "primula",
    "54": "sunflower",
    "55": "pelargonium",
    "56": "bishop of llandaff",
    "57": "gaura",
    "58": "geranium",
    "59": "orange dahlia",
    "60": "pink-yellow dahlia",
    "61": "cautleya spicata",
    "62": "japanese anemone",
    "63": "black-eyed susan",
    "64": "silverbush",
    "65": "californian poppy",
    "66": "osteospermum",
    "67": "spring crocus",
    "68": "bearded iris",
    "69": "windflower",
    "70": "tree poppy",
    "71": "gazania",
    "72": "azalea",
    "73": "water lily",
    "74": "rose",
    "75": "thorn apple",
    "76": "morning glory",
    "77": "passion flower",
    "78": "lotus lotus",
    "79": "toad lily",
    "80": "anthurium",
    "81": "frangipani",
    "82": "clematis",
    "83": "hibiscus",
    "84": "columbine",
    "85": "desert-rose",
    "86": "tree mallow",
    "87": "magnolia",
    "88": "cyclamen",
    "89": "watercress",
    "90": "canna lily",
    "91": "hippeastrum",
    "92": "bee balm",
    "93": "ball moss",
    "94": "foxglove",
    "95": "bougainvillea",
    "96": "camellia",
    "97": "mallow",
    "98": "mexican petunia",
    "99": "bromelia",
    "100": "blanket flower",
    "101": "trumpet creeper",
    "102": "blackberry lily"
}
class DreamBoothDataset(Dataset):
    def __init__(
        self,
        # instance_data_root,
        # instance_prompt,
        tokenizer,
        # class_data_root=None,
        # class_prompt=None,
        size=512,
        replay_memory=None,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        # Define the root directory and sub-directory paths
        self.root_dir = Path('/content/WSDD-Weight-Shared-Distilled-Diffusion-/oxford-102-flower-dataset/102 flower/flowers/train')
        if not self.root_dir.exists():
            raise ValueError("Root directory doesn't exist.")

        self.image_paths = []
        self.prompts = []

        # Iterate through each class folder to collect image paths and corresponding prompts
        for class_folder in self.root_dir.iterdir():
            if class_folder.is_dir():
                class_name = class_folder.name
                for image_file in class_folder.glob('*.jpg'):  # Adjust the extension as necessary
                    self.image_paths.append(image_file)
                    self.prompts.append(f"a photo of a <{flowers[class_name]}>")

        self.num_images = len(self.image_paths)
        self.replay_memory = replay_memory if replay_memory is not None else []

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self.num_images++len(self.replay_memory)

    def __getitem__(self, index):
        example = {}

        if index < self.num_images:
            # Load data from the original dataset
            image_path = self.image_paths[index]
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            prompt = self.prompts[index]
        else:
            # Load data from the replay memory
            replay_index = index - self.num_images
            image, prompt = self.replay_memory[replay_index]

        example['instance_images'] = self.image_transforms(image)
        example['instance_prompt_ids'] = self.tokenizer(
            prompt,
            padding='do_not_pad',
            truncation=True,
            max_length=self.tokenizer.model_max_length
        ).input_ids

        return example
    def update_replay_memory(self, new_data):
        """Update the replay memory with new data."""
        self.replay_memory.extend(new_data)

class PromptDataset(Dataset):
    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example

def get_activation(mem, name):
    def get_output_hook(module, input, output):
        mem[name] = output

    return get_output_hook

def add_hook(net, mem, mapping_layers):
    for n, m in net.named_modules():
        if n in mapping_layers:
            m.register_forward_hook(get_activation(mem, n))

def copy_weight_from_teacher(unet_stu, unet_tea, student_type):

    connect_info = {} # connect_info['TO-student'] = 'FROM-teacher'
    if student_type in ["bk_base", "bk_small"]:
        connect_info['up_blocks.0.resnets.1.'] = 'up_blocks.0.resnets.2.'
        connect_info['up_blocks.1.resnets.1.'] = 'up_blocks.1.resnets.2.'
        connect_info['up_blocks.1.attentions.1.'] = 'up_blocks.1.attentions.2.'
        connect_info['up_blocks.2.resnets.1.'] = 'up_blocks.2.resnets.2.'
        connect_info['up_blocks.2.attentions.1.'] = 'up_blocks.2.attentions.2.'
        connect_info['up_blocks.3.resnets.1.'] = 'up_blocks.3.resnets.2.'
        connect_info['up_blocks.3.attentions.1.'] = 'up_blocks.3.attentions.2.'
    elif student_type in ["bk_tiny"]:
        connect_info['up_blocks.0.resnets.0.'] = 'up_blocks.1.resnets.0.'
        connect_info['up_blocks.0.attentions.0.'] = 'up_blocks.1.attentions.0.'
        connect_info['up_blocks.0.resnets.1.'] = 'up_blocks.1.resnets.2.'
        connect_info['up_blocks.0.attentions.1.'] = 'up_blocks.1.attentions.2.'
        connect_info['up_blocks.0.upsamplers.'] = 'up_blocks.1.upsamplers.'
        connect_info['up_blocks.1.resnets.0.'] = 'up_blocks.2.resnets.0.'
        connect_info['up_blocks.1.attentions.0.'] = 'up_blocks.2.attentions.0.'
        connect_info['up_blocks.1.resnets.1.'] = 'up_blocks.2.resnets.2.'
        connect_info['up_blocks.1.attentions.1.'] = 'up_blocks.2.attentions.2.'
        connect_info['up_blocks.1.upsamplers.'] = 'up_blocks.2.upsamplers.'
        connect_info['up_blocks.2.resnets.0.'] = 'up_blocks.3.resnets.0.'
        connect_info['up_blocks.2.attentions.0.'] = 'up_blocks.3.attentions.0.'
        connect_info['up_blocks.2.resnets.1.'] = 'up_blocks.3.resnets.2.'
        connect_info['up_blocks.2.attentions.1.'] = 'up_blocks.3.attentions.2.'       
    else:
        raise NotImplementedError


    for k in unet_stu.state_dict().keys():
        flag = 0
        k_orig = k
        for prefix_key in connect_info.keys():
            if k.startswith(prefix_key):
                flag = 1
                k_orig = k_orig.replace(prefix_key, connect_info[prefix_key])            
                break

        if flag == 1:
            print(f"** forced COPY {k_orig} -> {k}")
        else:
            print(f"normal COPY {k_orig} -> {k}")
        unet_stu.state_dict()[k].copy_(unet_tea.state_dict()[k_orig])

    return unet_stu

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_false", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    parser.add_argument("--unet_config_path", type=str, default="./src/unet_config")     
    parser.add_argument("--unet_config_name", type=str, default="bk_small", choices=["bk_base", "bk_small", "bk_tiny"])   
    parser.add_argument("--lambda_sd", type=float, default=1.0, help="weighting for the denoising task loss")  
    parser.add_argument("--lambda_kd_output", type=float, default=1.0, help="weighting for output KD loss")  
    parser.add_argument("--lambda_kd_feat", type=float, default=1.0, help="weighting for feature KD loss")  
    parser.add_argument("--valid_prompt", type=str, default="a golden vase with different flowers")
    parser.add_argument("--valid_steps", type=int, default=500)
    parser.add_argument("--num_valid_images", type=int, default=2)
    parser.add_argument("--use_copy_weight_from_teacher", action="store_true", help="Whether to initialize unet student with teacher's weights",)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def main():
    args = parse_args()

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        # logging_dir=logging_dir,
        project_config=accelerator_project_config,
    )
    # accelerator.device='cuda'

    # Add custom csv logger and validation image folder
    val_img_dir = os.path.join(args.output_dir, 'val_img')
    os.makedirs(val_img_dir, exist_ok=True)


    csv_log_path = os.path.join(args.output_dir, 'log_loss.csv')
    print(csv_log_path)
    if not os.path.exists(csv_log_path):
        with open(csv_log_path, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(['epoch', 'step', 'global_step',
                                'loss_total', 'loss_sd', 'loss_kd_output', 'loss_kd_feat',
                                'lr', 'lamb_sd', 'lamb_kd_output', 'lamb_kd_feat'])

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process and (args.output_dir is not None):
        os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)

    # Define teacher and student
    unet_teacher = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
    )

    config_student = UNet2DConditionModel.load_config(args.unet_config_path, subfolder=args.unet_config_name)
    unet = UNet2DConditionModel.from_config(config_student, revision=args.non_ema_revision)
    # unet = UNet2DConditionModel.from_config('/content/WSDD-Weight-Shared-Distilled-Diffusion-/src/unet_config/bk_tiny/bk_small')

    # Copy weights from teacher to student
    if args.use_copy_weight_from_teacher:
        copy_weight_from_teacher(unet, unet_teacher, args.unet_config_name)
   

    # Freeze student's vae and text_encoder and teacher's unet
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet_teacher.requires_grad_(False)
    

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_config(config_student, revision=args.revision)
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets. As the amount of data grows, the time taken by load_dataset also increases.
    print("*** load dataset: start")
    t0 = time.time()
    train_dataset = DreamBoothDataset(
        # instance_data_root=args.instance_data_dir,
        # instance_prompt=args.instance_prompt,
        # class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        # class_prompt=args.class_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
    )
    print(f"*** load dataset: end --- {time.time()-t0} sec")

    # Preprocessing the datasets.
    # column_names = dataset.column_names
    # image_column = column_names[0]
    # caption_column = column_names[1]

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    # def tokenize_captions(examples, is_train=True):
    #     captions = []
    #     for caption in examples[caption_column]:
    #         if isinstance(caption, str):
    #             captions.append(caption)
    #         elif isinstance(caption, (list, np.ndarray)):
    #             # take a random caption if there are multiple
    #             captions.append(random.choice(caption) if is_train else caption[0])
    #         else:
    #             raise ValueError(
    #                 f"Caption column `{caption_column}` should contain either strings or lists of strings."
    #             )
    #     inputs = tokenizer(
    #         captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    #     )
    #     return inputs.input_ids

    # Preprocessing the datasets.
    # train_transforms = transforms.Compose(
    #     [
    #         transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
    #         transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
    #         transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.5], [0.5]),
    #     ]
    # )

    # def preprocess_train(examples):
    #     images = [image.convert("RGB") for image in examples[image_column]]
    #     examples["pixel_values"] = [train_transforms(image) for image in images]
    #     examples["input_ids"] = tokenize_captions(examples)
    #     return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset = dataset.shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        # train_dataset = dataset.with_transform(preprocess_train)

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # concat class and instance examples for prior preservation
        # if args.with_prior_preservation:
        #     input_ids += [example["class_prompt_ids"] for example in examples]
        #     pixel_values += [example["class_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            return_tensors="pt",
            max_length=tokenizer.model_max_length
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        return batch

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn
    )
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    if args.use_ema:
        ema_unet.to(accelerator.device)
    
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move student's text_encode and vae and teacher's unet to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet_teacher.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    # if accelerator.is_main_process:
    #     accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # Add hook for feature KD
    acts_tea = {}
    acts_stu = {}
    if args.unet_config_name in ["bk_base", "bk_small"]:
        mapping_layers = ['up_blocks.0', 'up_blocks.1', 'up_blocks.2', 'up_blocks.3',
                        'down_blocks.0', 'down_blocks.1', 'down_blocks.2', 'down_blocks.3']    
        mapping_layers_tea = copy.deepcopy(mapping_layers)
        mapping_layers_stu = copy.deepcopy(mapping_layers)

    elif args.unet_config_name in ["bk_tiny"]:
        mapping_layers_tea = ['down_blocks.0', 'down_blocks.1', 'down_blocks.2.attentions.1.proj_out',
                                'up_blocks.1', 'up_blocks.2', 'up_blocks.3']    
        mapping_layers_stu = ['down_blocks.0', 'down_blocks.1', 'down_blocks.2.attentions.0.proj_out',
                                'up_blocks.0', 'up_blocks.1', 'up_blocks.2']  

    if torch.cuda.device_count() > 1:
        print(f"use multi-gpu: # gpus {torch.cuda.device_count()}")
        # revise the hooked feature names for student (to consider ddp wrapper)
        for i, m_stu in enumerate(mapping_layers_stu):
            mapping_layers_stu[i] = 'module.'+m_stu

    add_hook(unet_teacher, acts_tea, mapping_layers_tea)
    add_hook(unet, acts_stu, mapping_layers_stu)

    # get wandb_tracker (if it exists)
    wandb_tracker = accelerator.get_tracker("wandb")
    T = 2.0  # Temperature
    alpha = 1  # Weight for distillation loss

    for epoch in range(first_epoch, args.num_train_epochs):

        unet.train()

        train_loss = 0.0
        train_loss_sd = 0.0
        train_loss_kd_output = 0.0
        train_loss_kd_feat = 0.0

        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss_sd = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Predict output-KD loss
                model_pred_teacher = unet_teacher(noisy_latents, timesteps, encoder_hidden_states).sample
                loss_kd_output = F.mse_loss(model_pred.float(), model_pred_teacher.float(), reduction="mean")
                # Predict feature-KD loss
                losses_kd_feat = []
                for (m_tea, m_stu) in zip(mapping_layers_tea, mapping_layers_stu):
                    a_tea = acts_tea[m_tea]
                    a_stu = acts_stu[m_stu]

                    if type(a_tea) is tuple: a_tea = a_tea[0]                        
                    if type(a_stu) is tuple: a_stu = a_stu[0]

                    tmp = F.mse_loss(a_stu.float(), a_tea.detach().float(), reduction="mean")
                    losses_kd_feat.append(tmp)
                loss_kd_feat = sum(losses_kd_feat)

                # Compute the final loss
                loss_stu=distillation_loss(model_pred.float(),model_pred_teacher.float(),target.float(),T,alpha)
                loss = args.lambda_sd * loss_sd + args.lambda_kd_output * loss_kd_output + args.lambda_kd_feat * loss_kd_feat+loss_stu
                

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                avg_loss_sd = accelerator.gather(loss_sd.repeat(args.train_batch_size)).mean()
                train_loss_sd += avg_loss_sd.item() / args.gradient_accumulation_steps

                avg_loss_kd_output = accelerator.gather(loss_kd_output.repeat(args.train_batch_size)).mean()
                train_loss_kd_output += avg_loss_kd_output.item() / args.gradient_accumulation_steps

                avg_loss_kd_feat = accelerator.gather(loss_kd_feat.repeat(args.train_batch_size)).mean()
                train_loss_kd_feat += avg_loss_kd_feat.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            new_replay_data = [(batch["pixel_values"][i], batch["input_ids"][i]) for i in range(len(batch["pixel_values"]))]
            train_dataset.update_replay_memory(new_replay_data)  # Update the replay memory with current batch data

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log(
                    {
                        "train_loss": train_loss, 
                        "train_loss_sd": train_loss_sd,
                        "train_loss_kd_output": train_loss_kd_output,
                        "train_loss_kd_feat": train_loss_kd_feat,
                        "lr": lr_scheduler.get_last_lr()[0]
                    }, 
                    step=global_step
                )

                if accelerator.is_main_process:
                    with open(csv_log_path, 'a') as logfile:
                        logwriter = csv.writer(logfile, delimiter=',')
                        logwriter.writerow([epoch, step, global_step,
                                            train_loss, train_loss_sd, train_loss_kd_output, train_loss_kd_feat,
                                            lr_scheduler.get_last_lr()[0],
                                            args.lambda_sd, args.lambda_kd_output, args.lambda_kd_feat])

                train_loss = 0.0
                train_loss_sd = 0.0
                train_loss_kd_output = 0.0
                train_loss_kd_feat = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(),
                    "sd_loss": loss_sd.detach().item(),
                    "kd_output_loss": loss_kd_output.detach().item(),
                    "kd_feat_loss": loss_kd_feat.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            # save validation images
            # if (args.valid_prompt is not None) and (step % args.valid_steps == 0) and accelerator.is_main_process:
            #     logger.info(
            #         f"Running validation... \n Generating {args.num_valid_images} images with prompt:"
            #         f" {args.valid_prompt}."
            #     )
                # create pipeline
                # pipeline = StableDiffusionPipeline.from_pretrained(
                #     args.pretrained_model_name_or_path,
                #     safety_checker=None,
                #     revision=args.revision,
                # )
                # pipeline = pipeline.to(accelerator.device)
                # pipeline.set_progress_bar_config(disable=True)
                # generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

                # if not os.path.exists(os.path.join(val_img_dir, "teacher_0.png")):
                #     for kk in range(args.num_valid_images):
                #         image = pipeline(args.valid_prompt, num_inference_steps=25, generator=generator).images[0]
                #         tmp_name = os.path.join(val_img_dir, f"teacher_{kk}.png")
                #         image.save(tmp_name)

                # set `keep_fp32_wrapper` to True because we do not want to remove
                # mixed precision hooks while we are still training
                # pipeline.unet = accelerator.unwrap_model(unet, keep_fp32_wrapper=True).to(accelerator.device)
              
                # # for kk in range(args.num_valid_images):
                # #     image = pipeline(args.valid_prompt, num_inference_steps=25, generator=generator).images[0]
                # #     tmp_name = os.path.join(val_img_dir, f"gstep{global_step}_epoch{epoch}_step{step}_{kk}.png")
                # #     print(tmp_name)
                # #     image.save(tmp_name)

                # del pipeline
                #torch.cuda.empty_cache()

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            revision=args.revision,
        )
        pipeline.save_pretrained(args.output_dir)

    accelerator.end_training()

if __name__ == "__main__":
    main()
