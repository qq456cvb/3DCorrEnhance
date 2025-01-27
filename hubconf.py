import torch
from finetune import FinetuneDINO


def dinov2_base():
    model = FinetuneDINO.load_from_checkpoint('https://huggingface.co/qq456cvb/3DCorrEnhance/resolve/main/dinov2_base.ckpt', r=4, backbone_size='base').eval()
    return model