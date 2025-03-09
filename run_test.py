import argparse
import datetime
import random
import time
from pathlib import Path

import torch
import torchvision.transforms as standard_transforms
import numpy as np

from PIL import Image
import cv2
from crowd_datasets import build_dataset
from engine import *
from models import build_model
import os
import warnings
warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for DSGCNet evaluation', add_help=False)
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")
    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")
    parser.add_argument('--weight_path', default='./weights/SHTechA.pth',
                        help='path where the trained weights saved')
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')
    parser.add_argument('--root_dir', default='./sha_a/test', type=str,
                        help="the directory containing scene_1, scene_2, ... subfolders")
    parser.add_argument('--output_dir', default='./pred_result', type=str,
                        help='path where to save the predicted images')
    parser.add_argument('--threshold', default=0.5, type=float,
                        help='score threshold for deciding whether a predicted point is valid')
    return parser


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(args)
    model.to(device)
    if args.weight_path is not None and os.path.exists(args.weight_path):
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        print(f"Loaded weights from {args.weight_path}")
    else:
        print(f"[Warning] Weight file not found: {args.weight_path}. Randomly initializing model.")

    model.eval()
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    scene_folders = [
        f for f in os.listdir(args.root_dir) 
        if os.path.isdir(os.path.join(args.root_dir, f)) and f.startswith('scene_')
    ]
    scene_folders.sort(key=lambda x: int(x.split('_')[-1]))
    for scene_name in scene_folders:
        scene_path = os.path.join(args.root_dir, scene_name)
        jpg_file = None
        for file_name in os.listdir(scene_path):
            if file_name.endswith('.jpg'):
                jpg_file = file_name
                break
        if not jpg_file:
            print(f"[Warning] No .jpg files found in {scene_path}, skipping.")
            continue
        img_path = os.path.join(scene_path, jpg_file)
        img_raw = Image.open(img_path).convert('RGB')
        width, height = img_raw.size
        new_width = width // 128 * 128
        new_height = height // 128 * 128
        img_raw = img_raw.resize((new_width, new_height), cv2.INTER_CUBIC)
        img = transform(img_raw)
        samples = torch.Tensor(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(samples)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], dim=-1)[:, :, 1][0] 
        outputs_points = outputs['pred_points'][0]
        valid_mask = outputs_scores > args.threshold
        points = outputs_points[valid_mask].detach().cpu().numpy().tolist()
        predict_cnt = int(valid_mask.sum().item())
        img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
        size = 3  
        for p in points:
            cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
        text = str(predict_cnt)
        font_face = cv2.FONT_HERSHEY_TRIPLEX
        font_scale = 2.0
        thickness = 3
        color = (255, 255, 255) 
        H, W, _ = img_to_draw.shape
        (text_w, text_h), baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
        x_pos = W - text_w - 10
        y_pos = H - 10
        cv2.putText(img_to_draw, text, (x_pos, y_pos), font_face, font_scale, color, thickness)
        base_name = os.path.splitext(jpg_file)[0]  
        out_file_name = f"{base_name}_pred.jpg"
        out_file_path = os.path.join(args.output_dir, out_file_name)
        cv2.imwrite(out_file_path, img_to_draw)
        print(f"[INFO] Processed {img_path} -> Predicted count {predict_cnt}, result saved to {out_file_path}")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser('DSGCNet batch prediction script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
