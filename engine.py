import math
import sys
from typing import Iterable
import torch
import util.misc as utils
import numpy as np

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,Dens_Loss,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    for samples, targets, gt_dmap in data_loader:
        
        samples = samples.to(device)
        gt_dmap= torch.stack(gt_dmap)
        gt_dmap = gt_dmap.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        et_dmap = outputs['density_out']
        density_loss = Dens_Loss(et_dmap,gt_dmap) / gt_dmap.shape[0] * 0.01
        loss_sum = losses + density_loss
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_sum.item()))
            print(loss_dict_reduced)
            sys.exit(1)
        optimizer.zero_grad()
        loss_sum.backward() 
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        den_loss_value = density_loss.item()
        loss_sum_value = loss_sum.item()
        metric_logger.update(loss_sum=loss_sum_value, losses=loss_value, den_loss=den_loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate_crowd_no_overlap(model, data_loader, device, vis_dir=None):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    maes = []
    mses = []
    density_maes = []
    density_mses = []
    for samples, targets in data_loader:
        samples = samples.to(device)
        outputs = model(samples)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
        outputs_points = outputs['pred_points'][0]

        gt_cnt = targets[0]['point'].shape[0]
        threshold = 0.5

        points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
        predict_cnt = int((outputs_scores > threshold).sum())
        if vis_dir is not None: 
            vis(samples, targets, [points], vis_dir)
        mae = abs(predict_cnt - gt_cnt)
        mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)
        maes.append(float(mae))
        mses.append(float(mse))
        et_dmap = outputs['density_out']
        et_dmap_sum = int(torch.sum(et_dmap))
        density_mae = abs(et_dmap_sum - gt_cnt)
        density_mse = (et_dmap_sum - gt_cnt) * (et_dmap_sum - gt_cnt)
        density_maes.append(float(density_mae))
        density_mses.append(float(density_mse))
    mae = np.mean(maes)
    mse = np.sqrt(np.mean(mses))
    density_mae = np.mean(density_maes)
    density_mse = np.sqrt(np.mean(density_mses))
    return mae, mse, density_mae, density_mse