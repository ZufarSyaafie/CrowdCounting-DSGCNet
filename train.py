import argparse
import datetime
import random
import time
import torch
from torch.utils.data import DataLoader
from torch import nn
from crowd_datasets import build_dataset
from engine import *
from models import build_model
import os
from tensorboardX import SummaryWriter
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for training P2PNet', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=2500, type=int)
    parser.add_argument('--lr_drop', default=800, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="Name of the convolutional backbone to use")

    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")

    parser.add_argument('--set_cost_point', default=0.05, type=float,
                        help="L1 point coefficient in the matching cost")

    parser.add_argument('--point_loss_coef', default=0.0002, type=float)

    parser.add_argument('--eos_coef', default=0.5, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    parser.add_argument('--dataset_file', default='SHHA')
    parser.add_argument('--data_root', default='',
                        help='path where the dataset is')
    
    parser.add_argument('--output_dir', default='./log',
                        help='path where to save, empty for no saving')
    parser.add_argument('--checkpoints_dir', default='./ckpt',
                        help='path where to save checkpoints, empty for no saving')
    parser.add_argument('--tensorboard_dir', default='./runs',
                        help='path where to save, empty for no saving')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_freq', default=1, type=int,
                        help='frequency of evaluation, default setting is evaluating in every 5 epoch')
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for training')

    return parser

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_log_name = os.path.join(args.output_dir, f'run_log_{timestamp}.txt')
    tensorboard_dir = os.path.join(args.tensorboard_dir, f'{timestamp}')
    with open(run_log_name, "w") as log_file:
        log_file.write('Eval Log %s\n' % time.strftime("%c"))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)
    with open(run_log_name, "a") as log_file:
        log_file.write("{}".format(args))
    device = torch.device('cuda')
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model, criterion = build_model(args, training=True)
    model.to(device)
    criterion.to(device)
    density_criterion = nn.MSELoss(size_average=False).to(device)
    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.Adam(param_dicts, lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    loading_data = build_dataset(args=args)
    train_set, val_set = loading_data(args.data_root)
    sampler_train = torch.utils.data.RandomSampler(train_set)
    sampler_val = torch.utils.data.SequentialSampler(val_set)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(train_set, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn_crowd_train, num_workers=args.num_workers)

    data_loader_val = DataLoader(val_set, 1, sampler=sampler_val,
                                    drop_last=False, collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

    print("Start training")
    start_time = time.time()
    mae = []
    mse = []
    density_mae = []
    density_mse = []
    writer = SummaryWriter(tensorboard_dir)
    
    step = 0
    for epoch in range(args.start_epoch, args.epochs):
        t1 = time.time()
        stat = train_one_epoch(
            model, criterion, data_loader_train, optimizer, density_criterion, device, epoch,
            args.clip_max_norm)

        if writer is not None:
            with open(run_log_name, "a") as log_file:
                log_file.write("loss/loss_sum@{}: {}".format(epoch, stat['loss_sum']))
                log_file.write("loss/losses@{}: {}".format(epoch, stat['losses']))
                log_file.write("loss/den_loss@{}: {}".format(epoch, stat['den_loss']))
                log_file.write("loss/loss_ce@{}: {}".format(epoch, stat['loss_ce']))
                
            writer.add_scalar('loss/loss_sum', stat['loss_sum'], epoch)    
            writer.add_scalar('loss/losses', stat['losses'], epoch)
            writer.add_scalar('loss/den_loss', stat['den_loss'], epoch)
            writer.add_scalar('loss/loss_ce', stat['loss_ce'], epoch)

        t2 = time.time()
        print('[ep %d][lr %.7f][%.2fs]' % \
              (epoch, optimizer.param_groups[0]['lr'], t2 - t1))
        with open(run_log_name, "a") as log_file:
            log_file.write('[ep %d][lr %.7f][%.2fs]' % (epoch, optimizer.param_groups[0]['lr'], t2 - t1))
        lr_scheduler.step()
        checkpoint_latest_path = os.path.join(args.checkpoints_dir, f'latest_{timestamp}.pth')
        torch.save({
            'model': model_without_ddp.state_dict(),
        }, checkpoint_latest_path)
        if epoch % args.eval_freq == 0 and epoch != 0:
            t1 = time.time()
            result = evaluate_crowd_no_overlap(model, data_loader_val, device)
            t2 = time.time()

            mae.append(result[0])
            mse.append(result[1])
            density_mae.append(result[2])
            density_mse.append(result[3])
            print('=======================================test=======================================')
            print("mae:", result[0], "mse:", result[1], "time:", t2 - t1, "best mae:", np.min(mae), )
            with open(run_log_name, "a") as log_file:
                log_file.write("mae:{}, mse:{}, time:{}, best mae:{}".format(result[0], 
                                result[1], t2 - t1, np.min(mae)))
            print('==================================================================================')
            print("density_mae:", result[2], "density_mse:", result[3], "time:", t2 - t1, "best density_mae:", np.min(density_mae), )
            with open(run_log_name, "a") as log_file:
                log_file.write("density_mae:{}, density_mse:{}, time:{}, best density_mae:{}".format(result[2], 
                                result[3], t2 - t1, np.min(density_mae)))
            print('=======================================test=======================================')
            if writer is not None:
                with open(run_log_name, "a") as log_file:
                    log_file.write("metric/mae@{}: {}".format(step, result[0]))
                    log_file.write("metric/mse@{}: {}".format(step, result[1]))
                    log_file.write("metric/density_mae@{}: {}".format(step, result[2]))
                    log_file.write("metric/density_mse@{}: {}".format(step, result[3]))
                writer.add_scalar('metric/mae', result[0], step)
                writer.add_scalar('metric/mse', result[1], step)
                writer.add_scalar('metric/density_mae', result[2], step)
                writer.add_scalar('metric/density_mse', result[3], step)
                step += 1

            if abs(np.min(mae) - result[0]) < 0.01:
                checkpoint_best_path = os.path.join(args.checkpoints_dir, f'best_mae_{timestamp}.pth')
                torch.save({
                    'model': model_without_ddp.state_dict(),
                }, checkpoint_best_path)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)