import argparse
import os
import json

import torch
from torch.utils.tensorboard import SummaryWriter

from data_loader import get_data_loader
from models import get_models, get_params, predict
from loss import get_losses, compute_losses
from optimizer import get_optimizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--params-path', required=True, help='Path to a json parameters')
    parser.add_argument('-d', '--device', choices=['cpu', 'cuda'], default='cuda', help='Device (cpu/cuda)')
    args = parser.parse_args()

    return args


def read_params(path):
    with open(path) as f:
        return json.loads(f.read())


def main(args):
    params = read_params(os.path.join(args.params_path, 'params.json'))

    writer = SummaryWriter(os.path.join(args.params_path, 'runs'))

    train_loader = get_data_loader(params['data_type'], train=True, batch_size=params['batch_size'])
    valid_loader = get_data_loader(params['data_type'], train=False, batch_size=params['batch_size'])

    models = get_models(params['models'], args.device)
    losses = get_losses(params['losses'])

    model_params = get_params(models)
    optimizer = get_optimizer(params['optimizer'], model_params)

    running_loss = 0.0
    for epoch in range(params['num_epochs']):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(args.device), y.to(args.device)
            loss_values = compute_losses(losses, models, x, y)
            optimizer.zero_grad()

            loss_values['total_loss'].backward()
            optimizer.step()

            running_loss += loss_values['total_loss'].item()
            if step % params['summary']['train_frequency'] == 0:
                writer.add_scalar(
                    'training loss',
                    running_loss / params['summary']['save_frequency'],
                    epoch * len(train_loader) + step
                )
                log_str = 'train step: {} '.format(epoch * len(train_loader) + step)
                for k, v in loss_values.items():
                    log_str += '{}: {:.4f} '.format(k, v.item())
                print(log_str)

            if step > 0 and step % params['summary']['valid_frequency'] == 0:
                loss_values = {}
                for val_step, (x, y) in enumerate(valid_loader):
                    x, y = x.to(args.device), y.to(args.device)
                    step_loss_values = compute_losses(losses, models, x, y)
                    for k, v in step_loss_values.items():
                        loss_values[k] = (val_step * loss_values.get(k, 0.0) + v) / (val_step + 1)
                log_str = 'valid step: {} '.format(epoch * len(train_loader) + step)
                for k, v in loss_values.items():
                    log_str += '{}: {:.4f} '.format(k, v.item())
                print(log_str)

            if step % params['summary']['save_frequency'] == 0:
                save_dir = os.path.join(args.params_path, 'model_backup')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = os.path.join(save_dir, str(epoch))
                save_dict = {
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_values,
                }
                for model_name, model in models.items():
                    save_dict[model_name] = model.state_dict()
                torch.save(save_dict, save_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)
