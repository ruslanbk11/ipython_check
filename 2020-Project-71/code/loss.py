import torch
import torch.nn as nn


CRITERIONS = {
    'mse': nn.MSELoss()
}

def get_losses(params):
    losses = {}
    for k, v in params.items():
        assert v['type'] in CRITERIONS, "loss type {} is unknown".format(CRITERIONS)
        losses[k] = {
            'scale': v['scale'],
            'type': CRITERIONS[v['type']]
        }

    return losses


def compute_losses(losses, models, x, y, training=False):
    for model in models.values():
        model.train() if training else model.eval()
    result = {}
    encoded_input = models['encoder_input'](x)
    if 'reconstruct_input' in losses:
        decoded_input = models['decoder_input'](encoded_input)
        loss_dict = losses['reconstruct_input']
        result['reconstruct_input'] = loss_dict['scale'] * loss_dict['type'](x, decoded_input)
    if 'reconstruct_target' in losses:
        encoded_target = models['encoder_target'](y)
        decoded_target = models['decoder_target'](encoded_target)
        loss_dict = losses['reconstruct_target']
        result['reconstruct_target'] = loss_dict['scale'] * loss_dict['type'](y, decoded_target)
    if 'coherence' in losses:
        encoded_target = models['encoder_target'](y)
        loss_dict = losses['coherence']
        result['coherence'] = loss_dict['scale'] * loss_dict['type'](encoded_input, encoded_target)
    if 'alignment' in losses:
        encoded_target = models['encoder_target'](y)
        aligned_encoded = models['alignment'](encoded_input)
        loss_dict = losses['alignment']
        result['alignment'] = loss_dict['scale'] * loss_dict['type'](encoded_target, aligned_encoded)
    if 'prediction' in losses:
        aligned_encoded = models['alignment'](encoded_input)
        decoded_aligned = models['decoder_target'](aligned_encoded)
        loss_dict = losses['prediction']
        result['prediction'] = loss_dict['scale'] * loss_dict['type'](y, decoded_aligned)

    result['total_loss'] = sum(result.values())

    return result
