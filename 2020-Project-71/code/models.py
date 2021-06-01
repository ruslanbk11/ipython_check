import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


ACTIVATIONS = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
    'none': None
}


class Base(nn.Module):
    def __init__(self, params):
        super(Base, self).__init__()
        self._model = self._create(params)
        self.num_params = sum(p.numel() for p in self.parameters())

    def _create(self, params):
        self._layers = []
        for l in params['arch']:
            if l['type'] == 'conv':
                n_in = l['n_in']
                n_out = l['n_out']
                activation_fn = l.get('activation_fn', 'relu')
                self._add_conv(n_in, n_out, activation_fn)
            elif l['type'] == 'pool':
                self._add_pool()
            elif l['type'] == 'upsample':
                self._add_upsample()
            else:
                raise ValueError("layer type '{}' is unknown".format(l['type']))
        return nn.Sequential(*self._layers)

    def _add_conv(self, n_in, n_out, activation_fn):
        self._layers.append(nn.Conv2d(n_in, n_out, 3, stride=1, padding=1))
        if activation_fn is not None:
            self._layers.append(ACTIVATIONS[activation_fn])

    def _add_pool(self):
        self._layers.append(nn.MaxPool2d(2))

    def _add_upsample(self):
        self._layers.append(nn.Upsample(scale_factor=2))

    def forward(self, x):
        return self._model(x)


def get_models(params, device):
    models = {}
    for model_name, model_params in params.items():
        model = Base(model_params).to(device)
        models[model_name] = model

        print('======================')
        print(model_name)
        print('# params:', model.num_params)

    print('======================')
    return models


def get_params(models):
    params = []
    for model in models.values():
        params.extend(model.parameters())
    return params


def predict(models, x):
    x = models['encoder_input'](x)
    x = models['alignment'](x)
    x = models['decoder_target'](x)

    return x


class DumbNet(nn.Module):
    def __init__(self, input_layer_size=28*14, hidden_layer=400,  criterion=nn.MSELoss()):
        super(DumbNet, self).__init__()
        self.fc1 = nn.Linear(input_layer_size, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, hidden_layer)
        self.fc3 = nn.Linear(hidden_layer, hidden_layer)
        self.fc4 = nn.Linear(hidden_layer, hidden_layer)
        self.fc5 = nn.Linear(hidden_layer, hidden_layer)
        self.fc6 = nn.Linear(hidden_layer, input_layer_size)
        self.criterion = criterion
        self.input_layer_size = input_layer_size
        self.number_of_weight_coefficients = 2*input_layer_size*hidden_layer + hidden_layer*hidden_layer*4

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        x = torch.tanh(self.fc4(x))
        x = self.fc5(x)
        x = torch.tanh(self.fc6(x))
        return x

class DumbNet2(nn.Module):
    def __init__(self, input_layer_size=28*14, criterion=nn.MSELoss()):
        super(DumbNet2, self).__init__()
        self.fc1 = nn.Linear(input_layer_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 100)
        self.fc5 = nn.Linear(100, 100)
        self.fc6 = nn.Linear(100, 100)
        self.fc7 = nn.Linear(100, 100)
        self.fc8 = nn.Linear(100, 100)
        self.fc9 = nn.Linear(100, 32)
        self.fc10 = nn.Linear(32, 64)
        self.fc11 = nn.Linear(64, 256)
        self.fc12 = nn.Linear(256, input_layer_size)

        self.criterion = criterion
        self.input_layer_size = input_layer_size
        self.number_of_weight_coefficients = 2*(input_layer_size*256 + 256*64 + 64*32 + 32*100) + 100*100*4


    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        x = torch.tanh(self.fc6(x))
        x = torch.tanh(self.fc7(x))
        x = torch.tanh(self.fc8(x))
        x = torch.tanh(self.fc9(x))
        x = torch.tanh(self.fc10(x))
        x = torch.tanh(self.fc11(x))
        x = torch.tanh(self.fc12(x))
        return x


class EncNet(nn.Module):
    def __init__(self, input_layer_size=32, criterion=nn.MSELoss()):
        super(EncNet, self).__init__()
        self.fc1 = nn.Linear(input_layer_size, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 100)
        self.fc6 = nn.Linear(100, input_layer_size)

        self.criterion = criterion
        self.input_layer_size = input_layer_size

        self.number_of_weight_coefficients = 2*(input_layer_size*100) + 100*100*4
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))    
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return torch.tanh(x)

    
class EncNet2(nn.Module):
    def __init__(self, input_layer_size=32, criterion=nn.MSELoss()):
        super(EncNet2, self).__init__()
        self.fc1 = nn.Linear(input_layer_size, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, 40)
        self.fc4 = nn.Linear(40, 40)
        self.fc5 = nn.Linear(40, input_layer_size)

        self.criterion = criterion
        self.input_layer_size = input_layer_size

        self.number_of_weight_coefficients = 2*(input_layer_size*40) + 40*40*2
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))    
        x = F.relu(self.fc3(x))    
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        
        return torch.tanh(x)
    

class LinearNet(nn.Module):
    def __init__(self, input_layer_size=32, criterion=nn.MSELoss()):
        super(LinearNet, self).__init__()
        self.fc1 = nn.Linear(input_layer_size, input_layer_size)
        self.criterion = criterion
        self.input_layer_size = input_layer_size
        self.number_of_weight_coefficients = 2*(input_layer_size*input_layer_size)

    def forward(self, x):
        x = self.fc1(x)
        return x


class Solver():
    def __init__(self, model, epoch_num, batch_size, optimizer, device=torch.device('cpu')):
        self.model = model
        self.model.to(device)
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.input_layer_size = model.input_layer_size
        self.criterion = model.criterion
        self.optimizer = optimizer
        self.device = device

    def fit(self, train_loader):
        logs = {}

        for epoch in range(self.epoch_num):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = Variable(data.float()).to(self.device), Variable(target.float()).to(self.device)
                data = data.view(-1, self.input_layer_size)
                target = target.view(-1, self.input_layer_size)
                self.optimizer.zero_grad()
                net_out = self.model(data)
                loss = self.criterion(net_out, target)
                loss.backward()
                self.optimizer.step()
            epoch_loss = loss.detach()
            logs['MSE loss'] = epoch_loss.item()

        print("Number of weight coefficients:", self.model.number_of_weight_coefficients)

            