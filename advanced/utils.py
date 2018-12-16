import torch

is_cuda = torch.cuda.is_available()
map_location = 'cpu' if is_cuda else 'cpu'