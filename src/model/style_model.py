import os

import torch

from .msg_net import Net

dn = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(dn, "21styles.model")

style_model = Net(ngf=128)
style_model.load_state_dict(torch.load(model_path), False)
