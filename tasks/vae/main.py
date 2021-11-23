from __future__ import print_function

import torch.utils.data

from Trainer import Trainer
from VAE import VAE

trainer = Trainer()

for epoch in range(1, 10):
    trainer.train(epoch)
    trainer.test(epoch)
    with torch.no_grad():
        sample = torch.randn(64, 20).to(trainer.device)
        sample = trainer.model.decode(sample).cpu()

trainer.save()

model = VAE(trainer.n_obs)
model.load_state_dict(torch.load('./runs/vae/vae_state_dict'))