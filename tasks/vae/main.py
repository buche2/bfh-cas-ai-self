from __future__ import print_function

import pandas as pd
import torch.utils.data
import numpy as np

from Trainer import Trainer
from VAE import VAE
from MyDataset import MyDataset

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

dataset = MyDataset()
train_loader, test_loader, val_loader = dataset.get_loaders()

samples = []
for i in range(100):
    with torch.no_grad():
        sample_features, _, _ = model(train_loader.dataset.tensors[0])

        #print(sample_features.shape)
        #print(train_loader.dataset.tensors[0].shape) # features size
        #print(train_loader.dataset.tensors[1].shape) # labels size
        #print(train_loader.dataset.tensors[1]) # labels
        sample_labels = train_loader.dataset.tensors[1];

        new_data_sample = torch.cat([sample_features, sample_labels], dim=1)
        samples.append(new_data_sample.numpy())

print(np.vstack(samples).shape) # beinhaltet tensoren

#print(pd.DataFrame(np.vstack(samples)))

generated_np = np.vstack(samples)
pd.DataFrame(np.vstack(generated_np))

original = train_loader.dataset.tensors[0].detach().numpy()[0, :]
generated_np = generated_np[0, :-1]


import matplotlib.pyplot as plt

plt.plot(original)
plt.plot(generated_np)

plt.show()


new_data_sample = torch.cat([torch.from_numpy(original), torch.from_numpy(generated_np)])

trainer = Trainer()

for epoch in range(1, 10):
    trainer.train(epoch)
    trainer.test(epoch)
    with torch.no_grad():
        sample = torch.randn(64, 20).to(trainer.device)
        sample = trainer.model.decode(new_data_sample).cpu()

trainer.save()

model = VAE(trainer.n_obs)
model.load_state_dict(torch.load('./runs/vae/vae_state_dict'))

dataset = MyDataset()
train_loader, test_loader, val_loader = dataset.get_loaders()

import matplotlib.pyplot as plt

plt.plot(original)
plt.plot(new_data_sample)

plt.show()