import os.path


import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import warnings
warnings.filterwarnings('ignore')

from SP500DataModule import SP500DataModule
from SP500ClassifierModel import SP500ClassifierModel

epochs = 30
batch_size = 48
learning_rate = 0.0001

train_val_test_split=[80,10,10]
gpus = 1 if torch.cuda.is_available() else 0
progress_bar_refresh_rate=16

datamodule = SP500DataModule(batch_size=batch_size, train_val_test_split=train_val_test_split)
model = SP500ClassifierModel(learning_rate=learning_rate)

trainer = pl.Trainer(gpus=gpus, max_epochs=epochs, progress_bar_refresh_rate=progress_bar_refresh_rate)

trainer.fit(model, datamodule)

trainer.test(datamodule=datamodule)

# 4. Run learning finder variation 1: auto_lr_find=True
datamodule = SP500DataModule(batch_size=batch_size, train_val_test_split=train_val_test_split)
model = SP500ClassifierModel(learning_rate=learning_rate)

trainer = pl.Trainer(gpus=gpus, max_epochs=epochs, progress_bar_refresh_rate=progress_bar_refresh_rate, auto_lr_find=True)

trainer.tune(model, datamodule)

# 5. Run learning finder variation 2: trainer.tuner.lr_find
datamodule = SP500DataModule(batch_size=batch_size, train_val_test_split=train_val_test_split)
model = SP500ClassifierModel(learning_rate=learning_rate)

trainer = pl.Trainer(min_epochs=5)
lr_finder = trainer.tuner.lr_find(model, datamodule)

# Results can be found in
print(lr_finder.results)

# Plot with
fig = lr_finder.plot(suggest=True)
fig.show()

# Pick point based on plot, or get suggestion
new_lr = lr_finder.suggestion()
print(new_lr)

# update hparams of the model
model.hparams.lr = new_lr

# Train with early stop
early_stop_callback = EarlyStopping(monitor="val_accuracy", min_delta=0.00, patience=3, verbose=False, mode="max")

# saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
if not os.path.exists('models'):
    os.makedirs('models')

checkpoint_callback = ModelCheckpoint(
    monitor="val_accuracy",
    dirpath="models",
    filename="sp500-{epoch:02d}-{val_accuracy:.2f}",
    save_top_k = 5,
    mode="max"
)

trainer = pl.Trainer(callbacks=[early_stop_callback, checkpoint_callback], progress_bar_refresh_rate=progress_bar_refresh_rate, max_epochs=epochs)

trainer.fit(model, datamodule)
trainer.test(datamodule=datamodule)


# Start tensorboard.
#%load_ext tensorboard
#%tensorboard --logdir lightning_logs/
