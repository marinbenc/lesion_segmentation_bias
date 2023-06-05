import os
from os import makedirs
import os.path as p
import json

import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

def save_args(args, folder):
    args_file = os.path.join(folder, 'args.json')
    makedirs(folder, exist_ok=True)
    with open(args_file, 'w') as fp:
      if isinstance(args, dict):
        json.dump(args, fp)
      else:
        json.dump(vars(args), fp)

def save_checkpoint(name, log_dir, model, epoch, optimizer, loss):
    file_name = p.join(log_dir, name)
    torch.save({
        #'epoch': epoch,
        'model': model.state_dict(),
        #'optimizer': optimizer.state_dict(),
        #'loss': loss
    }, file_name)

class Trainer:
  def __init__(self, model, optimizer, loss_fn, train_loader, val_loader, validation_fn, log_dir, checkpoint_name, scheduler=None, device='cuda'):
    self.model = model
    self.optimizer = optimizer
    self.loss_fn = loss_fn
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.device = device
    self.log_dir = log_dir
    self.checkpoint_name = checkpoint_name
    self.scheduler = scheduler
    self.device = device
    self.validation_fn = validation_fn

    self.writer = SummaryWriter(log_dir=self.log_dir)
    self.best_loss = float('inf')
    self.epochs_since_best = 0

  def _to_device(self, data):
    if isinstance(data, (list, tuple)):
      if len(data) == 1:
        return self._to_device(data[0])
      return [self._to_device(x) for x in data]
    elif isinstance(data, dict):
      return {k: self._to_device(v) for k, v in data.items()}
    else:
      return data.to(self.device, non_blocking=True)

  def train(self, epochs):
    self.epochs_since_best = 0
    self.best_loss = float('inf')

    for epoch in range(epochs):
      early_stop = self._train_epoch(epoch)
      if early_stop:
        break

  def _train_epoch(self, epoch):
    """
    Returns:
      early_stop: True if early stopping should be performed
    """
    if epoch == 0:
      self.best_loss = float('inf')
    
    self.model.train()
    loss_total = 0
    for batch_idx, batch in enumerate(self.train_loader):
      input = self._to_device(self.get_input(batch))
      target = self._to_device(self.get_target(batch))

      #print(input.shape)
      #print(target.shape)
      #print(target)

      self.optimizer.zero_grad()
      output = self.model(input)
      #print(output.shape)
      loss = self.loss_fn(output, target)
      loss.backward()
      self.optimizer.step()
      loss_total += loss.item()

      #show_torch(imgs=[input[0], target['seg'][0]])

    loss_total /= len(self.train_loader)
    self.writer.add_scalar('Loss/train', loss_total, epoch)

    print(f'Train Epoch: {epoch}\tTrain Loss: {loss_total:.6f}', end='', flush=True)

    if self.val_loader is not None:
      loss_total = 0
      self.model.eval()
      with torch.no_grad():
        for batch in self.val_loader:
          input = self._to_device(self.get_input(batch))
          target = self._to_device(self.get_target(batch))
          output = self.model(input)
          loss = self.validation_fn(output, target)
          loss_total += loss.item()
      loss_total /= len(self.val_loader)
      self.writer.add_scalar('Loss/valid', loss_total, epoch)
      print(f'\tValid Loss: {loss_total:.6f}', end='', flush=True)
    
    print()

    if self.scheduler is not None:
      self.scheduler.step(loss_total)

    if loss_total < self.best_loss and True:
        print('Saving new best model...')
        self.best_loss = loss_total
        self.epochs_since_best = 0
        save_checkpoint(self.checkpoint_name, self.writer.log_dir, self.model, epoch, self.optimizer, loss_total)
    
    if self.epochs_since_best > 10:
      print('Early stopping')
      return True
    
    # if (epoch) % 20 == 0:
    # show_torch(imgs=[input[0], target['seg'][0]])

    self.epochs_since_best += 1
    return False

  def get_input(self, batch):
    """Convert data loader output to input of the model"""
    return batch[0]

  def get_target(self, batch):
    """Convert data loader output to target of the model"""
    return batch[1:]