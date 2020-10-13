import os
import random
import time

import numpy as np
import jsonlines as jsonl
import torch
import torch.nn.functional as F

from .imdb_data_loader import IMDBDataLoader
from .lstm_model import LSTMModel


def eval_epoch(model, data_loader, device):
    with torch.no_grad():
        model.eval()
        eval_accuracy = 0
        eval_loss = 0
        n = 0
        for X_batch, y_batch in data_loader:
            if device == 'cuda':
                torch.cuda.empty_cache()
            output = model(X_batch)
            loss = F.cross_entropy(output, y_batch)
            eval_loss += loss.item()
            
            eval_accuracy += (output.cpu().argmax(axis=1) == y_batch.cpu()).sum()
            n += len(y_batch)
            
    eval_accuracy = float(eval_accuracy) / n
    eval_loss = float(eval_loss) / len(data_loader)
    return eval_loss, eval_accuracy
 

def train_epoch(model, opt, data_loader, device):
    model.train()
    train_loss = 0
    train_accuracy = 0
    n = 0
    for i, (X_batch, y_batch) in enumerate(data_loader):
        # fwd-bwd, optimize
        if device == 'cuda':
            torch.cuda.empty_cache()
        opt.zero_grad()
        output = model(X_batch)
        loss = F.cross_entropy(output, y_batch)
        loss.backward()
        opt.step()
        
        train_loss += loss.item()
        if np.isnan(train_loss):
            return None, None
        train_accuracy += (output.cpu().argmax(axis=1) == y_batch.cpu()).sum()
        n += len(y_batch)
    
    train_accuracy = float(train_accuracy) / n
    train_loss = float(train_loss) / len(data_loader)
    return train_loss, train_accuracy


def train(params, train_data, val_data, ft_vectors, n_epoch, device, checkpoints_dir):
    results_path = os.path.join(checkpoints_dir, '.'.join(['train_result', 'jsonl']))
    results_f = jsonl.open(results_path, 'w')
    
    eps = 1e-2
    emb_size = len(list(ft_vectors.values())[0])
    model = LSTMModel(emb_size,
                      params['hidden_size'],
                      params['num_layers'],
                      params['dropout'],
                      params['bidirectional'])
    model.to(device)
    random.seed(42)
    data_loader = IMDBDataLoader(train_data, params['batch_size'], ft_vectors, device)
    val_data_loader = IMDBDataLoader(val_data, params['batch_size'], ft_vectors, device)
    opt = torch.optim.Adam(model.parameters(), lr=params['lr'])
    
    for epoch in range(n_epoch):
        start_time = time.time()

        # train epoch
        train_loss, train_accuracy = train_epoch(model, opt, data_loader, device)
        if train_loss is None and train_accuracy is None:
            return None, None, None
        
        # validation epoch
        val_loss, val_accuracy = eval_epoch(model, val_data_loader, device)            

        epoch_time = time.time() - start_time
        print(f'Epoch [{epoch + 1}/{n_epoch}], epoch time: {epoch_time} train loss: {train_loss}, train accuracy: {train_accuracy}, val loss: {val_loss}, val accuracy: {val_accuracy}')
        if val_loss < eps:
            break
                
        checkpoint_path = os.path.join(checkpoints_dir, '.'.join([str(epoch + 1), 'pt']))
        torch.save(model.state_dict, checkpoint_path)
        results_f.write({'epoch': epoch + 1, 
                         'time': epoch_time, 
                         'val_loss': val_loss, 
                         'val_accuracy': val_accuracy, 
                         'train_loss': train_loss, 
                         'train_accuracy': train_accuracy})
    
    results_f.close()
    return model, val_loss, val_accuracy

def test(model, batch_size, test_data, ft_vectors, device, checkpoints_dir):
    results_path = os.path.join(checkpoints_dir, '.'.join(['test_result', 'jsonl']))
    results_f = jsonl.open(results_path, 'w')
    
    model.to(device)
    random.seed(42)
    data_loader = IMDBDataLoader(test_data, batch_size, ft_vectors, device)

    test_loss, test_accuracy = eval_epoch(model, data_loader, device)
    results_f.write({'test_loss': test_loss, 
                     'test_accuracy': test_accuracy})
    
    results_f.close()
    return model, test_loss, test_accuracy

