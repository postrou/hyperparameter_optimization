import os
import random
import time

import numpy as np
import jsonlines as jsonl
import torch
import torch.nn.functional as F

from .imdb_data_loader import IMDBDataLoader
from .lstm_model import LSTMModel


def train_model(params, train_data, val_data, ft_vectors, n_epoch, emb_size, device, checkpoints_dir, print_every=10):
    results_path = os.path.join(checkpoints_dir, '.'.join(['train_result', 'jsonl']))
    results_f = jsonl.open(results_path, 'w')
    
    eps = 1e-2
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
        model.train()
        start_time = time.time()
        train_loss = 0
        train_accuracy = 0
        running_loss = 0
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
            running_loss += loss.item()
            if np.isnan(running_loss):
                return None, None, None
            train_accuracy += (output.cpu().argmax(axis=1) == y_batch.cpu()).sum()
            n += len(y_batch)
            
            if (i + 1) % print_every == 0:
#                 logging.debug(running_loss)
                running_loss /= print_every
                print(f'Epoch: [{epoch + 1}/{n_epoch}], batch: [{i + 1}/{len(data_loader)}], time: {time.time() - start_time}, loss: {running_loss}')
#                 log_f.write(f'Epoch: [{epoch + 1}/{n_epoch}], batch: [{i + 1}/{len(data_loader)}], time: {time.time() - start_time}, loss: {running_loss}\n')
                running_loss = 0
        
        train_accuracy = float(train_accuracy) / n
        train_loss = float(train_loss) / len(data_loader)
        
        # validation on each epoch
        with torch.no_grad():
            model.eval()
            val_accuracy = 0
            val_loss = 0
            n = 0
            for X_batch, y_batch in val_data_loader:
                if device == 'cuda':
                    torch.cuda.empty_cache()
                output = model(X_batch)
                loss = F.cross_entropy(output, y_batch)
                val_loss += loss.item()
                
                val_accuracy += (output.cpu().argmax(axis=1) == y_batch.cpu()).sum()
#                 print(list(zip(output.cpu().argmax(axis=1), y_batch)))
#                 print((output.cpu().argmax(axis=1) == y_batch))
                n += len(y_batch)
                
            val_accuracy = float(val_accuracy) / n
            val_loss = float(val_loss) / len(val_data_loader)
            
            epoch_time = time.time() - start_time
            print(f'Epoch [{epoch + 1}/{n_epoch}], epoch time: {epoch_time} train loss: {train_loss}, train accuracy: {train_accuracy}, val loss: {val_loss}, val accuracy: {val_accuracy}')
#             log_f.write(f'Epoch [{epoch + 1}/{n_epoch}], val loss: {val_loss}, val accuracy: {val_accuracy}\n')
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

def test_model(model, batch_size, test_data, ft_vectors, device, checkpoints_dir):
    results_path = os.path.join(checkpoints_dir, '.'.join(['test_result', 'jsonl']))
    results_f = jsonl.open(results_path, 'w')
    
    model.to(device)
    random.seed(42)
    data_loader = IMDBDataLoader(test_data, batch_size, ft_vectors, device)
    model.eval()
    
    test_loss = 0
    test_accuracy = 0
    n = 0
    for i, (X_batch, y_batch) in enumerate(data_loader):
        # fwd-bwd, optimize
        if device == 'cuda':
            torch.cuda.empty_cache()

        output = model(X_batch)
        loss = F.cross_entropy(output, y_batch)

        test_loss += loss.item()
        test_accuracy += (output.cpu().argmax(axis=1) == y_batch.cpu()).sum()
        n += len(y_batch)

    test_accuracy = float(test_accuracy) / n
    test_loss = float(test_loss) / len(data_loader)
                
    results_f.write({'test_loss': test_loss, 
                     'test_accuracy': test_accuracy})
    
    results_f.close()
    return model, test_loss, test_accuracy

