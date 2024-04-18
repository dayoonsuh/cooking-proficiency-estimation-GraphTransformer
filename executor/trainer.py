
#! /usr/bin/python3
# Author : Kevin Feghoul

import copy
import numpy as np
import time
import torch
from sklearn.metrics import f1_score
from typing import List




def early_stopping(scores:List[float], patience:int, maximisation:bool=True):
    last_idx = len(scores) - 1
    if maximisation:
        idx_val = scores.index(max(scores))
    else:
        idx_val = scores.index(min(scores))
    if abs(last_idx - idx_val) >= patience:
        print("Early stopping ! Did not improve in {} iterations".format(patience))
        return True
    else:
        False


def train_model(
    model, 
    criterion, 
    optimizer, 
    apply_scheduler, 
    scheduler, 
    dataloaders, 
    device, 
    batch_size, 
    num_epochs, 
    early_stop, 
    patience, 
    nb_train, 
    nb_test, 
    print_every
    ):

    print('Model=%s, opt=%s(lr=%f), epochs=%d, device=%s\n' % \
        (type(model).__name__, type(optimizer).__name__,
        optimizer.param_groups[0]['lr'], num_epochs, device))

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())

    history = {}
    history['loss'] = []
    history['val_loss'] = []
    history['acc'] = []
    history['val_acc'] = []
    history['F1'] = []
    history['val_F1'] = []

    best_epoch_F1_val = 0
    best_epoch_acc_val = 0

    for epoch in range(1, num_epochs+1):

        loss_train_tmp, loss_test_tmp = [], []
        running_corrects_train_acc, running_corrects_test_acc = 0, 0
        running_corrects_train_F1, running_corrects_test_F1 = 0, 0
        cpt_train, cpt_test = 0, 0

        # Each epoch has a training and test phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
    
            # Iterate over data.
            for idx, (inputs, labels) in enumerate(dataloaders[phase]):

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    
                    outputs = model(inputs).to(device)
                    _, preds = torch.max(outputs, 1)
                    
                    loss = criterion(outputs, labels)

                    #F1 = f1_score(preds.cpu().numpy(), labels.cpu().numpy(), average='weighted') * 100
                    F1 = f1_score(preds.cpu().numpy(), labels.cpu().numpy(), average='weighted') * 100

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        loss_train_tmp.append(loss.item())
                        running_corrects_train_acc += torch.sum(preds == labels.data).cpu()
                        running_corrects_train_F1 += F1
                        cpt_train += 1

                if phase == 'test':
                    loss_test_tmp.append(loss.item())
                    running_corrects_test_acc += torch.sum(preds == labels.data).cpu()
                    running_corrects_test_F1 += F1
                    cpt_test += 1
                

        history['loss'].append(np.average(loss_train_tmp))
        history['val_loss'].append(np.average(loss_test_tmp))

        history['acc'].append((running_corrects_train_acc / nb_train) * 100)
        history['val_acc'].append((running_corrects_test_acc / nb_test) * 100)

        history['F1'].append((running_corrects_train_F1 / cpt_train))
        history['val_F1'].append((running_corrects_test_F1 / cpt_test))

        if apply_scheduler:
            scheduler.step(history['val_F1'][-1])

        if history['val_acc'][-1] > best_epoch_acc_val and epoch > 1:
            best_epoch_acc_val = history['val_acc'][-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        if epoch % print_every == 0:
            print("Epoch {}/{}, train loss: {:.3f}, test loss: {:.3f}, train acc: {:.3f}, val acc: {:.3f}, train F1: {:.3f}, val F1: {:.3f}, best val acc: {:.3f}".format(
                epoch, num_epochs, history['loss'][-1], history['val_loss'][-1], history['acc'][-1], history['val_acc'][-1], history['F1'][-1], history['val_F1'][-1], max(history['val_acc'])))

        if history['val_acc'][-1] == 100:
            print("Early stoping ! Top score achieved")
            break

        if early_stopping(history['val_acc'], patience):
            break       
    
    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    print("Best acc : {}".format(best_epoch_acc_val))

    model.load_state_dict(best_model_wts)

    # evaluation
    model.eval()
    model.to(device)

    all_labels = []

    with torch.no_grad():

        for idx, (inputs, labels) in enumerate(dataloaders["test"]):

            all_labels.extend(list(labels.to("cpu").numpy()))

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs).to(device)
            
            if idx == 0:
                _, preds = torch.max(outputs, 1)
            else:
                _, tmp_preds = torch.max(outputs, 1)
                preds = torch.cat((preds, tmp_preds), axis=0)


    return model, history, preds, np.array(all_labels)



def train_model_gnn(model, criterion, optimizer, apply_scheduler, scheduler, dataloaders, device, num_epochs, patience, nb_train, nb_test, print_every):

    print('Model=%s, opt=%s(lr=%f), epochs=%d, device=%s\n' % \
        (type(model).__name__, type(optimizer).__name__,
        optimizer.param_groups[0]['lr'], num_epochs, device))

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())

    history = {}
    history['loss'] = []
    history['val_loss'] = []
    history['acc'] = []
    history['val_acc'] = []
    history['F1'] = []
    history['val_F1'] = []

    best_epoch_F1_val = 0
    best_epoch_acc_val = 0

    for epoch in range(1, num_epochs+1):

        loss_train_tmp, loss_test_tmp = [], []
        running_corrects_train_acc, running_corrects_test_acc = 0, 0
        running_corrects_train_F1, running_corrects_test_F1 = 0, 0
        cpt_train, cpt_test = 0, 0

        # Each epoch has a training and test phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
    
            # Iterate over data.
            for idx, data in enumerate(dataloaders[phase]):
                data = data.to(device)

                #print(data.x.shape, data.y.shape, data.edge_index.shape)
                
                

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    
                    outputs = model(data.x, data.edge_index, data.batch, data.y.shape[0])#.to(device)
                    _, preds = torch.max(outputs, 1)
                    
                    loss = criterion(outputs, data.y)
                    print(loss)

                    #F1 = f1_score(preds.cpu().numpy(), labels.cpu().numpy(), average='weighted') * 100
                    F1 = f1_score(preds.cpu().numpy(), data.y.cpu().numpy(), average='macro') * 100
                    #print(preds[:20], labels[:20])
                    #print(labels.shape, preds.shape, F1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        loss_train_tmp.append(loss.item())
                        running_corrects_train_acc += torch.sum(preds == data.y.data).cpu()
                        running_corrects_train_F1 += F1
                        cpt_train += 1

                if phase == 'test':
                    loss_test_tmp.append(loss.item())
                    running_corrects_test_acc += torch.sum(preds == data.y.data).cpu()
                    running_corrects_test_F1 += F1
                    cpt_test += 1
                

        history['loss'].append(np.average(loss_train_tmp))
        history['val_loss'].append(np.average(loss_test_tmp))

        history['acc'].append((running_corrects_train_acc / nb_train) * 100)
        history['val_acc'].append((running_corrects_test_acc / nb_test) * 100)

        history['F1'].append((running_corrects_train_F1 / cpt_train))
        history['val_F1'].append((running_corrects_test_F1 / cpt_test))

        if apply_scheduler:
            scheduler.step(history['val_F1'][-1])

        if history['val_acc'][-1] > best_epoch_acc_val: # and history['val_F1'][-1] > 0:
            best_epoch_acc_val = history['val_acc'][-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        '''
        if epoch % print_every == 0:
            print("Epoch {}/{}, train loss: {:.3f}, test loss: {:.3f}, train acc: {:.3f}, val acc: {:.3f}, train F1: {:.3f}, val F1: {:.3f}, best val F1: {:.3f}".format(
                epoch, num_epochs, history['loss'][-1], history['val_loss'][-1], history['acc'][-1], history['val_acc'][-1], history['F1'][-1], history['val_F1'][-1], max(history['val_F1'])))
        '''

        if epoch % print_every == 0:
            print("Epoch {}/{}, train loss: {:.3f}, test loss: {:.3f}, train acc: {:.3f}, val acc: {:.3f}, train F1: {:.3f}, val F1: {:.3f}, best val acc: {:.3f}".format(
                epoch, num_epochs, history['loss'][-1], history['val_loss'][-1], history['acc'][-1], history['val_acc'][-1], history['F1'][-1], history['val_F1'][-1], max(history['val_acc'])))

        if history['val_acc'][-1] == 100:
            print("Early stoping ! Top score achieved")
            break

        if early_stopping(history['val_acc'], patience):
            break
                
    
    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    #print("Best acc : {}".format(best_epoch_acc_val))

    print("Best acc : {}".format(best_epoch_acc_val))

    model.load_state_dict(best_model_wts)

    # evaluation
    model.eval()
    model.to(device)

    all_labels = []

    with torch.no_grad():

        for idx, data in enumerate(dataloaders["test"]):

            all_labels.extend(list(data.y.to("cpu").numpy()))

            data = data.to(device)

            outputs = model(data.x, data.edge_index, data.batch, data.y.shape[0])
            
            if idx == 0:
                _, preds = torch.max(outputs, 1)
            else:
                _, tmp_preds = torch.max(outputs, 1)
                preds = torch.cat((preds, tmp_preds), axis=0)


    return model, history, preds, np.array(all_labels)



def train_model_stgcn(model, edge_index, criterion, optimizer, apply_scheduler, scheduler, dataloaders,  device, num_epochs, patience, nb_train, nb_test, print_every):

    print('Model=%s, opt=%s(lr=%f), epochs=%d, device=%s\n' % \
        (type(model).__name__, type(optimizer).__name__,
        optimizer.param_groups[0]['lr'], num_epochs, device))

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())

    history = {}
    history['loss'] = []
    history['val_loss'] = []
    history['acc'] = []
    history['val_acc'] = []
    history['F1'] = []
    history['val_F1'] = []

    best_epoch_F1_val = 0
    best_epoch_acc_val = 0

    for epoch in range(1, num_epochs+1):

        loss_train_tmp, loss_test_tmp = [], []
        running_corrects_train_acc, running_corrects_test_acc = 0, 0
        running_corrects_train_F1, running_corrects_test_F1 = 0, 0
        cpt_train, cpt_test = 0, 0

        # Each epoch has a training and test phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
    
            # Iterate over data.
            for idx, (inputs, labels) in enumerate(dataloaders[phase]):

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    # (B, N_nodes, F_in, T_in) - bs, 21, 3, 600
                    # 64, 600, 63
                    # (batch_size, N, F_in, T)

                    bs, ws, feat_dim = inputs.shape

                    inputs = inputs.reshape((bs, 21, 3, ws))
        
                    
                    outputs = model(inputs)

                   # print(inputs.shape, outputs.shape)

                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)

                    #F1 = f1_score(preds.cpu().numpy(), data1.y.cpu().numpy(), average='weighted') * 100
                    F1 = f1_score(preds.cpu().numpy(), labels.cpu().numpy(), average='binary') * 100

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        loss_train_tmp.append(loss.item())
                        running_corrects_train_acc += torch.sum(preds == labels).cpu()
                        running_corrects_train_F1 += F1
                        cpt_train += 1

                if phase == 'test':
                    loss_test_tmp.append(loss.item())
                    running_corrects_test_acc += torch.sum(preds == labels).cpu()
                    running_corrects_test_F1 += F1
                    cpt_test += 1
                

        history['loss'].append(np.average(loss_train_tmp))
        history['val_loss'].append(np.average(loss_test_tmp))

        history['acc'].append((running_corrects_train_acc / nb_train) * 100)
        history['val_acc'].append((running_corrects_test_acc / nb_test) * 100)

        history['F1'].append((running_corrects_train_F1 / cpt_train))
        history['val_F1'].append((running_corrects_test_F1 / cpt_test))

        if apply_scheduler:
            scheduler.step(history['val_F1'][-1])

        '''
        if history['val_acc'][-1] > best_epoch_acc_val and epoch > 1:
            best_epoch_acc_val = history['val_acc'][-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        '''

        
        if history['val_F1'][-1] > best_epoch_F1_val and epoch > 1:
            best_epoch_F1_val = history['val_F1'][-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        '''
        if epoch % print_every == 0:
            print("Epoch {}/{}, train loss: {:.3f}, test loss: {:.3f}, train acc: {:.3f}, val acc: {:.3f}, train F1: {:.3f}, val F1: {:.3f}, best val F1: {:.3f}".format(
                epoch, num_epochs, history['loss'][-1], history['val_loss'][-1], history['acc'][-1], history['val_acc'][-1], history['F1'][-1], history['val_F1'][-1], max(history['val_F1'])))
        '''

        if epoch % print_every == 0:
            print("Epoch {}/{}, train loss: {:.3f}, test loss: {:.3f}, train acc: {:.3f}, val acc: {:.3f}, train F1: {:.3f}, val F1: {:.3f}, best val acc: {:.3f}".format(
                epoch, num_epochs, history['loss'][-1], history['val_loss'][-1], history['acc'][-1], history['val_acc'][-1], history['F1'][-1], history['val_F1'][-1], max(history['val_acc'])))

        if history['val_acc'][-1] == 100:
            print("Early stoping ! Top score achieved")
            break

        if early_stopping(history['val_acc'], patience):
            break
                
    
    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    #print("Best acc : {}".format(best_epoch_acc_val))

    print("Best acc : {}".format(best_epoch_acc_val))

    model.load_state_dict(best_model_wts)

    # evaluation
    model.eval()
    model.to(device)

    all_labels = []

    with torch.no_grad():

        for idx, (inputs, labels) in enumerate(dataloaders['test']):

            inputs = inputs.to(device)
            labels = labels.to(device)

            all_labels.extend(list(labels.cpu().numpy()))
            
            bs, ws, feat_dim = inputs.shape
            inputs = inputs.reshape((bs, 21, 3, ws))
    
            #print(inputs.shape)
            
            outputs = model(inputs)
            
            if idx == 0:
                _, preds = torch.max(outputs, 1)
            else:
                _, tmp_preds = torch.max(outputs, 1)
                preds = torch.cat((preds, tmp_preds), axis=0)


    return model, history, preds, np.array(all_labels)



def train_model_stgcn2(model, criterion, optimizer, apply_scheduler, scheduler, dataloaders,  device, num_epochs, patience, nb_train, nb_test, print_every):

    print('Model=%s, opt=%s(lr=%f), epochs=%d, device=%s\n' % \
        (type(model).__name__, type(optimizer).__name__,
        optimizer.param_groups[0]['lr'], num_epochs, device))

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())

    history = {}
    history['loss'] = []
    history['val_loss'] = []
    history['acc'] = []
    history['val_acc'] = []
    history['F1'] = []
    history['val_F1'] = []

    best_epoch_F1_val = 0
    best_epoch_acc_val = 0

    for epoch in range(1, num_epochs+1):

        loss_train_tmp, loss_test_tmp = [], []
        running_corrects_train_acc, running_corrects_test_acc = 0, 0
        running_corrects_train_F1, running_corrects_test_F1 = 0, 0
        cpt_train, cpt_test = 0, 0

        # Each epoch has a training and test phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
    
            # Iterate over data.
            for idx, (inputs, labels) in enumerate(dataloaders[phase]):

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    # (B, N_nodes, F_in, T_in) - bs, 21, 3, 600
                    # 64, 600, 63
                    # (batch_size, N, F_in, T)

                    #print(inputs.shape)

                    bs, ws, _ = inputs.shape

                    #inputs = inputs.reshape((bs, 21, 3, ws))

                    inputs = inputs.reshape((bs, 3, ws, 21, 1))
                    #print(inputs.shape)
        
                    
                    outputs = model(inputs)

                   # print(inputs.shape, outputs.shape)

                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)

                    #F1 = f1_score(preds.cpu().numpy(), data1.y.cpu().numpy(), average='weighted') * 100
                    F1 = f1_score(preds.cpu().numpy(), labels.cpu().numpy(), average='binary') * 100

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        loss_train_tmp.append(loss.item())
                        running_corrects_train_acc += torch.sum(preds == labels).cpu()
                        running_corrects_train_F1 += F1
                        cpt_train += 1

                if phase == 'test':
                    loss_test_tmp.append(loss.item())
                    running_corrects_test_acc += torch.sum(preds == labels).cpu()
                    running_corrects_test_F1 += F1
                    cpt_test += 1
                

        history['loss'].append(np.average(loss_train_tmp))
        history['val_loss'].append(np.average(loss_test_tmp))

        history['acc'].append((running_corrects_train_acc / nb_train) * 100)
        history['val_acc'].append((running_corrects_test_acc / nb_test) * 100)

        history['F1'].append((running_corrects_train_F1 / cpt_train))
        history['val_F1'].append((running_corrects_test_F1 / cpt_test))

        if apply_scheduler:
            scheduler.step(history['val_F1'][-1])

        '''
        if history['val_acc'][-1] > best_epoch_acc_val and epoch > 1:
            best_epoch_acc_val = history['val_acc'][-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        '''

        
        if history['val_F1'][-1] > best_epoch_F1_val and epoch > 1:
            best_epoch_F1_val = history['val_F1'][-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        '''
        if epoch % print_every == 0:
            print("Epoch {}/{}, train loss: {:.3f}, test loss: {:.3f}, train acc: {:.3f}, val acc: {:.3f}, train F1: {:.3f}, val F1: {:.3f}, best val F1: {:.3f}".format(
                epoch, num_epochs, history['loss'][-1], history['val_loss'][-1], history['acc'][-1], history['val_acc'][-1], history['F1'][-1], history['val_F1'][-1], max(history['val_F1'])))
        '''

        if epoch % print_every == 0:
            print("Epoch {}/{}, train loss: {:.3f}, test loss: {:.3f}, train acc: {:.3f}, val acc: {:.3f}, train F1: {:.3f}, val F1: {:.3f}, best val acc: {:.3f}".format(
                epoch, num_epochs, history['loss'][-1], history['val_loss'][-1], history['acc'][-1], history['val_acc'][-1], history['F1'][-1], history['val_F1'][-1], max(history['val_acc'])))

        if history['val_acc'][-1] == 100:
            print("Early stoping ! Top score achieved")
            break

        if early_stopping(history['val_acc'], patience):
            break
                
    
    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    #print("Best acc : {}".format(best_epoch_acc_val))

    print("Best acc : {}".format(best_epoch_acc_val))

    model.load_state_dict(best_model_wts)

    # evaluation
    model.eval()
    model.to(device)

    all_labels = []

    with torch.no_grad():

        for idx, (inputs, labels) in enumerate(dataloaders['test']):

            inputs = inputs.to(device)
            labels = labels.to(device)

            all_labels.extend(list(labels.cpu().numpy()))
            
            bs, ws, _ = inputs.shape
            inputs = inputs.reshape((bs, 3, ws, 21, 1))
    
            #print(inputs.shape)
            
            outputs = model(inputs)
            
            if idx == 0:
                _, preds = torch.max(outputs, 1)
            else:
                _, tmp_preds = torch.max(outputs, 1)
                preds = torch.cat((preds, tmp_preds), axis=0)


    return model, history, preds, np.array(all_labels)



def train_model_2f(model, criterion, optimizer, apply_scheduler, scheduler, dataloaders, device, num_epochs, patience, nb_train, nb_test, print_every):

    print('Model=%s, opt=%s(lr=%f), epochs=%d, device=%s\n' % \
        (type(model).__name__, type(optimizer).__name__,
        optimizer.param_groups[0]['lr'], num_epochs, device))

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())

    history = {}
    history['loss'] = []
    history['val_loss'] = []
    history['acc'] = []
    history['val_acc'] = []

    best_epoch_acc_val = 0

    for epoch in range(1, num_epochs+1):

        loss_train_tmp, loss_test_tmp = [], []
        running_corrects_train, running_corrects_test = 0, 0

        # Each epoch has a training and test phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
    
            # Iterate over data.
            for idx, (inp1, inp2, labels) in enumerate(dataloaders[phase]):

                
                inp1 = inp1.to(device)
                inp2 = inp2.to(device)
                inputs = [inp1, inp2]
                    
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs).to(device)
                    _, preds = torch.max(outputs, 1)
                    #print(outputs.shape, labels.shape)
                    #exit()
                    loss = criterion(outputs, labels)

                    #loss += 0.1

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        loss_train_tmp.append(loss.item())
                        running_corrects_train += torch.sum(preds == labels.data).cpu()

                if phase == 'test':
                    loss_test_tmp.append(loss.item())
                    running_corrects_test += torch.sum(preds == labels.data).cpu()
                

        history['loss'].append(np.average(loss_train_tmp))
        history['val_loss'].append(np.average(loss_test_tmp))

        history['acc'].append((running_corrects_train / nb_train) * 100)
        history['val_acc'].append((running_corrects_test / nb_test) * 100)

        if apply_scheduler:
            scheduler.step(history['val_acc'][-1])

        if history['val_acc'][-1] > best_epoch_acc_val:
            best_epoch_acc_val = history['val_acc'][-1]
            best_model_wts = copy.deepcopy(model.state_dict())
    
        if epoch % print_every == 0:
            print("Epoch {}/{}, train loss: {:.3f}, test loss: {:.3f}, train acc: {:.3f}, val acc: {:.3f}, best val acc: {:.3f}".format(
                epoch, num_epochs, history['loss'][-1], history['val_loss'][-1], history['acc'][-1], history['val_acc'][-1], max(history['val_acc'])))

        if history['val_acc'][-1] == 100:
            print("Early stoping ! Top score achieved")
            break

        if early_stopping(history['val_acc'], patience):
            break
                
    
    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    print("Best accuracy : {}".format(best_epoch_acc_val))

    model.load_state_dict(best_model_wts)

    # evaluation
    model.eval()
    model.to(device)

    all_labels = []

    with torch.no_grad():

        for idx, (inp1, inp2, labels) in enumerate(dataloaders["test"]):

            all_labels.extend(list(labels.to("cpu").numpy()))

            inp1 = inp1.to(device)
            inp2 = inp2.to(device)
            inputs = [inp1, inp2]
            
            labels = labels.to(device)

            outputs = model(inputs).to(device)
            
            if idx == 0:
                _, preds = torch.max(outputs, 1)
            else:
                _, tmp_preds = torch.max(outputs, 1)
                preds = torch.cat((preds, tmp_preds), axis=0)

    return model, history, preds, np.array(all_labels)



def train_model_3f(model, criterion, optimizer, apply_scheduler, scheduler, dataloaders, device, num_epochs, patience, nb_train, nb_test, print_every):

    print('Model=%s, opt=%s(lr=%f), epochs=%d, device=%s\n' % \
        (type(model).__name__, type(optimizer).__name__,
        optimizer.param_groups[0]['lr'], num_epochs, device))

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())

    history = {}
    history['loss'] = []
    history['val_loss'] = []
    history['acc'] = []
    history['val_acc'] = []
    history['F1'] = []
    history['val_F1'] = []

    best_epoch_F1_val = 0
    best_epoch_acc_val = 0


    for epoch in range(1, num_epochs+1):

        loss_train_tmp, loss_test_tmp = [], []
        running_corrects_train_acc, running_corrects_test_acc = 0, 0
        running_corrects_train_F1, running_corrects_test_F1 = 0, 0
        cpt_train, cpt_test = 0, 0

        # Each epoch has a training and test phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
    
            # Iterate over data.
            for idx, (inp1, inp2, inp3, labels) in enumerate(dataloaders[phase]):

                
                inp1 = inp1.to(device)
                inp2 = inp2.to(device)
                inp3 = inp3.to(device)
                inputs = [inp1, inp2, inp3]
                    
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs).to(device)
                    _, preds = torch.max(outputs, 1)
                    #print(outputs.shape, labels.shape)
                    #exit()
                    loss = criterion(outputs, labels)

                     #F1 = f1_score(preds.cpu().numpy(), labels.cpu().numpy(), average='weighted') * 100
                    F1 = f1_score(preds.cpu().numpy(), labels.cpu().numpy(), average='binary') * 100
                    #print(preds[:20], labels[:20])
                    #print(labels.shape, preds.shape, F1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        loss_train_tmp.append(loss.item())
                        running_corrects_train_acc += torch.sum(preds == labels.data).cpu()
                        running_corrects_train_F1 += F1
                        cpt_train += 1

                if phase == 'test':
                    loss_test_tmp.append(loss.item())
                    running_corrects_test_acc += torch.sum(preds == labels.data).cpu()
                    running_corrects_test_F1 += F1
                    cpt_test += 1

                   
                

        history['loss'].append(np.average(loss_train_tmp))
        history['val_loss'].append(np.average(loss_test_tmp))

        history['acc'].append((running_corrects_train_acc / nb_train) * 100)
        history['val_acc'].append((running_corrects_test_acc / nb_test) * 100)

        history['F1'].append((running_corrects_train_F1 / cpt_train))
        history['val_F1'].append((running_corrects_test_F1 / cpt_test))

        if apply_scheduler:
            scheduler.step(history['val_F1'][-1])

        if history['val_acc'][-1] > best_epoch_acc_val and epoch > 1:
            best_epoch_acc_val = history['val_acc'][-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        if apply_scheduler:
            scheduler.step(history['val_acc'][-1])

        if history['val_acc'][-1] > best_epoch_acc_val:
            best_epoch_acc_val = history['val_acc'][-1]
            best_model_wts = copy.deepcopy(model.state_dict())
    
        if epoch % print_every == 0:
            print("Epoch {}/{}, train loss: {:.3f}, test loss: {:.3f}, train acc: {:.3f}, val acc: {:.3f}, train F1: {:.3f}, val F1: {:.3f}, best val acc: {:.3f}".format(
                epoch, num_epochs, history['loss'][-1], history['val_loss'][-1], history['acc'][-1], history['val_acc'][-1], history['F1'][-1], history['val_F1'][-1], max(history['val_acc'])))

        if history['val_acc'][-1] == 100:
            print("Early stoping ! Top score achieved")
            break

        if early_stopping(history['val_acc'], patience):
            break
                
    
    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    print("Best accuracy : {}".format(best_epoch_acc_val))

    model.load_state_dict(best_model_wts)

    # evaluation
    model.eval()
    model.to(device)

    all_labels = []

    with torch.no_grad():

        for idx, (inp1, inp2, inp3, labels) in enumerate(dataloaders["test"]):

            all_labels.extend(list(labels.to("cpu").numpy()))

            inp1 = inp1.to(device)
            inp2 = inp2.to(device)
            inp3 = inp3.to(device)
            inputs = [inp1, inp2, inp3]
            
            labels = labels.to(device)

            outputs = model(inputs).to(device)
            
            if idx == 0:
                _, preds = torch.max(outputs, 1)
            else:
                _, tmp_preds = torch.max(outputs, 1)
                preds = torch.cat((preds, tmp_preds), axis=0)

    return model, history, preds, np.array(all_labels)


