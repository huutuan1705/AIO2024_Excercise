import time
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_epoch(model, optimizer, criterion, train_dataloader, device, epoch=0, log_interval=50):
    model.train()
    total_acc, total_count = 0, 0
    losses = []
    start_time = time.time()
    
    for idx, (inputs, labels) in enumerate(tqdm(train_dataloader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        predictions = model(inputs)
        
        # compute loss
        loss = criterion(predictions, labels)
        losses.append(loss.item())
        
        # backward
        loss.backward()
        optimizer.step()
        total_acc += (predictions.argmax(1) == labels).sum().item()
        total_count += labels.size(0)
        
        if idx%log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(
                "Epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f}".format(epoch, idx, len(train_dataloader), total_acc/total_count)
            )
    
            total_acc, total_count = 0, 0
            start_time = time.time()
        
    epoch_acc = total_acc / total_count
    epoch_loss = sum(losses) / len(losses)
    
    return epoch_acc, epoch_loss

def evaluate_epoch(model, criterion, valid_dataloader, device):
    model.eval()
    total_acc, total_count = 0, 0
    losses = []
    
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(tqdm(valid_dataloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            predictions = model(inputs)
            
            # compute loss
            loss = criterion(predictions, labels)
            losses.append(loss.item())           
            total_acc += (predictions.argmax(1) == labels).sum().item()
            total_count += labels.size(0)
    
    epoch_acc = total_acc / total_count
    epoch_loss = sum(losses) / len(losses)
    
    return epoch_acc, epoch_loss

def train(model, model_name, save_model, optimizer, criterion, train_dataloader, 
          valid_dataloader, num_epochs, device):
    train_accs, train_losses = [], []
    eval_accs, eval_losses = [], []
    best_loss_eval = 100
    times = []
    
    for epoch in range(1, num_epochs+1):
        # Training
        train_acc, train_loss = train_epoch(model, optimizer, criterion, train_dataloader,
                                            device, epoch)
        train_accs.append(train_acc)
        train_losses.append(train_loss)
        
        # Evaluation
        eval_acc, eval_loss = evaluate_epoch(model, criterion, valid_dataloader, device)
        eval_accs.append(eval_acc)
        eval_losses.append(eval_loss)
        
        # Save best model
        if eval_loss < best_loss_eval:
            torch.save(model.state_dict(), save_model + f'/{model_name}.pt')
    
        print("-"*59)
        print(
            "End of epoch: {:3d} | Train accuracy: {:8.3f} | Train loss: {:8.3f}\n"
            "Valid accuracy: {:8.3f} | Valid loss: {:8.3f}".format(epoch, train_acc, train_loss, eval_acc, eval_loss)
        )    
        print("-"*59)
        
    # Load best model
    model.load_state_dict(torch.load(save_model + f'/{model_name}.pt'))
    model.eval()
    metrics = {
        "train_accuracy": train_acc,
        "train_loss": train_loss,
        "valid_accuracy": eval_acc,
        "valid_loss": eval_loss
    }
    
    return model, metrics   

def plot_result(num_epochs, train_accs, eval_accs, train_losses, eval_losses):
    epochs = len(range(num_epochs))
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    axs[0].plot(epochs, train_accs, label="Training")
    axs[0].plot(epochs, eval_accs, label="Evaluation")
    axs[1].plot(epochs, train_losses, label="Training")
    axs[1].plot(epochs, eval_losses, label="Evaluation")
    axs[0].set_xlabel("Epochs")
    axs[1].set_xlabel("Epochs")
    axs[0].set_ylabel("Accuracy")
    axs[1].set_ylabel("Loss")
    plt.legend()