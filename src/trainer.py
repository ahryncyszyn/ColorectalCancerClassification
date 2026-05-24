from tqdm import tqdm
import torch
import csv
import os

def train_model(model, train_loader, test_loader, device, criterion, optimizer, scheduler, config):
    log_dir = config['log_dir']
    os.makedirs(log_dir, exist_ok = True)
    
    # creating paths for files with metrics csv and best model weights
    csv_path = os.path.join(log_dir, f"{config['model_name']}_metrics.csv")
    ckpt_path = os.path.join(log_dir, f"{config['model_name']}_best.pth")
    
    with open(csv_path, mode = 'w', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_accuracy", "learning_rate"])

    best_acc = 0.0

    # model training
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}", leave=False)
        for images, labels in train_loader:
            images, labels = images.to(device), labels.squeeze().to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        avg_train_loss = total_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.squeeze().to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total
        
        # saving best model weights
        if test_accuracy > best_acc:
            best_acc = test_accuracy
            torch.save(model.state_dict(), ckpt_path)
        
        with open(csv_path, mode = 'a', newline = '') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, test_accuracy, current_lr])

        print(f"Epoch {epoch+1}/{config['epochs']} | Loss: {avg_train_loss:.4f} | Acc: {test_accuracy:.2f}% | LR: {current_lr:.7f}")