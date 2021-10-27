import torch
def train_loop(data_loader, model, optimizer, criterion, epochs, device):
    print(device)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("Epoch: {} Loss {}".format(epoch, running_loss/len(data_loader)))

