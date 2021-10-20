def train_loop(data_loader, model, optimizer, criterion, epochs):

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            input, labels = data

            optimizer.zero_grad()
            outputs = model.forward(input)
            loss = criterion(outputs, labels)
            loss.backwards()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 0:  # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
