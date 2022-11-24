import torch
import torch.optim as optim
from torchvision import transforms
from torchvision.models.mobilenet import mobilenet_v2
from torch.optim.lr_scheduler import StepLR
from torch.nn import CrossEntropyLoss
import pandas as pd
from torch.utils.data import DataLoader

import Dataset


def train(model, device, train_loader, optimizer, epoch):
    log_interval = 10
    loss_func = CrossEntropyLoss()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def tst(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    loss_func = CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    torch.cuda.empty_cache()
    batch_size = 200
    learning_rate = 1.0
    reduce_lr_gamma = 0.7
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: {} Epochs: {} Batch size: {}'.format(device, epochs, batch_size))

    kwargs = {'batch_size': batch_size}
    if torch.cuda.is_available():
        kwargs.update({'num_workers': 1, 'pin_memory': True})

    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset1 = Dataset.People('valid.csv', 'valid', transform=transform)
    dataset2 = Dataset.People('train.csv', 'train', transform=transform)
    dataset3 = Dataset.People('sample_submission.csv', 'test', transform=transform)
    print('Length train: {} Length test: {}'.format(len(dataset1), len(dataset2)))

    valid_loader = torch.utils.data.DataLoader(dataset1, **kwargs)
    train_loader = torch.utils.data.DataLoader(dataset2, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset3, **kwargs)
    print('Number of train batches: {} Number of test batches: {}'.format(len(train_loader), len(test_loader)))

    model = mobilenet_v2(weights=True)
    model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=10)
    model.to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)

    scheduler = StepLR(optimizer, step_size=1, gamma=reduce_lr_gamma)
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        tst(model, device, valid_loader)
        scheduler.step()

    torch.save(model.state_dict(), "mnist_cnn.pt")

    predictions = []
    for data, target in test_loader:
        data = data.to(device)
        output = model(data)
        pred = output.softmax(dim=1)
        predictions += list(pred.cpu().detach().numpy()[:, 1])

    fin_res = pd.read_csv('sample_submission.csv')
    fin_res['target_people'] = predictions
    fin_res.to_csv('My.csv', index=False)
    print('Submission saved in: {}'.format('submission.csv'))


if __name__ == '__main__':
    arrOfBatch = [200, 180, 120, 100, 70, 60, 50, 40, 30, 20]
    arrOfEpochs = [3, 5, 10, 20]
    main()
