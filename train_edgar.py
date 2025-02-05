import datetime
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import unet
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import utils.dataset as dataset

# define how the model is supposed to learn
def train(dataloader, model, loss_fn, optimizer, tb_writer):
    size = len(dataloader.dataset)
    model.cuda()
    model.train()

    # convert data to the right device type
    for batch, (start_image, start_gaze, target_image, target_gaze) in enumerate(dataloader):
        # send everything to gpu
        start_image, target_image, target_gaze = start_image.cuda(), target_image.cuda(), target_gaze.cuda()

        prediction = model(start_image, target_gaze)
        loss = loss_fn(prediction, target_image)
    
        # Use backpropagation to update weights
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
        # show progress, every 100 batches
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1)
            print(f"loss: {loss:>7f} [{current}/{size}]")
            tb_writer.add_scalar('Loss/Train', loss, batch)

            grid = torchvision.utils.make_grid(start_image)
            writer.add_image("Start",grid)
            grid = torchvision.utils.make_grid(target_image)
            writer.add_image("Target",grid)
            grid = torchvision.utils.make_grid(prediction)
            writer.add_image("Prediction",grid)
            writer.add_graph(model, (start_image, target_gaze))

def test(dataloader, model, loss_fn):
    model.eval()
    test_loss = 0
    num_batches = 5
    with torch.no_grad():
        for batch, (start_image, start_gaze, target_image, target_gaze) in enumerate(dataloader):
            # send everything to gpu
            start_image, target_image, target_gaze = start_image.cuda(), target_image.cuda(), target_gaze.cuda()
            pred = model(start_image, target_gaze)
            test_loss += loss_fn(pred, target_image).item()
            if batch == num_batches:
                # return average loss
                return (test_loss / num_batches)

if __name__ == "__main__":

    writer = SummaryWriter()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load and normalize dataset
    training_data = dataset.GazeImages_forUNet(normalize=False)

    # split into training and validation set 
    generator = torch.Generator().manual_seed(42)
    train_set, validation_set = random_split(training_data, [.8, .2], generator=generator)

    edgar = unet.unet()
    edgar.load_state_dict(torch.load("29-01-25_L1_continue_pt-2/29-01-25_15-16-Unet-390-epochs.pth"))

    # Mean Squared error, because it is a regression task, not classification
    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(edgar.parameters(), lr=1e-3)

    ## Do the Training
    epochs = 200
    batch_size = 20
    train_dataloader = DataLoader(train_set, batch_size=batch_size, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(validation_set, batch_size=batch_size)
    # save time of training
    date = datetime.datetime.today().strftime("%d-%m-%y_%H-%M")
    for epoch in tqdm(range(epochs), "Training"):
        print(f"Epoch {epoch+1}\n------------------------------------------")
        train(train_dataloader, edgar, loss_fn, optimizer, writer)
        avg_loss = test(test_dataloader, edgar, loss_fn)
        writer.add_scalar("Average loss", avg_loss, epoch + 1)
        print("Done!")

        if epoch%10 == 0:
            torch.save(edgar.state_dict(), f"{date}-Unet-{epoch}-epochs.pth")
            print("saved progress")


