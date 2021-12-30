from pydicom.sequence import validate_dataset
from dataset.dataloader import MedDataSets3D

from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
from hparam import hparams as hp

from models.unet3d import UNet3D

from utils import  metrics
import os
import numpy as np
# from collections import OrderedDict
from utils.logger import MyWriter

def train(model, train_loder, optimizer, loss_func, n_labels, alpha):
    pass

def main(resume=False):
    checkpoint_dir = "{}/{}".format(hp.checkpoints, hp.name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    os.makedirs("{}/{}".format(hp.log, hp.name), exist_ok=True)
    writer = MyWriter("{}/{}".format(hp.log, hp.name))

    # load model
    model = UNet3D().cuda()
    model = torch.nn.DataParallel(model, device_ids=hp.devicess)

    # dice loss
    criterion = metrics.DiceLoss()
    
    # init the optimizer
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, nesterov=True)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)

    # decay LR
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hp.step_size, gamma=hp.gamma)

    # init traing parameters
    best_loss = 999
    start_epoch = 0

    # check / load checkpoint
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)

            start_epoch = checkpoint["epoch"]

            best_loss = checkpoint["best_loss"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(resume))
    
    # load data
    train_dataset = MedDataSets3D('E:/Process_Data', length = (0,-25))
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size = hp.batch_size, num_workers=hp.num_workers, shuffle=True)
    validate_dataset = MedDataSets3D('E:/Process_Data', length = (-25,None))
    validate_dl = torch.utils.data.DataLoader(train_dataset, batch_size = hp.batch_size, num_workers=hp.num_workers, shuffle=True)

    step = 0
    for epoch in range(start_epoch, hp.num_epochs):
        print("Epoch {}/{}".format(epoch, hp.num_epochs - 1))
        print("-" * 10)

        # step the learning rate scheduler
        lr_scheduler.step()

        # load eval functions
        # instantiate teh metrics 
        train_acc = metrics.MetricTracker()
        train_loss = metrics.MetricTracker()

        # iterate all data
        loader = tqdm(train_dl, desc="training")
        for idx, data in enumerate(loader):
            # get the inputs and wrap in Variable
            inputs = data["image"].cuda()
            labels = data["label"].cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # prob_map = model(inputs) # last activation was a sigmoid
            # outputs = (prob_map > 0.3).float()
            outputs = model(inputs)
            # outputs = torch.nn.functional.sigmoid(outputs)

            loss = criterion(outputs, labels)

            # backward
            loss.backward()
            optimizer.step()

            train_acc.update(metrics.dice_coeff(outputs, labels), outputs.size(0))
            train_loss.update(loss.data.item(), outputs.size(0))

            # tensorboard logging
            if step % hp.logging_step == 0:
                writer.log_training(train_loss.avg, train_acc.avg, step)
                loader.set_description(
                    "Training Loss: {:.4f} Acc: {:.4f}".format(
                        train_loss.avg, train_acc.avg
                    )
                )

            # validation
            if step % hp.validation_interval == 0:
                valid_metrics = validation(
                    validate_dl, model, criterion, writer, step
                )
                save_path = os.path.join(
                    checkpoint_dir, "%s_checkpoint_%04d.pt" % (hp.name, step)
                )
                # store best loss and save a model checkpoint
                best_loss = min(valid_metrics["valid_loss"], best_loss)
                torch.save(
                    {
                        "step": step,
                        "epoch": epoch,
                        "arch": "ResUnet",
                        "state_dict": model.state_dict(),
                        "best_loss": best_loss,
                        "optimizer": optimizer.state_dict(),
                    },
                    save_path,
                )
                print("Saved checkpoint to: %s" % save_path)

            step += 1

def validation(valid_loader, model, criterion, logger, step):

    # logging accuracy and loss
    valid_acc = metrics.MetricTracker()
    valid_loss = metrics.MetricTracker()

    # switch to evaluate mode
    model.eval()

    # Iterate over data.
    for idx, data in enumerate(tqdm(valid_loader, desc="validation")):

        # get the inputs and wrap in Variable
        inputs = data["sat_img"].cuda()
        labels = data["map_img"].cuda()

        # forward
        # prob_map = model(inputs) # last activation was a sigmoid
        # outputs = (prob_map > 0.3).float()
        outputs = model(inputs)
        # outputs = torch.nn.functional.sigmoid(outputs)

        loss = criterion(outputs, labels)

        valid_acc.update(metrics.dice_coeff(outputs, labels), outputs.size(0))
        valid_loss.update(loss.data.item(), outputs.size(0))
        if idx == 0:
            logger.log_images(inputs.cpu(), labels.cpu(), outputs.cpu(), step)
    logger.log_validation(valid_loss.avg, valid_acc.avg, step)

    print("Validation Loss: {:.4f} Acc: {:.4f}".format(valid_loss.avg, valid_acc.avg))
    model.train()
    return {"valid_loss": valid_loss.avg, "valid_acc": valid_acc.avg}

if __name__ == "__main__":
    main()