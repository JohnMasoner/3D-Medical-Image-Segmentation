class hparams:
    checkpoints = 'checkpoints'
    log = "logs"
    name = 'test'
    crop_or_pad_size = 512,512,32   # W,H,D

    # optimizer parameters
    lr = 0.001
    step_size = 20
    gamma = 0.1

    #GPU parameters
    devicess = [0]

    # training parameters
    batch_size = 8
    num_workers = 1 # number of workers
    num_epochs = 60 # number of epochs