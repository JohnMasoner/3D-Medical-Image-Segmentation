class hparams:
    checkpoints = 'checkpoints'
    log = "logs"
    name = 'test'
    logging_step = 100
    validation_interval =  2000
    # crop_or_pad_size = 512,512,32   # W,H,D

    # optimizer parameters
    lr = 0.001
    step_size = 20
    gamma = 0.1

    #GPU parameters
    devicess = [0]

    # training parameters
    batch_size = 1
    num_workers = 1 # number of workers
    num_epochs = 60 # number of epochs