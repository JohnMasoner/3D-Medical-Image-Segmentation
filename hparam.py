class hparams:
    filedir = 'E:/Process_Data'
    checkpoints = 'checkpoints'
    log = "logs"
    name = '1-5test'
    logging_step = 10
    validation_interval =  20
    
    # training parameters
    batch_size = 8
    num_workers = 2 # number of workers
    num_epochs = 60 # number of epochs

    # optimizer parameters
    lr = 0.001
    step_size = batch_size * 20
    rand_crop_size = [256,256,64] # W, H, D
    gamma = 0.1

    #GPU parameters
    devicess = [0,1]