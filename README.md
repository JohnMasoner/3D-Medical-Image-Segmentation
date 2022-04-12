# 3D-Medical-Image-Segmentation

I prefer to use my other repo [MedicalZoo](https://github.com/JohnMasoner/MedicalZoo) which includes this library along with algorithms for 2D models and multimodal medical image segmentation. Of course this repo is more pure.

## About The Project

Inspired by [Pytorch-Medical-Segmentation](https://github.com/MontaEllis/Pytorch-Medical-Segmentation) of [Ellis](https://github.com/MontaEllis),I wrote this project for 3D medical imaging.I didn't use Torchio as the Dataloader for this project, I just used numpy as the Dataloader,which improved the efficiency of data reading.

## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Requirements

* pytorch1.7
* python>=3.6

### Installation

1. Clone the repo
    ```sh
   git clone https://github.com/JohnMasoner/3D-Medical-Image-Segmentation.git
   ```
2. Installation the requirements
    ```sh
    pip install -r requirement.txt
    ```
3. Custom the configuration
    You can modify the variables in the `hparam.py` to fit your data.
4. Training
    ```sh
    python train.py
    ```

## TODO
- [ ] test
- [ ] more models

For more details or any questions, please feel easy to contact us by email <masoner6429@gmail.com>.


