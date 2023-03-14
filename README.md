# Image Superresolution using Pytorch


### Requirements
Install requirements from the txt file as
```shell
$ pip install -r requirements.txt
```


### Dataset
We use DIV2K dataset for training and Set5, Set14, B100, and Urban100 dataset for the benchmark test. Here are the following steps to prepare datasets.

1. Download [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K) and unzip on `dataset` directory as below:
  ```
  dataset
  └── DIV2K
      ├── DIV2K_train_HR
      ├── DIV2K_train_LR_bicubic
      ├── DIV2K_valid_HR
      └── DIV2K_valid_LR_bicubic
  ```
2. To accelerate training, we first convert training images to h5 format as follow (h5py module has to be installed).
```shell
$ python div2h5.py
```

3. Add the benchmark test dataset in the dataset directory as DIV2K.

### Training
To train the model, update the scale in train.py and run       
 ```shell
$ python train.py
```

### Testing
For testing, run the following code after training the model.
```shell
$ python sample.py
```

### Results
We achieved high performance for image SR, with PSNR value of 36.02 for scale factor of 2 and 30.04 for scale factor of 4. The input and outpur from our model can be seen below.

<details>
<summary>Input Image</summary>
<p align="center">
  <img src="TestImages/1.jpg">
</p>
</details>
<details>
<summary>Output Image</summary>
<p align="center">
  <img src="assets/output1.jpg">
</p>
</details>







 
