# Multimodal Image Segmentation

This repository contains our proposed image segmentation frameworks applicable to both 2D and 3D segmentation. These include architectures and losses of:

1. **HartleyMHA**

   Ken C. L. Wong, Hongzhi Wang, and Tanveer Syeda-Mahmood, “HartleyMHA: self-attention in frequency domain for resolution-robust and parameter-efficient 3D image segmentation,” in *International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, 2023, pp. 364–373.
    ```
   @inproceedings{Conference:Wong:MICCAI2023:hartleymha,
     title =       {{HartleyMHA}: self-attention in frequency domain for resolution-robust and parameter-efficient {3D} image segmentation},
     author =      {Wong, Ken C. L. and Wang, Hongzhi and Syeda-Mahmood, Tanveer},
     booktitle =   {International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
     pages =       {364--373},
     year =        {2023},
   }
   ```

2. **FNOSeg3D**

   Ken C. L. Wong, Hongzhi Wang, and Tanveer Syeda-Mahmood, “FNOSeg3D: resolution-robust 3D image segmentation with Fourier neural operator,” in *IEEE International Symposium on Biomedical Imaging (ISBI)*, 2023, pp. 1–5.
    ```
   @inproceedings{Conference:Wong:ISBI2023:fnoseg3d,
     title =       {{FNOSeg3D}: resolution-robust {3D} image segmentation with {Fourier} neural operator},
     author =      {Wong, Ken C. L. and Wang, Hongzhi and Syeda-Mahmood, Tanveer},
     booktitle =   {IEEE International Symposium on Biomedical Imaging (ISBI)},
     pages =       {1--5},
     year =        {2023},
   }
   ```

3. **V-Net-DS (V-Net with deep supervision)**

   Ken C. L. Wong, Mehdi Moradi, Hui Tang, and Tanveer Syeda-Mahmood, “3D segmentation with exponential logarithmic loss for highly unbalanced object sizes,” in *International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, 2018, pp. 612–619. [[pdf](https://arxiv.org/pdf/1809.00076.pdf)]
    ```
   @inproceedings{Conference:Wong:MICCAI2018:3d,
     title =       {{3D} segmentation with exponential logarithmic loss for highly unbalanced object sizes},
     author =      {Wong, Ken C. L. and Moradi, Mehdi and Tang, Hui and Syeda-Mahmood, Tanveer},
     booktitle =   {International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
     pages =       {612--619},
     year =        {2018},
   }
   ```

4. **Pearson’s Correlation Coefficient (PCC) loss**

   Ken C. L. Wong and Mehdi Moradi, “3D segmentation with fully trainable Gabor kernels and Pearson’s correlation coefficient,” in *Machine Learning in Medical Imaging*, 2022, pp. 53–61. [[pdf](https://arxiv.org/pdf/2201.03644.pdf)]
   ```
   @inproceedings{Workshop:Wong:MLMI2022:3d,
      title =       {{3D} segmentation with fully trainable {Gabor} kernels and {Pearson's} correlation coefficient},
      author =      {Wong, Ken C. L. and Moradi, Mehdi},
      booktitle =   {Machine Learning in Medical Imaging},
      pages =       {53--61},
      year =        {2022},
    }
    ```
   
## Technical Details

The code is developed with Python 3.6.9 and Keras in TensorFlow 2.6.2, and the "channel-last" format is assumed. If you are only interested in the architectures, the `nets` module is all you need, though the hyperparameters are stored under [experiments/config_files](experiments/config_files) as they are dataset specific. The experimental setups such as data splits and training procedure are in the `experiments` module. The `nets` module is dataset independent, while some functions in the `experiments` module (e.g., dataset partitioning) are written exclusively for BraTS'19.

In experiments, parameters or arguments are provided through a config file using the Python's module `ConfigParser`. The config file is saved to the output directory for future reference. Examples of the config files used in our experiments are provided under [experiments/config_files](experiments/config_files) for reproducibility.

For your convenience, we include an example of our experimental setup in [BraTS2019_example.zip](BraTS2019_example.zip) which contains the necessary folders and config files for testing. To perform experiments on the BraTS'19 dataset, please obtain the dataset from [CBICA](https://ipp.cbica.upenn.edu/). In our example, we assume that the images are downsampled by the nearest neighbor interpolation to 120x120x78 for training, and the original data hierarchy remains unchanged. For inference, the official validation dataset with the original image size of 240x240x155 should be used.


### Setting Up the Virtual Environment

There are multiple Python packages required to run the code. You can install them by the following steps:

1. Create a virtual environment (https://docs.python.org/3/library/venv.html). Note that the default `python` in your system may be Python 2. You can use the following command to ensure Python 3 is used:
   ```
   python3 -m venv /path/to/new/virtual/environment
   ```

2. Upgrade `pip` in the *activated* virtual environment:
   ```
   pip install --upgrade pip
   ```
   It is important to upgrade ```pip``` as the installed version can be outdated and the next step may fail.

3. Install the required Python packages using:
   ```
   pip install tensorflow natsort SimpleITK tensorflow_addons matplotlib pandas pydot
   ```
   > **_Note:_** The Linux (not Python) library `graphviz` is required by the function `tensorflow.keras.utils.plot_model`. If you encounter the corresponding runtime error, you can either install `graphviz` by `sudo apt-get install graphviz` if you have sudo privileges, or set `is_plot_model = False` in the training config file to skip `plot_model`. Furthermore, the code is developed using TensorFlow 2.6.2. Different versions of TensorFlow 2 can be used but are not guaranteed to work.

For more information on troubleshooting, see [Troubleshooting](troubleshooting.md).


### Data Partitioning

The [experiments/data_split](experiments/data_split) folder contains the script and config files for partitioning the BraTS'19 dataset. The program goes through the dataset folders to extract the patient IDs and groups them into training, validation, and testing sets. The resulted lists of file paths are saved as txt files. To run the script, we first modify the `config_partitioning.ini` config file, then use the command line:
```
python partitioning.py /path/to/config_partitioning.ini
```
The split examples used in our experiments are provided under [`split_examples`](experiments/data_split/split_examples). They are also included in [BraTS2019_example.zip](BraTS2019_example.zip).


### Training

To perform training, we first modify the `config_<arch>.ini` file, then run:
```
python run.py /path/to/config_<arch>.ini
```
where `<arch>` stands for an architecture (e.g., fnoseg). The config files of different architectures are only different in the `[model]` section and `output_dir`. Note that in our example, the validation and testing sets are the same. The config files used in our experiments can be found under [experiments/config_files](experiments/config_files). They are also included in [BraTS2019_example.zip](BraTS2019_example.zip).


### Inference

To perform inference, we first modify the `config_inference_<arch>.ini` file, then run:
```
python inference.py /path/to/config_inference_<arch>.ini
```
where `<arch>` stands for an architecture (e.g., fnoseg). The segmentation results can be uploaded to [CBICA](https://ipp.cbica.upenn.edu/) for the official performance validation. The config files used in our experiments can be found under [experiments/config_files](experiments/config_files). They are also included in [BraTS2019_example.zip](BraTS2019_example.zip).


### Results Statistics

We find that in the official validation results, the "enhancing tumor" (ET) region has sensitivity of NaN. We also find that `Hausdorff95_ET` = NaN when `Sensitivity_ET` = 1. These indicate that there may be no positives for ET for some cases. Therefore, when computing the means and variances for the ET region (e.g., `Dice_ET`, `Hausdorff95_ET`), those cases with `Sensitivity_ET` equals to NaN or 1 are ignored.

## Contact Information

Ken C. L. Wong (<clwong@us.ibm.com>)
