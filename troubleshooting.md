# Troubleshooting

The following solutions are based on our experience on Ubuntu 18.04, Python 3.6.9, and Keras in TensorFlow 2.6.2.

## TensorFlow

The runtime error is common with TensorFlow, which is usually caused by missing or mismatched CUDA library or cuDNN library. Note that `sudo` privileges are usually required when handling the library issues. The TensorFlow dependencies on CUDA and cuDNN can be found at:\
https://www.tensorflow.org/install/source#gpu


### Installing CUDA Toolkit (not the driver)

A specific version of CUDA Toolkit may be required. To install the CUDA Toolkit without messing up the existing driver, I use the runfile installation:\
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#runfile-installation \
As we *do not* install the driver, step 1-4 in Section 8.2 are unnecessary. The runfile can be found online, for example, the one for CUDA 11.2 can be found at:\
https://developer.nvidia.com/cuda-11-2-0-download-archive \
After running the runfile and getting inside the user interface:
1. Choose `Continue` for the “driver found” warning.
2. In the “CUDA Installer” section, uncheck everything except `CUDA Toolkit`.
3. Also in the “CUDA Installer” section, select `Options` &rarr; `Toolkit Options` &rarr; uncheck everything &rarr; `Done` &rarr; `Done` &rarr; `Install`.

Wait until the installation is finished with the summary shown.

### Installing cuDNN

Multiple runtime issues can be solved by installing cuDNN. The official installation guide is provided at:\
https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html \
The different versions of cuDNN installers are available at:\
https://developer.nvidia.com/rdp/cudnn-archive
> **_Note:_** The installer used must be consistent with the TensorFlow dependencies. For example, if the TensorFlow version requires CUDA 11, you should use the cuDNN installer for CUDA 11 but not for CUDA 12.

In this [Stack Overflow post](https://stackoverflow.com/questions/66977227/could-not-load-dynamic-library-libcudnn-so-8-when-running-tensorflow-on-ubun), a Linux user reported that the installation can be achieved by:
```
sudo apt-get install libcudnn8
```

### Runtime error: "Could not load dynamic library 'libcudnn.so.8'"

This may happen when importing TensorFlow. First we can check if the file exists by using:
```
find /usr -name "libcudnn.so.8"
```
If it does not exist, cuDNN should be installed. If you know that the file exists in an uncommon location, you can modify (or create) `~/.bash_profile` with the line:
```
export LD_LIBRARY_PATH=/directory/containing/the/file:$LD_LIBRARY_PATH
```
so that the library file can be found.


### Runtime error: "Could not load dynamic library 'libcudart.so.11.0'"

This may happen when importing TensorFlow. First we can check if the file exists by using:
```
find /usr -name "libcudart.so.11.0"
```
If it does not exist, CUDA Toolkit should be installed. If you know that the file exists in an uncommon location, you can modify (or create) `~/.bash_profile` with the line:
```
export LD_LIBRARY_PATH=/directory/containing/the/file:$LD_LIBRARY_PATH
```
so that the library file can be found.


### Runtime error: "Unknown: Failed to get convolution algorithm"

This may happen when training starts. This may be caused by an incompatible cuDNN version, for example, cuDNN for CUDA 12 is installed while TensorFlow requires cuDNN for CUDA 11. You can find the required version using the following Python code under the activated virtual environment:
```python
import tensorflow as tf
print(tf.sysconfig.get_build_info())
```


### The program is running but no GPU is used

This may happen after training starts. This may be caused by a missing or incompatible cuDNN library. First we can check if TensorFlow can find the GPUs using the following Python code under the activated virtual environment:
```python
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```
If TensorFlow cannot not find the GPUs, a compatible cuDNN needs to be installed.


## GraphViz

You may encounter an error message ends with:
>ImportError: ('You must install pydot (\`pip install pydot`) and install graphviz (see instructions at https://graphviz.gitlab.io/download/) ', 'for plot_model/model_to_dot to work.')

The Linux (not Python) library `graphviz` is required by the function `tensorflow.keras.utils.plot_model`. If you encounter the corresponding runtime error, you can install `graphviz` if you have `sudo` privileges:
```
sudo apt-get install graphviz
```
Or you can set `is_plot_model = False` in the training config file to skip `plot_model`.