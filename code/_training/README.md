# Instructions to train the NN

## Directory Overview `root/code/_training`

* `../current_test_data/` - Constant path directory to have our script pointing to.
* `../examples/` - Location for different examples....`submoudles` should be used if a repository is avalable.
* `../imagenet/` - Location for different datasets.
* `../submodules/` - Location for git submodules; to add one: `git submoudle add <url>`.
* `../` - Location for script files

### Dependencies

* For worry free compatability with `rbaas293`'s tensorflow-gpu setup:

```powershell
# Install dependent packages
pip3 install tensorflow==1.13.0rc0 pillow virtualenv jupyter matplotlib numpy
# If you have a supported graphics card:
pip3 install tensorflow-gpu==1.13.0rc0
```

* Non-CLI installable for GPU support:

1. [Download & Install CUDA Toolkit 10.0](https://developer.nvidia.com/cuda-10.0-download-archive)
2. [Download & "Install" cuDNN==>= 7.4.1 [SADLY YOU HAVE TO SIGN UP]](https://developer.nvidia.com/cudnn)
3. Make sure/add the above programs to your PATH (open admin command prompt):

```cmd
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin;%PATH%
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\extras\CUPTI\libx64;%PATH%
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\include;%PATH%
SET PATH=C:\tools\cuda\bin;%PATH%
```

*IF THE ABOVE INSTRUCTIONS ARE OUT OF DATE, SEE [NVIDIA WINDOWS GUIDE](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)

### Get up and running with our NN

1. Open PowerShell and navigate to the root of this repository.
2. To make sure you have permision to run the following script (ignore if not on windows): `Set-ExecutionPolicy -ExecutionPolicy Bypass -scope CurrentUser` 
3. Run Script: `.\start_jupyter-notebook.ps1`
4. A localhost browser window will pop up. the root directory of the repository is shown.
5. Navigate to `root/code/_training`
6. Open `do_the_thing.ipynb` and run thru the NN interactivly.
7. Enjoy and Have FUN!!
