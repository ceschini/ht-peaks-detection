# Hough Transform Peaks Detection

<!-- introduction presenting the project here -->

The scripts are contained in jupyter notebooks, and can be accessed trhough google-colab, kaggle or any other notebook kernel. Following are the instructions to install an run locally.

## Installation

Make sure you run the submodules command in order to get the dataset from it's repo. Even though it can be installed in the global python environment, it is prefferable to install it in a virtual environment as follows:

1. Clone the repository with the submodule option:

    ```bash
    git clone --recurse-submodules https://github.com/ceschini/ht-peaks-detection.git
    ```

2. Install ```virtualenv``` if you don't already have it.

    ```pip install virtualenv```

3. Create a new virtual environment via the ```virtualenv``` package.

    ```virtualenv .venv```

4. Install python packages requirements.

    ```pip install -r requirements.txt```

## Usage

You can explore the scripts in any online notebook kernel available, or you can run it locally by using an IDE with notebook support, like ```vs-code```. You can also run the [Jupyter Notebook](https://docs.jupyter.org/en/latest/) interface by issuing the following command on a terminal:

1. ```cd ht-peaks-detection/```
2. ```jupyter notebook```

***

Heloísa and Lucas, March 20, 2023.
