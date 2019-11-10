# DL Cryo-EM 

Scripts to generate a huge amount of 2D projections with corresponding angles of 3D volumes.

## Installation
Create the conda environment in which the project will be ran.
```
# create environment
$ conda env create -f environment.yml

# activate environment
$ conda activate protein_reconstruction
```

[Optional] Test if some dependencies are installed:
```
# tensorflow check
$ python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

# astra toolbox check
$ python3 -c "import astra;astra.test_CUDA()"
```

## Run

```
usage: generator.py [-h] --input INPUT [--proj-num PROJ_NUM] --ang-coverage
                    ANG_COVERAGE [--ang-shift ANG_SHIFT] [--output OUTPUT]

Generator of 2D projections of 3D Cryo-Em volumes

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT, -mrc INPUT
                        Input file of 3D volume (*.mrc format)
  --proj-num PROJ_NUM, -num PROJ_NUM
                        Number of 2D projections. Default 5000
  --ang-coverage ANG_COVERAGE, -cov ANG_COVERAGE
                        List of max values for each axis. E.g. `0.5,0.5,2.0`
                        means it: x axis angle and y axis angle take values in
                        range [0, 0.5*pi], z axis angles in range [0, 2.0*pi]
  --ang-shift ANG_SHIFT, -shift ANG_SHIFT
                        Start of angular coverage. Default 0
  --output OUTPUT, -mat OUTPUT
                        Name of output file containing projections with angles
                        (with the extension)

```

Main use is:
```
python generator.py -mrc data/bgal.mrc 

python generator.py -mrc data/bgal.mrc -cov 1
```

## Misc information

### Package versions
The following versions of the packages are installed with Astra-toolbox installation.
```
python-3.6.8
cudnn-7.1.3
cudatoolkit-8.0
```