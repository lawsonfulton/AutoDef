This is the code for the paper [Latent-space Dynamics for Reduced Deformable Simulation](https://www.lawsonfulton.com/publication/latent-space-dynamics/)

# Notes
This has been tested to work on a clean install of Ubuntu 18.04.
If your system is not a clean install, there may be dependency conflicts.

# Build Instructions
```
git clone --recursive https://github.com/zero-impact/AutoDef.git
cd AutoDef
sudo ./installDependencies.sh # This could take a while (building tensorflow from source)

# Now build the main project
cd src/AutoDefRuntime
mkdir build && cd build
cmake .. && make -j8
```

WARNING: installDependencies.sh will overwrite your `~/.keras/keras.json` file if it is present. The default float type will be set to 64 bi+ts from the default 32.

The build process may take a long time (30-60 minutes) depending on your hardware. This is due to the fact that we use a modified version of Tensorflow and must build from source.


### MKL Support
Uncomment #define EIGEN_USE_MKL_ALL in src/AutoDefRuntime/src/main.cpp if MKL is installed you want to enable MKL support.


# Usage Instructions

### Training a Model
To interactively generate training data and subsequently train a reduced model, use the script `scripts/unified_gen_and_train.py <config>.json <model_output_dir>`

Example:
```
./scripts/unified_gen_and_train.py configs/X.json models/X
```

Instructions:
1. When the opengl viewer opens, press 'a' to start recording the low resolution simulation.
2. Click and drag on the model to interact. Click and drag on the background to change viewing angle.
3. Press 'q' to end recording (Do not close the window manually!). A new viewer will open and show the full-resolution model being played-back.
4. Wait for simulation to finish. Training will begin and finish automatically.

Training parameters can be changed by editing `<config>.json`.

### Running a Model
After training has completed, the model can be run using `src/AutoDefRuntime/build/bin/AutoDefRuntime <model_dir>`.

Continuing from the example above
```
./src/AutoDefRuntime/build/bin/AutoDefRuntime models/X_example
```

Instructions:
1. Press 'a' to start the simulation.
2. Click and drag the model to interact.

The simulation parameters (such as the type of reduced space) can be changed by editing `<model_dir>/sim_config.json`

# License
MIT License

Copyright (c) 2019 Lawson Fulton

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
