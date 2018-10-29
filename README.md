This has been tested to work on a clean install of Ubuntu 18.04.
If your system is not a clean install, there may be dependency conflicts.

# Build Instructions
```
$ git clone --recursive git@github.com:zero-impact/AutoDef.git
$ cd AutoDef
$ sudo ./installDependencies.sh # When Anaconda asks if you would like to update .bashrc answer 'yes'
$ ./build.sh
```

WARNING: installDependencies.sh will overwrite your `~/.keras/keras.json` file if it is present. The default float type will be set to 64 bits from the default 32.


### MKL Support
Uncomment #define EIGEN_USE_MKL_ALL in src/AutoDefRuntime/src/main.cpp if MKL is installed you want to enable MKL support.


# Usage Instructions