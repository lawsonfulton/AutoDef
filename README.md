This has been tested to work on a clean install of Ubuntu 18.04.
If your system is not a clean install, there may be dependency conflicts.

```
$ git clone --recursive git@github.com:zero-impact/AutoDef.git
$ cd AutoDef
$ sudo ./installDependencies.sh # When Anaconda asks if you would like to update .bashrc answer 'yes'
$ ./build.sh
```

Uncomment #define EIGEN_USE_MKL_ALL in src/AutoDefRuntime/src/main.cpp if MKL is installed you want to enable MKL support.


TODO
- See if tensorflow_cc built the python module
- Need a way of setting kers float64 in keras.json
- Make sure that I can run tensorflow from tensorflowcc
