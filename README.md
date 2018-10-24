Build and install tensorflow cc
Install Boost

Build gauss?
Install MKL? (make optional)

```
$ git clone --recursive git@github.com:zero-impact/AutoDef.git
$ cd AutoDef
$ sudo ./installDependencies.sh # When Anaconda asks if you would like to update .bashrc answer 'yes'
$ ./build.sh
```

Uncomment #define EIGEN_USE_MKL_ALL in src/AutoDefRuntime/src/main.cpp if you want to enable MKL support.