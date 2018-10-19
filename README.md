Build and install tensorflow cc
Install Boost

Build gauss?
Install MKL? (make optional)

```
git clone --recursive git@github.com:zero-impact/AutoDef.git
cd AutoDef
./installDependencies.sh

# Build GAUSS
cd extern/GAUSS
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
cd ../../../

# Build Cubacode
cd src/Cubacode
mkdir build
cd build
cmake ..
make
cd ../../../

# Build Main app
cd src/AutoDefRuntime
mkdir build
cd build
cmake ..
make

```
