# Build GAUSS
cd extern/GAUSS
bash installGAUSS_Ubuntu.sh
cd ../../

# Build Libigl Python bindings
cd extern/libigl/python
mkdir build
cd build; cmake ..; make -j8;
cd ../../../../

# Build Cubacode
cd src/cubacode
mkdir build
cd build
cmake ..
make -j8
cd ../../../

# Build Main app
cd src/AutoDefRuntime
mkdir build
cd build
cmake ..
make -j8
cd ../../../