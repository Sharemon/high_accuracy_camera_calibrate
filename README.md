# high_accuracy_camera_calibrate
camera calibration project using iterative refinement of control points to improve accuracy of calibration

There are 4 modes provided(refer to ./app/demo.cpp):
* generate_calibration_board: used to generate a calibration board on the monitor
* simulate_calibration_image: used to simulate images of calibration boards with different posed 
* single_camera_calibrate: used to calibrate a single camera
* stereo_camera_calibrate: used to calibrate a stereo cameras

## Dependencies

* OpenCV
* Eigen

## Compilation

```shell
mkdir build
cd build
cmake ..
make -j
```

## Run

```shell
cd buid
cp ../run.sh ./
./run.sh
```

## References

> Datta, A., Kim, J.S. and Kanade, T., 2009, September. Accurate camera calibration using iterative refinement of control points. In 2009 IEEE 12th International Conference on Computer Vision Workshops, ICCV Workshops (pp. 1201-1208). IEEE.