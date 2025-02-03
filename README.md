# ROI-VehicleTimer
A c++ based application to detect moving vehicles and time them while inside a certain region of interest.

# Requirements
1. Opencv 4.x
2. C++
3. Docker (Optional)
     
# Compilation command
g++ vehicleDetection.cpp -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_video -lopencv_videoio -lopencv_highgui -lopencv_dnn -lopencv_objdetect -lopencv_imgcodecs

# Run
./a.out
