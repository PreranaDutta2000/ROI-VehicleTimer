# Use an official Ubuntu image
FROM ubuntu:latest

# Install dependencies and OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
    g++ \
    ffmpeg \
    libopencv-core-dev \
    libopencv-imgproc-dev \
    libopencv-video-dev \
    libopencv-dnn-dev \
    libopencv-imgcodecs-dev \
    libopencv-videoio-dev

# Set the working directory
WORKDIR /app

# Copy source code, video file, and YOLO model files into the container
COPY vehicleDetection.cpp /app/
COPY VID-20250131-WA0003.mp4 /app/
COPY yolov4-tiny.cfg /app/
COPY yolov4-tiny.weights /app/

# Compile the C++ code
RUN g++ vehicleDetection.cpp -o vehicleDetection -I/usr/include/opencv4 -L/usr/lib/x86_64-linux-gnu -lopencv_core -lopencv_imgproc -lopencv_video -lopencv_videoio -lopencv_highgui -lopencv_dnn -lopencv_imgcodecs -lopencv_objdetect

# Set the default command to run the program
CMD ["./vehicleDetection"]

