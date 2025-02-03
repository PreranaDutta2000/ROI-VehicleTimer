#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <chrono>

using namespace cv;
using namespace dnn;
using namespace std;
using namespace chrono;

double euclideanDistance(Point p1, Point p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

int main() {
    // loading and using Yolo V4 model for vehicle detection
    String model = "yolov4-tiny.weights";
    String config = "yolov4-tiny.cfg";
    Net net = readNetFromDarknet(config, model);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    VideoCapture cap("VID-20250131-WA0003.mp4");
    if (!cap.isOpened()) {
        cout << "Error: Could not open video file!" << endl;
        return -1;
    }

    int frame_width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(CAP_PROP_FPS));
    VideoWriter outputVideo("/app/output/video_output.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(frame_width, frame_height));
	
    if (!outputVideo.isOpened()) {
        cout << "Error: Could not create video file" << endl;
        return -1;
    }
    Mat frame,fFrame;
	int count = 0;
    //selected ROI region
	int x1 = 176, y1 = 163; 
    int x2 = 337, y2 = 343; 
	bool carInsideROI = false;
    steady_clock::time_point startTime;
    int elapsedSeconds = 0;
	ostringstream timeText;


	vector<Point> rect_pts;
    rect_pts.push_back(Point(324, 0));
    rect_pts.push_back(Point(640, 0));
    rect_pts.push_back(Point(640, 360));
    rect_pts.push_back(Point(400, 360));
	vector<vector<Point>> polygons;
    polygons.push_back(rect_pts);


	vector<Point> rect_pts2;
    rect_pts2.push_back(Point(0, 0));
    rect_pts2.push_back(Point(640, 0));
    rect_pts2.push_back(Point(640, 89));
    rect_pts2.push_back(Point(0, 89));
	vector<vector<Point>> polygons2;
    polygons2.push_back(rect_pts2);

	vector<Point> triangle_pts;
    triangle_pts.push_back(Point(0, 0));
    triangle_pts.push_back(Point(268, 0));
    triangle_pts.push_back(Point(0, 180));

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            cout << "End of video" << endl;
            break;
        }
		fFrame = frame.clone();
		fillPoly(frame, polygons, Scalar(0, 255, 0)); 
		fillPoly(frame, polygons2, Scalar(0, 255, 0)); 
		fillPoly(frame, triangle_pts, Scalar(0, 255, 0));
		rectangle(fFrame, Point(x1, y1), Point(x2, y2), Scalar(255, 0, 0), 2);
        //YOLO detection process
        Mat blob;
        blobFromImage(frame, blob, 1/255.0, Size(416, 416), Scalar(0, 0, 0), true, false);
        net.setInput(blob);
        vector<Mat> outs;
        net.forward(outs, net.getUnconnectedOutLayersNames());

        vector<Rect> boxes;
        vector<float> confidences;
        vector<Point> centroids;
        for (auto& out : outs) {
            for (int i = 0; i < out.rows; i++) {
                Mat row = out.row(i);
                float confidence = row.at<float>(4);
                
                if (confidence > 0.4) {
                    int x_center = (int)(row.at<float>(0) * frame.cols);
                    int y_center = (int)(row.at<float>(1) * frame.rows);
                    int width = (int)(row.at<float>(2) * frame.cols);
                    int height = (int)(row.at<float>(3) * frame.rows);
                    boxes.push_back(Rect(x_center - width / 2, y_center - height / 2, width, height));
                    confidences.push_back(confidence);
                    centroids.push_back(Point(x_center, y_center)); 
                }
            }
        }
        vector<int> indices;
        NMSBoxes(boxes, confidences, 0.5, 0.4, indices);

        vector<bool> merged(indices.size(), false);
        vector<Rect> finalBoxes;
        vector<Point> finalCentroids;
        for (size_t i = 0; i < indices.size(); i++) {
            if (merged[i]) continue;

            Rect mergedBox = boxes[indices[i]];
            Point mergedCentroid = centroids[indices[i]];

            for (size_t j = i + 1; j < indices.size(); j++) {
                if (merged[j]) continue;
                if (euclideanDistance(mergedCentroid, centroids[indices[j]]) < 20) {
                    mergedBox = mergedBox | boxes[indices[j]];
                    mergedCentroid = Point(mergedBox.x + mergedBox.width / 2, mergedBox.y + mergedBox.height / 2);
                    merged[j] = true;
					//cout << "Merged Box Area: " << mergedBox.area() << endl;
                }
            }

            finalBoxes.push_back(mergedBox);
		
            finalCentroids.push_back(mergedCentroid);
        }
        for (size_t i = 0; i < finalBoxes.size(); i++) {
            rectangle(fFrame, finalBoxes[i], Scalar(0, 255, 0), 2);
            circle(fFrame, finalCentroids[i], 5, Scalar(0, 0, 255), -1);
            /*putText(fFrame, "(" + to_string(finalCentroids[i].x) + "," + to_string(finalCentroids[i].y) + ")", 
                    Point(finalCentroids[i].x + 5, finalCentroids[i].y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);*/
        }
		for (size_t i = 0; i < finalCentroids.size(); i++) 
		{
			//cout<<"x "<<finalCentroids[i].x<<"y "<<finalCentroids[i].y<<endl;
			if((int)finalCentroids[i].x > 180 && (int)finalCentroids[i].x < 340 && (int)finalCentroids[i].y > 163 && (int)finalCentroids[i].y < 343)
			{
					//start timeer
					if (!carInsideROI) {
                    startTime = steady_clock::now();
                    carInsideROI = true;
                	} 
					else {
                    // Update elapsed time
                    elapsedSeconds = duration_cast<milliseconds>(steady_clock::now() - startTime).count();
					cout<<"Frame No: "<<count++<<" "<<"Time In Milliseconds: "<<elapsedSeconds<<endl;
                	}
					int seconds = elapsedSeconds/1000;
					int milliseconds = elapsedSeconds%1000;
					putText(fFrame, "Time:" + to_string(seconds) + ":" + to_string(milliseconds), 
					Point(finalCentroids[i].x + 10, finalCentroids[i].y - 10), 
					FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 0), 2);   
			}
            else 
			{
                // If car leaves the ROI, reset timer
                carInsideROI = false;
                elapsedSeconds = 0;
            }
			
		}
		outputVideo.write(fFrame);
        //imshow("Vehicle Detection with Selective Merging", fFrame);
		//imwrite("frame/" + to_string(count++) + ".png", fFrame);
        //waitKey(1);
    }

    cap.release();
	outputVideo.release(); 
    destroyAllWindows();
    return 0;
}

