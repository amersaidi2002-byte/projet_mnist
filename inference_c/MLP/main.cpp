/**
 * Pi 5 Camera Stream Passthrough
 * Reçoit flux H264, renvoie vers client
 */
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <csignal>
#include <atomic>

#include "neural_network.h"
#include "model_weights.h"





std::atomic<bool> running(true);
void sigHandler(int) { running = false; }

int main(int argc, char** argv) {

    float l1[FC1_SIZE];
    float l2[FC2_SIZE];
    float output[OUTPUT_SIZE];
    int pred=0;

    DenseLayer fc1 = {INPUT_SIZE, FC1_SIZE, fc1_weights, fc1_bias};
    DenseLayer fc2 = {FC1_SIZE, FC2_SIZE, fc2_weights, fc2_bias};
    DenseLayer fc3 = {FC2_SIZE, OUTPUT_SIZE, fc3_weights, fc3_bias};
    signal(SIGINT, sigHandler);
    signal(SIGTERM, sigHandler);

    int inPort = argc > 1 ? std::stoi(argv[1]) : 5000;
    int outPort = argc > 2 ? std::stoi(argv[2]) : 8554;
    int w = argc > 3 ? std::stoi(argv[3]) : 1280;
    int h = argc > 4 ? std::stoi(argv[4]) : 720;

    std::cout << "=== Pi5 Camera ===" << std::endl;
    std::cout << "In:" << inPort << " Out:" << outPort << " " << w << "x" << h << std::endl;

    std::string capPipe = 
        "tcpclientsrc host=127.0.0.1 port=" + std::to_string(inPort) + " ! "
        "h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=1 sync=0";

    std::string outPipe = 
        "appsrc ! videoconvert ! video/x-raw,format=I420 ! "
        "x264enc tune=zerolatency speed-preset=ultrafast bitrate=4000 key-int-max=15 ! "
        "video/x-h264,profile=baseline ! h264parse config-interval=1 ! "
        "mpegtsmux ! tcpserversink host=0.0.0.0 port=" + std::to_string(outPort);

    cv::VideoCapture cap(capPipe, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) { std::cerr << "Erreur: input" << std::endl; return 1; }

    cv::VideoWriter writer(outPipe, cv::CAP_GSTREAMER, 0, 60, cv::Size(w, h), true);
    if (!writer.isOpened()) { std::cerr << "Erreur: output" << std::endl; return 1; }

    cv::Mat frame;
    int count = 0;
    std::string pjh="la prediction est ";
    // Type explicite du time_point
    std::chrono::steady_clock::time_point t0 =
        std::chrono::steady_clock::now();

    while (running && cap.read(frame)) {
        if (frame.empty()) continue;

        // === TRAITEMENT OPENCV ICI ===
        cv::Mat gray;
        cv::cvtColor(frame,gray,cv::COLOR_BGR2GRAY);

        cv::GaussianBlur(gray,gray,cv::Size(3,3),0);

        cv::Mat bin;
        cv::threshold(gray,bin,125,255,cv::THRESH_BINARY_INV);

        // extraire les contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(bin.clone(),contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);//CHAIN_APPROX_SIMPLE pour l'optimisation des points

        //prendre le contour avec le plus grand air = size
        int best = -1;
        double maxarea = 0.0;
        double area_image = (double)bin.cols * (double)bin.rows;

        for (int i = 0; i < (int)contours.size(); i++) {
            double a = cv::contourArea(contours[i]);
            if (a > maxarea && a <= 0.9 * area_image) {
                maxarea = a;
                best = i;
            }
        }

        // le bounding box

        cv::Rect box=cv::boundingRect(contours[best]);
        cv::Mat roi=bin(box);

        // choix de la taille du carre
        int s=cv::max(roi.cols, roi.rows);

        // créer l'image carre qui contiendra l'image centré
        cv::Mat square=cv::Mat::zeros(s,s,CV_8UC1);
        // les formules de recentrage
        int x=(s-roi.cols)/2;
        int y=(s-roi.rows)/2;

        roi.copyTo(square(cv::Rect(x,y,roi.cols,roi.rows)));

        cv::rectangle(frame,box,cv::Scalar(255,0,0),2); // (255,0,0)=(R,G,B) =rouge pour afficher le bounding box
        
        cv::Mat img28;
        cv::resize(square,img28,cv::Size(28,28),0,0,cv::INTER_AREA);

        // Normalisation
        cv::Mat x_norm;
        img28.convertTo(x_norm, CV_32F, 1.0/255.0);
        

        // x (MLP/CNN)
        cv::Mat x_flat=x_norm.reshape(1,1);
            const float* x_ptr=x_flat.ptr<float>(0);
            dense_forward(&fc1, x_ptr, l1);
            relu(l1, FC1_SIZE);

            dense_forward(&fc2, l1, l2);
            relu(l2, FC2_SIZE);

            dense_forward(&fc3, l2, output);

            pred = argmax(output, OUTPUT_SIZE);

        
        cv::putText(frame,pjh+std::to_string(pred),cv::Point(box.x,box.y),cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar(255,0,0),3);

        writer.write(frame);
        count++;

        std::chrono::steady_clock::time_point now =
            std::chrono::steady_clock::now();

        std::chrono::duration<double> dt = now - t0;

        if (dt.count() >= 1.0) {
            std::cout << "FPS: "
                      << static_cast<int>(count / dt.count())
                      << std::endl;

            count = 0;
            t0 = now;
        }
    }

    return 0;
}

