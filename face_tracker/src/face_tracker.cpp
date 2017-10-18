/*
 * Copyright (C) 2017, Lentin Joseph and Qbotics Labs Inc.
 * Email id : qboticslabs@gmail.com
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the names of Stanford University or Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * This code will track the faces using ROS
 */

// ROS headers
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>

// OpenCV headers
#include "opencv2/objdetect.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// Custom centroid header
#include <face_tracker_pkg/centroid.h>

// Windows name
static const std::string RAW_WINDOW      = "raw_image_window";
static const std::string DETECTED_WINDOW = "face_detector";

// Namespaces
using namespace std;
using namespace cv;

// Face detector class
class Face_Detector
{

    // Node handler
    ros::NodeHandle nh;

    // Image transport (message type)
    image_transport::ImageTransport it;

    // Image transport pub. and sub.
    image_transport::Subscriber image_sub;
    image_transport::Publisher image_pub;

    // Centroid message publisher
    ros::Publisher face_centroid_pub;

    // Face centroid message
    face_tracker_pkg::centroid face_centroid;

    // Config details
    string input_image_topic;
    string output_image_topic;
    string haar_file_face;

    int face_tracking;
    int display_original_image;
    int display_tracking_image;
    int center_offset;
    int screenmaxx;


public:

    // Member initialisation for
    // the image_transport instance
    Face_Detector(): it(nh) {

        // Load config.yaml
        try{
            nh.getParam("input_image_topic", input_image_topic);
            nh.getParam("output_image_topic", output_image_topic);
            nh.getParam("haar_file_face", haar_file_face);
            nh.getParam("face_tracking", face_tracking);
            nh.getParam("display_original_image", display_original_image);
            nh.getParam("display_tracking_image", display_tracking_image);
            nh.getParam("center_offset", center_offset);
            nh.getParam("screenmaxx", screenmaxx);

            ROS_INFO("Successfully Loaded tracking parameters");
        }

        catch(int e)
        {
            ROS_WARN("Parameters are not properly loaded from file, please check !");
            return 1;
        }

        // Subscribe to input video feed and publish output video feed
        image_sub = it.subscribe(input_image_topic, 1, &Face_Detector::callback, this);
        image_pub = it.advertise(output_image_topic, 1);
        face_centroid_pub = nh.advertise<face_tracker_pkg::centroid>("/face_centroid", 10);

    }

    ~Face_Detector() {
        if( display_original_image == 1 or display_tracking_image == 1)
            cv::destroyWindow(OPENCV_WINDOW);
    }

    // Image detection on raw image
    void callback(const sensor_msgs::ImageConstPtr& msg) {

        // CvBridge instance for ROS-OPENCV
        // image conversion
        cv_bridge::CvImagePtr cv_ptr;

        // Convert ROS input image to MAT format
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        }

        catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return 1;
        }

        // Load the haar_face classifier
        CascadeClassifier cascade;
        if( !cascade.load( haar_file_face ) )
        {
            cerr << "ERROR: Could not load classifier cascade" << endl;
        }

        if (display_original_image == 1){
            imshow("Original Image", cv_ptr->image);
        }

        // Face detection in action
        detectAndDraw( cv_ptr->image, cascade );

        // Publish image
        image_pub.publish(cv_ptr->toImageMsg());

        waitKey(30);

    }

    void detectAndDraw( Mat& img, CascadeClassifier& cascade) {
        double t = 0;
        double scale = 1;
        vector<Rect> faces, faces2;
        const static Scalar colors[] = {
            Scalar(255,0,0),
            Scalar(255,128,0),
            Scalar(255,255,0),
            Scalar(0,255,0),
            Scalar(0,128,255),
            Scalar(0,255,255),
            Scalar(0,0,255),
            Scalar(255,0,255)
        };

        Mat gray, smallImg;

        cvtColor( img, gray, COLOR_BGR2GRAY );
        double fx = 1 / scale ;
        resize( gray, smallImg, Size(), fx, fx, INTER_LINEAR );
        equalizeHist( smallImg, smallImg );

        t = (double)cvGetTickCount();
        cascade.detectMultiScale( smallImg, faces, 1.1, 15, 0 |CASCADE_SCALE_IMAGE, Size(30, 30) );

        t = (double)cvGetTickCount() - t;

        for ( size_t i = 0; i < faces.size(); i++ )
        {
            Rect r = faces[i];
            Mat smallImgROI;
            vector<Rect> nestedObjects;
            Point center;
            Scalar color = colors[i%8];
            int radius;

            double aspect_ratio = (double)r.width/r.height;
            if( 0.75 < aspect_ratio && aspect_ratio < 1.3 )
            {
                center.x = cvRound((r.x + r.width*0.5)*scale);
                center.y = cvRound((r.y + r.height*0.5)*scale);
                radius = cvRound((r.width + r.height)*0.25*scale);
                circle( img, center, radius, color, 3, 8, 0 );

                face_centroid.x = center.x;
                face_centroid.y = center.y;


                //Publishing centroid of detected face
                face_centroid_pub.publish(face_centroid);

            }
            else
            rectangle( img, cvPoint(cvRound(r.x*scale), cvRound(r.y*scale)),
            cvPoint(cvRound((r.x + r.width-1)*scale), cvRound((r.y + r.height-1)*scale)),
            color, 3, 8, 0);

        }

        //Adding lines and left | right sections

        Point pt1, pt2,pt3,pt4,pt5,pt6;

        //Center line
        pt1.x = screenmaxx / 2;
        pt1.y = 0;

        pt2.x = screenmaxx / 2;
        pt2.y = 480;

        //Left center threshold
        pt3.x = (screenmaxx / 2) - center_offset;
        pt3.y = 0;

        pt4.x = (screenmaxx / 2) - center_offset;
        pt4.y = 480;

        //Right center threshold
        pt5.x = (screenmaxx / 2) + center_offset;
        pt5.y = 0;

        pt6.x = (screenmaxx / 2) + center_offset;
        pt6.y = 480;

        line(img,  pt1,  pt2, Scalar(0, 0, 255),0.2);
        line(img,  pt3,  pt4, Scalar(0, 255, 0),0.2);
        line(img,  pt5,  pt6, Scalar(0, 255, 0),0.2);

        putText(img, "Left", cvPoint(50,240), FONT_HERSHEY_SIMPLEX, 1, cvScalar(255,0,0), 2, CV_AA);
        putText(img, "Center", cvPoint(280,240), FONT_HERSHEY_SIMPLEX, 1, cvScalar(0,0,255), 2, CV_AA);
        putText(img, "Right", cvPoint(480,240), FONT_HERSHEY_SIMPLEX, 1, cvScalar(255,0,0), 2, CV_AA);

        if (display_tracking_image == 1){

            imshow( "Face tracker", img );
        }

    }

};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "Face tracker");
    Face_Detector ic;
    ros::spin();
    return 0;
}
