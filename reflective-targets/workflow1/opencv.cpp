// compile: g++ opencv.cpp -o opencv.o `pkg-config --cflags --libs opencv`

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

int main() {
	cv::VideoCapture cap(2);

	/*
	 * Must run "opkg install v4l-utils" on the roborio
	 * 
	 * Also, use -l to list the controls
	 *     - Valid exposures are "5, 10, 20, 39, 78, 156, 312, 625, 1250, 2500, 5000, 10000, 20000"
	 */
	system("v4l2-ctl -d 2 -c exposure_auto=1,exposure_absolute=10");

	cv::Mat frame;
	cv::namedWindow("Plain", cv::WINDOW_AUTOSIZE);

	cv::Mat filtered;
	cv::namedWindow("Filtered", cv::WINDOW_AUTOSIZE);

	cv::Mat result;
	cv::namedWindow("Result", cv::WINDOW_AUTOSIZE);

	while (1) {
		cap >> frame;
		if (frame.empty())
			break;

		imshow("Plain", frame);

		/* Remove noise -- we're only going for orange on the mask. */
		cv::blur(frame, filtered, cv::Size(3, 3));

		/* Use gray. */
		cv::cvtColor(filtered, filtered, cv::COLOR_BGR2GRAY);

		/* Threshold. */
		cv::threshold(filtered, filtered, 100, 255, cv::THRESH_BINARY);

		// /* Use HSV. */
		// cv::cvtColor(filtered, filtered, cv::COLOR_BGR2HSV);

		// /* Try to threshold the tape. */
		// cv::inRange(filtered, cv::Scalar(0, 0, 40), cv::Scalar(255, 255, 255), filtered);

		/* Get rid of spots. */
		// cv::erode(filtered, filtered, cv::Mat(), cv::Point(-1, -1), 2);

		/* Find contours. */
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(filtered, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

		/* Remove small contours. */
		for (int i = 0; i < contours.size(); i++) {
			if (cv::contourArea(contours[i]) <= 1000) {
				contours.erase(contours.begin() + i);
				i--;
			}
		}

		/* Find rectangles */
		std::vector<std::vector<cv::Point>> polygons;
		polygons.resize(contours.size());

		for (int i = 0; i < contours.size(); i++) {
			cv::approxPolyDP(contours[i], polygons[i], 10, true);

			/* Maybe implement a little error detection. */
			// if (polygons[i].size() > 5 || polygons[i].size() < 4) {
				
			// }
		}

		imshow("Filtered", filtered);

		result = frame;

		for (int i = 0; i < contours.size(); i++) {
			cv::drawContours(result, contours, i, cv::Scalar(0, 0, 255));

			cv::line(result, polygons[i][0], polygons[i][polygons[i].size() - 1], cv::Scalar(0, 255, 0));
			for (int x = 1; x < polygons[i].size(); x++) {
				cv::line(result, polygons[i][x], polygons[i][x - 1], cv::Scalar(0, 255, 0));
			}
		}

		imshow("Result", result);
		
		if (char(cv::waitKey(10)) == 27)
			break;
	}
}
