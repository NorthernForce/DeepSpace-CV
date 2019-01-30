// compile: g++ opencv.cpp -o opencv.o `pkg-config --cflags --libs opencv`

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

int main() {
	cv::VideoCapture cap(2);

	cv::Mat frame;
	// cv::namedWindow("Plain", cv::WINDOW_AUTOSIZE);

	cv::Mat filtered;
	cv::namedWindow("Filtered", cv::WINDOW_AUTOSIZE);

	cv::Mat result;
	cv::namedWindow("Result", cv::WINDOW_AUTOSIZE);

	while (1) {
		cap >> frame;
		if (frame.empty())
			break;

		// imshow("Plain", frame);

		/* Remove noise -- we're only going for orange on the filtered. */
		cv::blur(frame, filtered, cv::Size(10, 10));

		/* Use HSV. */
		cv::cvtColor(filtered, filtered, cv::COLOR_BGR2HSV);

		/* Try to threshold the general ball color. */
		/* TODO: Find good values. */
		cv::inRange(filtered, cv::Scalar(2, 55, 150), cv::Scalar(15, 255, 255), filtered);

		/* Get rid of spots. */
		cv::erode(filtered, filtered, cv::Mat());
		cv::dilate(filtered, filtered, cv::Mat());

		imshow("Filtered", filtered);

		/* Find contours. */
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(filtered, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

		/* Find contour moments. */
		std::vector<cv::Moments> moments;
		moments.resize(contours.size());

		/* Find moments. */
		for (int i = 0; i < contours.size(); i++) {
			moments[i] = cv::moments(contours[i], false);
		}

		/* Remove small contours. */
		for (int i = 0; i < contours.size(); i++) {
			/* TODO: Find good area size. */
			if (moments[i].m00 <= 1000) {
				contours.erase(contours.begin() + i);
				moments.erase(moments.begin() + i);
				i--;
			}
		}

		// /* Find polygons. Lotta sides = circle? */
		// std::vector<std::vector<cv::Point>> polygons;
		// polygons.resize(contours.size());

		// for (int i = 0; i < contours.size(); i++) {
		// 	cv::approxPolyDP(contours[i], polygons[i], 10, true);
		// }

		result = frame;

		/* Draw results. */
		for (int i = 0; i < contours.size(); i++) {
			cv::drawContours(result, contours, i, cv::Scalar(0, 0, 255));
			cv::circle(result, cv::Point(moments[i].m10/moments[i].m00, 
										moments[i].m01/moments[i].m00), 1,
										cv::Scalar(0, 255, 0), 2);
		}

		imshow("Result", result);
		
		if (char(cv::waitKey(10)) == 27)
			break;
	}
}
