// compile: g++ opencv.cpp -o opencv.o `pkg-config --cflags --libs opencv`

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

int main() {
	cv::VideoCapture cap(0);

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

		cv::Mat mask;

		/* Remove noise -- we're only going for orange on the mask. */
		cv::blur(frame, mask, cv::Size(10, 10));

		/* Use HSV. */
		cv::cvtColor(mask, mask, cv::COLOR_BGR2HSV);

		/* Try to threshold the general ball color. */
		cv::inRange(mask, cv::Scalar(2, 55, 150), cv::Scalar(15, 255, 255), mask);

		/* Get rid of spots. */
		cv::erode(mask, mask, cv::Mat());
		cv::dilate(mask, mask, cv::Mat());

		/* Get a gray frame. */
		cv::cvtColor(frame, filtered, cv::COLOR_BGR2GRAY);

		/* Small blur for noise. */
		cv::blur(filtered, filtered, cv::Size(2, 2));

		/* Apply the mask. */
		cv::bitwise_and(filtered, mask, filtered);

		imshow("Filtered", filtered);

		result = frame;

		std::vector<cv::Vec3f> circles;
		/* Don't know the best settings for this. */
		cv::HoughCircles(filtered, circles, cv::HOUGH_GRADIENT, 1, 10, 250, 40);

		/* Draw the circles on the result image. */
		for (int i = 0; i < circles.size(); i++) {
             cv::circle(result, cv::Point(circles[i][0], circles[i][1]), circles[i][2], cv::Scalar(0, 255, 0));
        }

		imshow("Result", result);
		
		if (char(cv::waitKey(10)) == 27)
			break;
	}
}
