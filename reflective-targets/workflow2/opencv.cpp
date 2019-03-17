// compile: g++ opencv.cpp -o opencv.o `pkg-config --cflags --libs opencv`

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

struct ReflectiveTape {
	cv::Point center;
	double area = 0;
	bool isLeft;
};

struct ReflectiveTarget {
	cv::Point center;
	ReflectiveTape leftTape;
	ReflectiveTape rightTape;
	double totalArea = 0;
};

const int k_minArea = 15;
// // Green:
// const cv::Scalar k_minHSV = cv::Scalar(35, 150, 50);
// const cv::Scalar k_maxHSV = cv::Scalar(90, 255, 255);
// // Pink:
// const cv::Scalar k_minHSV = cv::Scalar(165, 150, 150);
// const cv::Scalar k_maxHSV = cv::Scalar(185, 255, 255);
// Bright:
const cv::Scalar k_minHSV = cv::Scalar(0, 0, 150);
const cv::Scalar k_maxHSV = cv::Scalar(255, 255, 255);

double m_horizontalOffset = 0;
double m_verticalOffset = 0;

int main() {
	cv::VideoCapture cap(0);

	/*
	 * Must run "opkg install v4l-utils" on the roborio
	 * 
	 * Also, use -l to list the controls
	 *     - Valid exposures are "5, 10, 20, 39, 78, 156, 312, 625, 1250, 2500, 5000, 10000, 20000"
	 */
	system("v4l2-ctl -d 0 -c exposure_auto=1,exposure_absolute=5,white_balance_temperature_auto=0");

	cv::namedWindow("Plain", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Filtered", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Result", cv::WINDOW_AUTOSIZE);

	while (1) {
		cv::Mat frame;
		cap >> frame;
		if (frame.empty())
			break;

		cv::imshow("Plain", frame);

		cv::Mat filtered = frame.clone();
		cv::Mat debug = frame.clone();

		// Attempt to remove some noise.
		cv::blur(filtered, filtered, cv::Size(3, 3));

		// Use HSV.
		cv::cvtColor(filtered, filtered, cv::COLOR_BGR2HSV);

		// Try to threshold the tape.
		cv::inRange(filtered, k_minHSV, k_maxHSV, filtered);

		// Find contours.
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(filtered, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

		// Analyze contours
		std::vector<ReflectiveTape> tapes;
		for (auto &contour : contours) {
			cv::Moments test = cv::moments(contour, false);

			// Test for minimum area.
			if (test.m00 >= k_minArea) {
				ReflectiveTape tape;
				tape.area = test.m00;
				tape.center = cv::Point(test.m10/test.m00, test.m01/test.m00);
				
				// Find furthest left and right points (extreme points)
				auto points = std::minmax_element(contour.begin(), contour.end(),
					[](cv::Point &a, cv::Point &b) {
						return a.x < b.x;
				});

				// Compares heights of extreme points to determine whether left or right
				if (points.first->y < points.second->y) {
					tape.isLeft = false;
				}
				else {
					tape.isLeft = true;
				}

				tapes.push_back(tape);
			}
		}

		// Return if not tapes were found
		if (tapes.empty()) {
			m_horizontalOffset = 0;
			m_verticalOffset = 0;

			cv::imshow("Filtered", filtered);
			cv::imshow("Result", debug);
			
			if (char(cv::waitKey(10)) == 27)
				break;

			continue;
		}

		// Sort tapes left to right
		std::sort(tapes.begin(), tapes.end(),
			[](ReflectiveTape &a, ReflectiveTape &b) {
				return a.center.x < b.center.x;
		});

		// Groups tapes into targets
		std::vector<ReflectiveTarget> targets;
		for (auto &tape : tapes) {
			if (tape.isLeft) {
				if (targets.empty() || targets.back().rightTape.area > 0) {
					// Creates a new target if all previous are complete
					ReflectiveTarget target;
					target.leftTape = tape;

					targets.push_back(target);
				}
				else if (targets.back().leftTape.area < tape.area) {
					// Overrides previous tape if not complete and area is larger
					targets.back().leftTape = tape;
				}
			}
			else {
				if (targets.empty()) {
					// If the first tape is right, it's alone
					ReflectiveTarget target;
					target.rightTape = tape;

					targets.push_back(target);
				}
				else if (targets.back().rightTape.area < tape.area) {
					// Override the previous pair if new right tape is larger
					targets.back().rightTape = tape;
				}
			}
		}

		// Define target areas and centers
		for (auto &target : targets) {
			// Only use average center if both tapes are found
			if (target.leftTape.area > 0 && target.rightTape.area == 0) {
				target.center = target.leftTape.center;
			}
			else if (target.leftTape.area == 0 && target.rightTape.area > 0) {
				target.center = target.rightTape.center;
			}
			else if (target.leftTape.area > 0 && target.rightTape.area > 0) {
				target.center = cv::Point((target.leftTape.center.x + target.rightTape.center.x) / 2,
										  (target.leftTape.center.y + target.rightTape.center.y) / 2);
			}

			// Total area is just addition of taoe areas
			target.totalArea = target.leftTape.area + target.rightTape.area;
		}

		// Find target with the greatest area
		auto largestTarget = *std::max_element(targets.begin(), targets.end(),
			[](ReflectiveTarget &a, ReflectiveTarget &b) {
				return a.totalArea < b.totalArea;
		});

		// Debug
		cv::drawContours(debug, contours, -1, cv::Scalar(0, 255, 0));
		for (auto &target : targets) {
			cv::line(debug, target.leftTape.center, target.rightTape.center, cv::Scalar(255, 0, 0));
			cv::circle(debug, target.center, 1, cv::Scalar(0, 0, 255), 2);
		}
		cv::circle(debug, largestTarget.center, 2, cv::Scalar(0, 255, 255), 2);

		// Convert to -1.0 to 1.0 where quadrant I is positive
		m_horizontalOffset = (largestTarget.center.x - frame.cols / 2.0) / (frame.cols / 2.0);
		m_verticalOffset = (largestTarget.center.y - frame.rows / 2.0) / (frame.rows / 2.0) * -1;

		std::cout << "m_horizontalOffset: " << m_horizontalOffset << " m_verticalOffset: " << m_verticalOffset << "\n";

		cv::imshow("Filtered", filtered);
		cv::imshow("Result", debug);
		
		if (char(cv::waitKey(10)) == 27)
			break;
	}
}
