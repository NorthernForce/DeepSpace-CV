#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <cmath>

const int k_invertHue = 0;
const int k_minHue = 0;
const int k_maxHue = 180;
const int k_minSat = 0;
const int k_maxSat = 150;
const int k_minVal = 220;
const int k_maxVal = 255;

const double k_polyAccuracy = 2;
const double k_minLength = 3;
//const double k_maxAreaDiff = 2;

const double k_maxFavoringAreaDiff = 0.1;
const double k_maxFavoringCenterOffset = 4;
const double k_maxSoftenerThreshold = 500;
const double k_maxFavoringBoundary = 0.3;

//const double k_minArea = 15;

//const double k_maxFavoringAreaDiff = 0.1;
//const double k_maxFavoringCenterOffset = 4;
//const double k_maxSoftenerThreshold = 700;
//const double k_maxFavoringBoundary = 0.3;

struct ReflectiveTapeEdge {
	cv::Point center;
	double length = 0;
	double angle = 0;
};

struct ReflectiveTapeBlob {
	ReflectiveTapeEdge top, right, bot, left;
	cv::Point center;
	double area = 0;
	bool isLeft;
	//bool isOut = true;
};

struct ReflectiveTargetBlob {
	cv::Point center;
	ReflectiveTapeBlob left;
	ReflectiveTapeBlob right;
	double area = 0;
};

class Utilities {
public:
	static cv::Point CalcAvgPoint(cv::Point a, cv::Point b) {
		return cv::Point((a.x + b.x) / 2, (a.y + b.y) / 2);
	}

	static double CalcLineLength(cv::Point a, cv::Point b) {
		return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2));
	}

	static double CalcLineAngle(cv::Point origin, cv::Point outer) {
		double angle = std::atan2(origin.y - outer.y, outer.x - origin.x);

		if (angle < 0) {
			angle += 6.28318530718;
		}

		//if (flipY) {
			//angle += 3.14159265;
		//}

		return angle;
	}

	static double CalcLineAngleDeg(cv::Point origin, cv::Point outer) {
		return CalcLineAngle(origin, outer) * 57.2957795131;
	}
};

int main() {
	cv::VideoCapture cap(0);
	cap.set(cv::CAP_PROP_FRAME_WIDTH, 240);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 180);

	/*
	 * Must run "opkg install v4l-utils" on the roborio
	 *
	 * Also, use -l to list the controls
	 *     - Valid exposures are "5, 10, 20, 39, 78, 156, 312, 625, 1250, 2500, 5000, 10000, 20000"
	 */
	 //system("v4l2-ctl -d 0 -c exposure_auto=1,exposure_absolute=5,white_balance_temperature_auto=0");

	cv::namedWindow("Plain", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Result", cv::WINDOW_AUTOSIZE);

	while (1) {
		cv::Mat frame;
		cap >> frame;
		if (frame.empty())
			break;

		cv::Mat plain;
		cv::resize(frame, plain, cv::Size(), 4, 4, 0);
		cv::imshow("Plain", plain);

		cv::Mat filtered = frame.clone();

		// Attempt to remove some noise.
		cv::blur(filtered, filtered, cv::Size(3, 3));

		// Use HSV
		cv::cvtColor(filtered, filtered, cv::COLOR_BGR2HSV);

		//// Gather values from the Smart Dashboard
		//int invertHue = frc::SmartDashboard::GetNumber("Vision: ReflectiveTape: HUE INVERT", k_invertHue);
		//int minHue = frc::SmartDashboard::GetNumber("Vision: ReflectiveTape: HUE MIN", k_minHue);
		//int maxHue = frc::SmartDashboard::GetNumber("Vision: ReflectiveTape: HUE MAX", k_maxHue);
		//int minSat = frc::SmartDashboard::GetNumber("Vision: ReflectiveTape: SAT MIN", k_minSat);
		//int maxSat = frc::SmartDashboard::GetNumber("Vision: ReflectiveTape: SAT MAX", k_maxSat);
		//int minVal = frc::SmartDashboard::GetNumber("Vision: ReflectiveTape: VAL MIN", k_minVal);
		//int maxVal = frc::SmartDashboard::GetNumber("Vision: ReflectiveTape: VAL MAX", k_maxVal);
		int invertHue = k_invertHue;
		int minHue = k_minHue;
		int maxHue = k_maxHue;
		int minSat = k_minSat;
		int maxSat = k_maxSat;
		int minVal = k_minVal;
		int maxVal = k_maxVal;

		// Threshold the image
		if (invertHue == 0) {
			cv::inRange(filtered, cv::Scalar(minHue, minSat, minVal), cv::Scalar(maxHue, maxSat, maxVal), filtered);
		}
		else {
			cv::Mat lower, upper;
			cv::inRange(filtered, cv::Scalar(0, minSat, minVal), cv::Scalar(minHue, maxSat, maxVal), lower);
			cv::inRange(filtered, cv::Scalar(maxHue, minSat, minVal), cv::Scalar(180, maxSat, maxVal), upper);
			cv::bitwise_or(lower, upper, filtered);
		}

		// For debugging show the bitmap on the smartdashboard
		// frame = filtered.clone();

		// Get rid of spots.
		// cv::erode(filtered, filtered, cv::Mat(), cv::Point(-1, -1), 2);
		// cv::dilate(filtered, filtered, cv::Mat(), cv::Point(-1, -1), 2);

		// Find contours.
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(filtered, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

		// Analyze the contours
		std::vector<ReflectiveTapeBlob> tapes;
		for (auto& contour : contours) {
			std::vector<cv::Point> poly;
			cv::approxPolyDP(contour, poly, k_polyAccuracy, true);
			poly.push_back(poly.front());

			//std::cout << "angle: ";

			std::vector<ReflectiveTapeEdge> edges;
			for (int pointI = 0; pointI < (poly.size() - 1); pointI++) {
				double length = Utilities::CalcLineLength(poly[pointI], poly[pointI + 1]);

				if (length > k_minLength) {
					ReflectiveTapeEdge edge;

					edge.length = length;
					edge.center = Utilities::CalcAvgPoint(poly[pointI], poly[pointI + 1]);
					edge.angle = Utilities::CalcLineAngleDeg(poly[pointI], poly[pointI + 1]);

					//std::cout << edge.angle << ", ";

					edges.push_back(edge);

					cv::line(frame, poly[pointI], poly[pointI + 1], cv::Scalar(0, 0, 255));
				}
			}

			//cv::circle(frame, poly[0], 1, cv::Scalar(255, 255, 255));
			//cv::circle(frame, poly[1], 1, cv::Scalar(150, 150, 150));

			//std::cout << "\n";

			// Make sure enough edges are detected
			if (edges.size() < 2) {
				continue;
			}

			// Sort edges by size
			std::sort(edges.begin(), edges.end(),
				[](ReflectiveTapeEdge& a, ReflectiveTapeEdge& b) {
					return a.length > b.length;
				});

			ReflectiveTapeBlob tape;
			for (auto& edge : edges) {
				switch ((int)((edge.angle + 45) / 90) % 4) {
				case 0: if (tape.bot.length < edge.length) tape.bot = edge; break;
				case 1: if (tape.right.length < edge.length) tape.right = edge; break;
				case 2: if (tape.top.length < edge.length) tape.top = edge; break;
				case 3: if (tape.left.length < edge.length) tape.left = edge; break;
				}
			}

			/*cv::circle(frame, tape.top.center, 1, cv::Scalar(255, 0, 0));
			cv::circle(frame, tape.left.center, 1, cv::Scalar(255, 255, 0));
			cv::circle(frame, tape.bot.center, 1, cv::Scalar(0, 255, 0));
			cv::circle(frame, tape.right.center, 1, cv::Scalar(0, 255, 255));*/

			//double botAdd = (tape.bot.length > 0) ? -(tape.bot.angle) : 0;
			double rightAdd = (tape.right.length > 0) ? (tape.right.angle - 90) : 0;
			//double topAdd = (tape.top.length > 0) ? -(tape.top.angle - 180) : 0;
			double leftAdd = (tape.left.length > 0) ? (tape.left.angle - 270) : 0;

			//double totalAngle = botAdd + rightAdd + topAdd + leftAdd;
			double totalAngle = rightAdd + leftAdd;
			if (totalAngle < 0) {
				tape.isLeft = true;
				cv::circle(frame, tape.right.center, 1, cv::Scalar(0, 255, 0));
				tape.center = tape.right.center;
			}
			else {
				tape.isLeft = false;
				cv::circle(frame, tape.left.center, 1, cv::Scalar(255, 0, 0));
				tape.center = tape.left.center;
			}

			//tape.area = (tape.left.length + tape.right.length) / 2;
			// Calculate tape area (as a trapezoid)
			double base = tape.left.length + tape.right.length;
			double height = tape.right.center.x - tape.left.center.x;
			tape.area = std::abs((base) / 2 * height);

			tapes.push_back(tape);

			////cv::drawContours(frame, std::vector<std::vector<cv::Point>>{poly}, -1, cv::Scalar(0, 255, 0));
			//cv::line(frame, poly[0], poly.back(), cv::Scalar(255, 0, 0));
			//for (int i = 1; i < poly.size(); i++) {
			//	cv::line(frame, poly[i-1], poly[i], cv::Scalar(255, 0, i*50%255));
			//}
		}

		//// Analyze the contours (as tapes)
		//std::vector<ReflectiveTapeBlob> tapes;
		//for (auto& contour : contours) {
		//	double testArea = cv::contourArea(contour);

		//	// Test for minimum area.
		//	if (testArea >= k_minArea) {
		//		ReflectiveTapeBlob tape;
		//		tape.area = testArea;

		//		// TODO: TRY TO FIX THIS
		//		// // Find the extreme points
		//		// cv::Point top, right, bot, left;
		//		// std::tie(top, right, bot, left) = Utilities::FindExtremePoints(contour);

		//		// Find extreme points
		//		cv::Point leftTop = contour[0], leftBot = contour[0], rightTop = contour[0], rightBot = contour[0];
		//		for (auto& point : contour) {
		//			if (point.x < leftTop.x) {
		//				leftTop = point;
		//				leftBot = point;
		//			}
		//			else if (point.x == leftTop.x) {
		//				if (point.y < leftTop.y) {
		//					leftTop = point;
		//				}
		//				else if (point.y > leftBot.y) {
		//					leftBot = point;
		//				}
		//			}

		//			if (point.x > rightTop.x) {
		//				rightTop = point;
		//				rightBot = point;
		//			}
		//			else if (point.x == rightTop.x) {
		//				if (point.y < rightTop.y) {
		//					rightTop = point;
		//				}
		//				else if (point.y > rightBot.y) {
		//					rightBot = point;
		//				}
		//			}
		//		}

		//		cv::Point right = Utilities::CalcAvgPoint(rightTop, rightBot);
		//		cv::Point left = Utilities::CalcAvgPoint(leftTop, leftBot);

		//		// Check whether the contour goes out of the frame
		//		cv::Rect rect = cv::boundingRect(contour);

		//		tape.isOut = true;
		//		if (rect.x <= 0) {
		//			tape.isLeft = true;
		//		}
		//		else if ((rect.x + rect.width) >= (frame.cols - 1)) {
		//			tape.isLeft = false;
		//		}
		//		else {
		//			// Compares heights of extreme points to determine whether left or right
		//			if (left.y < right.y) {
		//				tape.isLeft = false;
		//			}
		//			else {
		//				tape.isLeft = true;
		//			}

		//			if (rect.y > 0 && (rect.y + rect.height) < (frame.rows - 1)) {
		//				tape.isOut = false;
		//			}
		//		}

		//		// Set the key points for targetting
		//		if (tape.isLeft) {
		//			tape.center = right;
		//		}
		//		else {
		//			tape.center = left;
		//		}

		//		tapes.push_back(tape);
		//	}
		//}

		// Return if no tapes were found
		if (tapes.empty()) {
			//m_horizontalOffset = 0;
			//m_verticalOffset = 0;
			//return;
			continue;
		}

		// Sort tapes left to right
		std::sort(tapes.begin(), tapes.end(),
			[](ReflectiveTapeBlob & a, ReflectiveTapeBlob & b) {
				return a.center.x < b.center.x;
			});

		// Groups tapes into targets
		std::vector<ReflectiveTargetBlob> targets;
		for (auto& tape : tapes) {
			if (tape.isLeft) {
				if (targets.empty() || targets.back().right.area > 0) {
					// Creates a new target if all previous are complete
					ReflectiveTargetBlob target;
					target.left = tape;

					targets.push_back(target);
				}
				else if (targets.back().left.area < tape.area) {
					// Overrides previous tape if not complete and area is larger
					targets.back().left = tape;
				}
			}
			else {
				if (targets.empty()) {
					// If the first tape is right, it's alone
					ReflectiveTargetBlob target;
					target.right = tape;

					targets.push_back(target);
				}
				else if (targets.back().right.area < tape.area) {
					// Override the previous pair if new right tape is larger
					//if (tape.area > targets.back().left.area / k_maxAreaDiff) {
						// Only use the new tape if it is larger than half the left's area
						targets.back().right = tape;
					//}
				}
			}
		}

		// Define target areas and centers
		for (auto& target : targets) {
			// Total area is just addition of both areas
			target.area = target.left.area + target.right.area;

			if (target.right.area == 0) {
				// Don't use a single tape unless it doesn't touch a boundary
				//if (target.left.isOut) {
				//	target.center = cv::Point(frame.cols / 2, frame.rows / 2);
				//}
				//else {
					target.center = cv::Point(target.left.center);
				//}
			}
			else if (target.left.area == 0) {
				// Don't use a single tape unless it doesn't touch a boundary
				//if (target.right.isOut) {
				//	target.center = cv::Point(frame.cols / 2, frame.rows / 2);
				//}
				//else {
					target.center = cv::Point(target.right.center);
				//}
			}
		//	//else if (target.left.isOut || target.right.isOut) {
		//	//	// Use the raw center if either tape is out
		//	//	target.center = Utilities::CalcAvgPoint(target.right.center, target.left.center);;
		//	//}
			else {
				// Find offset severity based on tape areas
				double severity = (target.left.area / target.area - 0.5) / k_maxFavoringAreaDiff;
				if (severity < -1) {
					severity = -1;
				}
				else if (severity > 1) {
					severity = 1;
				}

				// Find the offset softener (larger area = less offset)
				double softener = (k_maxSoftenerThreshold - target.area) / k_maxSoftenerThreshold;
				if (softener < 0) {
					softener = 0;
				}

				// Calculate the true severity of the difference of areas
				severity *= softener * k_maxFavoringCenterOffset;

				cv::Point avgCenter = Utilities::CalcAvgPoint(target.left.center, target.right.center);
				int centerX = avgCenter.x;
				int centerY = avgCenter.y;

				cv::Rect boundary = cv::Rect(frame.cols * (0.5 - k_maxFavoringBoundary), 0, frame.cols * k_maxFavoringBoundary * 2, frame.rows - 1);
				if (boundary.contains(target.right.center) && boundary.contains(target.left.center)) {
					centerX += std::round(std::abs(target.right.center.x - target.left.center.x) * severity);
					centerY += std::round(std::abs(target.right.center.y - target.left.center.y) * severity);
				}

				target.center = cv::Point(centerX, centerY);
			}
		}

		// Find target with the greatest area
		auto largestTarget = *std::max_element(targets.begin(), targets.end(),
			[](ReflectiveTargetBlob & a, ReflectiveTargetBlob & b) {
				return a.area < b.area;
			});

		//// Debugging
		//cv::drawContours(frame, contours, -1, cv::Scalar(0, 255, 0));
		for (auto& target : targets) {
			cv::line(frame, target.left.center, target.right.center, cv::Scalar(255, 0, 0));
			cv::circle(frame, target.center, 1, cv::Scalar(0, 0, 255), 2);
		}
		cv::circle(frame, largestTarget.center, 2, cv::Scalar(0, 255, 255), 2);

		cv::Mat result;
		cv::resize(frame, result, cv::Size(), 4, 4, 0);
		cv::imshow("Result", result);

		//// Convert center to -1.0 to 1.0 where quadrant I is positive
		//m_horizontalOffset = (largestTarget.center.x - frame.cols / 2.0) / (frame.cols / 2.0);
		//m_verticalOffset = (largestTarget.center.y - frame.rows / 2.0) / (frame.rows / 2.0) * -1;

		if (char(cv::waitKey(10)) == 27)
			break;
	}
}
