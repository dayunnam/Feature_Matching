#include <iostream>
#include "opencv2/opencv.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/xfeatures2d.hpp"
#endif


using namespace cv;
using namespace std;

string features_type = "BRISK"; //AKAZE, BRISK, ORB

namespace {
	const char* about = "Feature Matching";
	const char* keys =
		"{base b        |       | base image file}"
		"{target t        |       | target image file}"
		"{output o        |       | output folder}"
		"{show s       |       | show result image}"
		"{help h ?       |       | Usage examples1:  --b=../dataset/Image_B07.tif --t=../dataset/Image_B01.tif  --o=../dataset --s=false}";
}
// --b=../dataset/sea_test0.jpg --t=../dataset/sea_test1.jpg  --o=../dataset --s=true
// --b=../dataset/city_test0.jpg --t=../dataset/city_test1.jpg  --o=../dataset --s=true


int main(int argc, char* argv[])
{
	//Parse arguments
	CommandLineParser parser(argc, argv, keys);
	parser.about(about);
	if (argc < 2 || parser.has("help")) {
		parser.printMessage();
		return -1;
	}

	if (!parser.has("b") || !parser.has("t")) {
		cerr << "no input image" << endl;
		return -1;
	}
	bool show_result = false;
	if (parser.has("show")) {
		show_result = parser.get<bool>("show");
	}

	string out_f = parser.has("o") ? parser.get<String>("o") : ".";

	Mat img0 = imread(parser.get<String>("b"), IMREAD_ANYCOLOR | IMREAD_ANYDEPTH); // base image
	Mat img1 = imread(parser.get<String>("t"), IMREAD_ANYCOLOR | IMREAD_ANYDEPTH); // target image

	if (img0.empty() || img1.empty()) return -1;


	if (img0.type() == CV_16UC1) {
		cv::Mat img0_c3(img0.rows, img0.cols, CV_8UC3);
		cv::Mat img1_c3(img0.rows, img0.cols, CV_8UC3);

		//cout << img0.type() << endl;
		for (int i = 0; i < img0.rows; i++) {
			for (int j = 0; j < img0.cols; j++)
			{
				img0_c3.at<Vec3b>(i, j)[0] = img0.at<unsigned short>(i, j) >> 3;
				img0_c3.at<Vec3b>(i, j)[1] = img0.at<unsigned short>(i, j) >> 3;
				img0_c3.at<Vec3b>(i, j)[2] = img0.at<unsigned short>(i, j) >> 3;

				img1_c3.at<Vec3b>(i, j)[0] = img1.at<unsigned short>(i, j) >> 3;
				img1_c3.at<Vec3b>(i, j)[1] = img1.at<unsigned short>(i, j) >> 3;
				img1_c3.at<Vec3b>(i, j)[2] = img1.at<unsigned short>(i, j) >> 3;
			}
		}

		img0 = img0_c3.clone();
		img1 = img1_c3.clone();
		imwrite(out_f + "/img0_org_8bit_color.tif", img0);
		imwrite(out_f + "/img1_org_8bit_color.tif", img1);
	}

	// Extract Feature
	std::cout << "(1)Extracting Feature" << std::endl;
	Ptr<FeatureDetector> extractor;
	if (features_type == "orb" || features_type == "ORB")
	{
		extractor = cv::ORB::create();
	}
	else if (features_type == "akaze" || features_type == "AKAZE")
	{
		extractor = cv::AKAZE::create();
	}
	else if (features_type == "brisk" || features_type == "BRISK")
	{
		extractor = cv::BRISK::create();
	}

#ifdef HAVE_OPENCV_XFEATURES2D
	else if (features_type == "surf" || features_type == "SURF")
	{
		extractor = cv::xfeatures2d::SURF::create();
	}
	else if (features_type == "sift" || features_type == "SIFT") {
		extractor = cv::xfeatures2d::SIFT::create();
	}
#endif
	else
	{
		std::cout << "Unknown 2D features type: '" << features_type << "'.\n";
		return -1;
	}

	vector<KeyPoint> key_pt0, key_pt1;
	Mat desc0, desc1;
	extractor->detectAndCompute(img0, Mat(), key_pt0, desc0);
	extractor->detectAndCompute(img1, Mat(), key_pt1, desc1);

	std::cout << "\t# keypoint1 : " << key_pt0.size() << std::endl;
	std::cout << "\t# keypoint2 : " << key_pt1.size() << std::endl;

	// Find Correspondence
	std::cout << "(2)Matching" << std::endl;
	//Ptr<DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
	Ptr<DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming(2)");
	vector<DMatch> matches;
	matcher->match(desc0, desc1, matches);

	//Remove outlier and Find homography matrix
	std::cout << "(3)Removing outliers and finding homography matrix" << std::endl;
	vector<Point2f> pts0, pts1;
	for (size_t i = 0; i < matches.size(); i++) {
		pts0.push_back(key_pt0.at(matches.at(i).queryIdx).pt);
		pts1.push_back(key_pt1.at(matches.at(i).trainIdx).pt);
	}


	Mat inlier_mask;
	Mat H = findHomography(pts1, pts0, inlier_mask, cv::RANSAC, 5.0);

	int inlier_num = 0;
	for (size_t i = 0; i < matches.size(); i++) {
		if (inlier_mask.at<bool>(i, 0))
			inlier_num++;
	}
	std::cout << "\t# inliers : " << inlier_num << std::endl;
	std::cout << "\tCorrect match rate : " << static_cast<float>(inlier_num) / static_cast<float>(matches.size())*100.0 << "%" << std::endl;

	cout << "\tHomography matrix: \n\t" << H.at<double>(0, 0) << " " << H.at<double>(0, 1) << " " << H.at<double>(0, 2) << std::endl;
	cout << "\t" << H.at<double>(1, 0) << " " << H.at<double>(1, 1) << " " << H.at<double>(1, 2) << std::endl;
	cout << "\t" << H.at<double>(2, 0) << " " << H.at<double>(2, 1) << " " << H.at<double>(2, 2) << std::endl;

	// Compose images
	std::cout << "(4)Composing images" << std::endl;
	Mat matched_img_withoutRANSAC, matched_img_withRANSAC, stitched_img;
	warpPerspective(img1, stitched_img, H, cv::Size(img0.cols * 2, img0.rows));
	stitched_img.colRange(0, img0.cols) = img0 * 1;

	drawMatches(img0, key_pt0, img1, key_pt1, matches, matched_img_withoutRANSAC, cv::Scalar::all(-1), cv::Scalar::all(-1)); // Remove 'inlier_mask' if you want to show all putative matches
	drawMatches(img0, key_pt0, img1, key_pt1, matches, matched_img_withRANSAC, cv::Scalar::all(-1), cv::Scalar::all(-1), inlier_mask); // Remove 'inlier_mask' if you want to show all putative matches

	Mat img0_inlier = img0.clone();
	for (size_t i = 0; i < matches.size(); i++) {
		if (inlier_mask.at<bool>(i, 0)) {
			circle(img0_inlier, pts0[i], 1, Scalar(0, 255, 0), 2); // green
			//line(img0_inlier, pts0[i], pts1[i], Scalar(0, 255, 0), 5); // green
		}
		else {
			circle(img0_inlier, pts0[i], 1, Scalar(0, 0, 255), 2); // red
			//line(img0_inlier, pts0[i], pts1[i], Scalar(0, 0, 255), 5); // red
		}
	}


	//Show Results
	Mat  img0_feature, img1_feature;
	drawKeypoints(img0, key_pt0, img0_feature, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(img1, key_pt1, img1_feature, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	if (show_result) {

		imshow("Keypoint0 (base image)", img0_feature);
		imshow("Keypoint1 (target image)", img1_feature);
		imshow("Inlier (green) vs Outlier (red)", img0_inlier);
		imshow("matched image without RANSAC", matched_img_withoutRANSAC);
		imshow("matched image without RANSAC", matched_img_withoutRANSAC);
		imshow("matched image with RANSAC", matched_img_withRANSAC);
		imshow("stitched image", stitched_img);
		waitKey(0);
	}




	//write tif file 
	imwrite(out_f + "/img0_feature.tif", img0_feature);
	imwrite(out_f + "/img1_feature.tif", img1_feature);
	imwrite(out_f + "/inlier(green)_outlier(red).tif", img0_inlier);
	imwrite(out_f + "/matched_image_withoutRANSAC.tif", matched_img_withoutRANSAC);
	imwrite(out_f + "/matched_image_withRANSAC.tif", matched_img_withRANSAC);
	imwrite(out_f + "/stitched_image.tif", stitched_img);

	//write text file
	string out_txt = out_f + "/matches_info.txt";
	FILE* out_txt_file = fopen(out_txt.c_str(), "wt");
	fprintf(out_txt_file, "(base image) 특징점 좌표\t(target image) 특징점 좌표\t매칭 거리\n");
	for (size_t i = 0; i < matches.size(); i++) {
		if (inlier_mask.at<bool>(i, 0)) {
			fprintf(out_txt_file, "(%f, %f)\t(%f, %f)\t%f\n", pts0[i].x, pts0[i].y, pts1[i].x, pts1[i].y, matches.at(i).distance);
		}
	}

	fclose(out_txt_file);

	std::cout << "[End of Program]" << std::endl;
	return 0;

}
