//视频匹配
#include <iostream>
#include <fstream>	
#include <opencv2/opencv.hpp>
#include <iomanip>
#include <chrono>
#include <string>

using namespace cv;
using namespace std;

int main()
{

	Mat X;  //传感器矩阵（观测值）
	Mat Q;  //预测误差
	Mat R;  //传感器噪声方差，对角矩阵
	Mat XPF;//Kalman预测矩阵
	Mat F;  //状态转移矩阵
	Mat H; 

	float FData[4][4] = { { 1, 1, 0, 0 },
	{ 0, 1, 0, 0 },
	{ 0, 0, 1, 1 },
	{ 0, 0, 0, 1 } };

	float HData[2][4] = { { 1, 0, 0, 0 },
	{ 0, 0, 1, 0 } };


	F = Mat(4, 4, CV_32FC1, FData[0]);
	H = Mat(2, 4, CV_32FC1, HData[0]); 

	X = Mat::zeros(4, 1, CV_32FC1);//传感器观测值
	XPF = Mat::zeros(4, 1, CV_32FC1);//kalman估计值


	Q = Mat::eye(4, 4, CV_32FC1);//预测方差
	R = Mat::eye(2, 2, CV_32FC1);//传感器测量方差
	R = 100000 * R;

	Q.at<float>(0, 0) = 0.5;
	Q.at<float>(2, 2) = 0.5;
	Q = 0.01*Q;

	RNG randN;
	sqrt(Q, Q);//获得标准差
	sqrt(R, R);
	Mat GuassX(4, 1, CV_32FC1);//传感器误差矩阵

	Mat img,imgOld;//原图像
	Mat tmp;//模板图像
	Mat rlt;//匹配之后的图像


	//原图像的位置
	//img = imread("sample\\1_sample_img.png");
	//模板图像的位置
	tmp = imread("template-picture\\tmp.png");

	double MaxVal;
	double mean_x1;
	//region of interest ROI感兴趣区域（黄色框区域）（目标运动区域）
	double roi_x = 70;
	double roi_y = 100;
	double roi_xx = 330;
	double roi_yy = 450;
	Point max_pt, max_ptOld;


	//保存模板匹配的视频
	VideoWriter writer("template.avi", VideoWriter::fourcc('D', 'I', 'V', '3'), 10.0, Size(749, 559));  

	Rect roi = { (int)roi_x, (int)roi_y, (int)roi_xx, (int)roi_yy };

	//原视频的位置
	VideoCapture cap("video\\pulse=60000.avi");

	//原视频的画面数量
	int max_frame = (int)cap.get(CAP_PROP_FRAME_COUNT);
	//原视频的高度
	int img_h = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
	//原视频的宽度
	int img_w = (int)cap.get(CAP_PROP_FRAME_WIDTH);
	//原视频的帧数
	double fps = cap.get(CAP_PROP_FPS);

	std::cout << "clock():\n";
	std::ofstream csv("csv_file\\a=0.csv");

	//初始化第一个状态
	X.at<float>(0, 0) = 0;
	X.at<float>(1, 0) = 0;
	X.at<float>(2, 0) = 0;
	X.at<float>(3, 0) = 0;

	Mat PCov = Mat::eye(4, 4, CV_32FC1);//协方差矩阵 
	Mat Eye = Mat::eye(4, 4, CV_32FC1);

	for (int i = 0; i < max_frame; i++) {

		//取出的每一帧画面，让该画面为上定义的img
		cap >> img;

		// 进行视频模板匹配
		matchTemplate(img, tmp, rlt, TM_CCORR_NORMED);

		//minMaxLoc(Mat型的输出结果，类似度最低的值，类似度最高的值，低坐标，高坐标)
		minMaxLoc(rlt, NULL, &MaxVal, NULL, &max_pt);


		randN.fill(GuassX, RNG::NORMAL, 0, 1);
		GuassX = Q*GuassX;//得到相应方差的高斯矩阵

		if (i > 0)
		{
			if ((abs(max_pt.x - XPF.at<float>(0, 0)) < 70) && (abs(max_pt.y - XPF.at<float>(2, 0)) < 70))
			{
				X.at<float>(0, 0) = max_pt.x;
				X.at<float>(2, 0) = max_pt.y;
			}	
		}
		else if (i == 0)
		{
			X.at<float>(0, 0) = max_pt.x;
			X.at<float>(2, 0) = max_pt.y;
			X.copyTo(XPF);
		}
		Mat X2 = F * X + GuassX;
		X2.copyTo(X);//观测量预测值

		if (i > 0)
		{
			Mat XP1 = F*XPF;//预测
			Mat P1 = F*PCov*(F.t()) + Q; //预测协方差误差

			Mat sumCov = H*P1*(H.t()) + R;
			Mat kAdd = P1*(H.t())*(sumCov.inv());//卡尔曼增益
			if ((abs(XPF.at<float>(0, 0) - max_pt.x) < 70) && (abs(max_pt.y - XPF.at<float>(2, 0)) < 70))
			{
				X.at<float>(1, 0) = max_pt.x - max_ptOld.x;
				X.at<float>(3, 0) = max_pt.y - max_ptOld.y;
			}
			Mat XP2 = XP1 + kAdd*(H*X - H*XP1);//预测值校正

			XP2.copyTo(XPF);

			Mat tempPCov = (Eye - kAdd*H)*P1;//协方差校正
			tempPCov.copyTo(PCov);
		}

		max_ptOld.x = max_pt.x;
		max_ptOld.y = max_pt.y;

		// 在视频中用红色框圈出目标-------------------------------------------------------------------------
		rectangle(img, max_pt, Point(max_pt.x + tmp.cols, max_pt.y + tmp.rows), Scalar(0, 0, 255), 2, 8, 0);

		// 在视频中用绿色框圈出目标预测-------------------------------------------------------------------------
		rectangle(img, Point(XPF.at<float>(0, 0), XPF.at<float>(2, 0)), Point(XPF.at<float>(0, 0) + tmp.cols, XPF.at<float>(2, 0) + tmp.rows), Scalar(0, 255, 0), 2, 8, 0);

		//在视频中用绿色画出ROI区域的中心线-----------------------------------------------------------------
		//line(img, Point(roi_x + roi_xx / 2, 0), Point(roi_x + roi_xx / 2, img_h), Scalar(0, 255, 0), 2, 8, 0);

		//在视频中用黄色圈出ROI区域--------------------------------------------------------------------------
		//rectangle(img, roi, Scalar(0, 255, 255), 2, 8, 0);

		//红色目标区域的中心线---------------------------------------------------------------------------
		//mean_x1 = (max_pt.x + max_pt.x + tmp.cols) / 2;
		//line(img, Point(mean_x1, 0), Point(mean_x1, img_h), Scalar(0, 0, 255), 2, 8, 0);
		//writer << img;          //*************************************************************
		//当前画面数的号码
		std::cout << "Number of picture " << i << std::endl;

		//目标与中心线的距离----------------------------------------------------------------------------------------
		//std::cout << "Distance = " << roi_x + roi_xx / 2 - mean_x1 << std::endl;

		if (max_pt.x + tmp.cols / 2 < img_w / 2)
		{
			std::cout << "left" << std::endl;
		}
		else if (max_pt.x + tmp.cols / 2 == img_w / 2)
		{
			std::cout << "center" << std::endl;
		}
		else if (max_pt.x + tmp.cols / 2 > img_w / 2)
		{
			std::cout << "right" << std::endl;
		}

		//位置用Excel保存
		//csv << i << "," << roi_x + roi_xx / 2 - mean_x1 << std::endl;

		//circle(img, Point(max_pt.x + tmp.cols, max_pt.y + tmp.rows), 10, Scalar(255, 0, 230), 2, 1);
		//circle(img, Point(max_pt.x + tmp.cols / 2, max_pt.y + tmp.rows / 2), 10, Scalar(255, 0, 230), 2, 1);

		// 显示视频名
		imshow("Video", img);

		//当前画面数的号码
		std::cout << "Number of picture " << i << std::endl;

		//当前画面与目标的匹配程度和目标的坐标
		std::cout << "Similarity=" << MaxVal << std::endl;

		std::cout << "Target coordinates = " << Point(max_pt.x + tmp.cols / 2, max_pt.y + tmp.rows / 2) << std::endl;
        
        //当前卡尔曼预测坐标
        std::cout <<"Prediction coordinates =" << Point(XPF.at<float>(0, 0) + tmp.cols / 2, XPF.at<float>(2, 0) + tmp.rows / 2) << std::endl;

		//匹配度和匹配框中心点坐标用Excel保存
		csv << i << "," << MaxVal << "," << max_pt.x + tmp.cols / 2 << "," << max_pt.y + tmp.rows / 2 << ",";
        //预测框中心点坐标用Excel保存
		csv << XPF.at<float>(0, 0) + tmp.cols / 2 << "," << XPF.at<float>(2, 0) + tmp.rows / 2 << endl;
		/*if (getchar() == 'q')
			break;*/
		// 
		waitKey(1);
	}

	//动画信息
	std::cout << "フレーム番号 " << max_frame - 1 << std::endl;
	std::cout << "動画の高さ " << img_h << std::endl;
	std::cout << "動画の幅 " << img_w << std::endl;
	std::cout << "動画のfps " << fps << std::endl;

	return 0;
}
