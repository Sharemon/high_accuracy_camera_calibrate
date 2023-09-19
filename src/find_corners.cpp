/**
 * Copyright @2023 Sharemon. All rights reserved.
 *
 @author: sharemon
 @date: 2023-09-07
 */

#include "find_corners.hpp"
#include "generate_pattern.hpp"
#include <vector>
#include <iostream>
#include <opencv2/ximgproc/find_ellipses.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace high_accuracy_corner_detector;


/// @brief 计算两个点之间的距离
/// @param p1 点1
/// @param p2 点2
/// @return 距离
inline double calc_dist_between_points(cv::Point2f p1, cv::Point2f p2)
{
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

/// @brief 计算椭圆拟合的相关性
/// @param contour 轮廓点
/// @param ellipse_rect 椭圆拟合得到的矩形框
/// @return 相关性
double calc_ellipse_fit_relation(const std::vector<cv::Point>& contour, const cv::RotatedRect& ellipse_rect)
{
    double a = ellipse_rect.size.width / 2;
    double b = ellipse_rect.size.height / 2;
    double theta = ellipse_rect.angle / 180.0 * CV_PI;
    double cx = ellipse_rect.center.x;
    double cy = ellipse_rect.center.y;

    double sqsum_result = 0;
    double sqsum_residual = 0;
    for (auto pt: contour)
    {
        double x_trans = (pt.x - cx) * cos(theta) - (pt.y - cy) * sin(theta);
        double y_trans = (pt.x - cx) * sin(theta) + (pt.y - cy) * cos(theta);

        double result = pow(x_trans,2)/pow(a,2) + pow(y_trans,2)/pow(b,2);
        double residual = pow(sqrt(result) - 1, 2);

        sqsum_result += result;
        sqsum_residual += (result - residual);
    }

    return sqsum_residual / sqsum_result;
}


/// @brief 计算椭圆弧度范围
/// @param contour 轮廓点
/// @param ellipse_rect 椭圆拟合得到的矩形框
/// @return 弧度范围
double calc_ellipse_angle_range(const std::vector<cv::Point>& contour, const cv::RotatedRect& ellipse_rect)
{
    cv::Point2f center = ellipse_rect.center;

    // 将360°分成36个区域，每各区域10°，统计每一各区域中的点数
    std::vector<int> angles(36, 0);
    for (auto pt : contour)
    {
        cv::Point2f pt_2_center(pt.x - center.x, pt.y - center.y);
        angles[(atan2(pt_2_center.y, pt_2_center.x) + CV_PI) * 180 / CV_PI / 10]++;
    }

    // 统计36个区域中有轮廓点的区域个数
    int sum = 0;
    for (int i = 0; i < angles.size(); i++)
    {
        if (angles[i] > 0)
        {
            sum++;
        }
    }

    return (sum / 36.0);
}


/// @brief 计算新的椭圆中心距离已有椭圆中心的最小距离
/// @param ellipses 已有椭圆列表
/// @param ellipse_compared 新的椭圆
/// @return 最小距离
double min_dist_from_existed_ellipses(const std::vector<cv::RotatedRect>& ellipses, const cv::RotatedRect& ellipse_compared)
{
    double dist_min = DBL_MAX;
    for (auto ellipse: ellipses)
    {
        double dist = calc_dist_between_points(ellipse.center, ellipse_compared.center);
        if (dist < dist_min)
        {
            dist_min = dist;
        }
    }

    return dist_min;
}


/// @brief 正平面控制点y方向比较函数
/// @param a 点a
/// @param b 点b
/// @return 是否交换a,b
bool control_pts_frontal_compare_y(cv::Point2f a, cv::Point2f b)
{
    return (a.y < b.y);
}

/// @brief 正平面控制点x方向比较函数
/// @param a 点a
/// @param b 点b
/// @return 是否交换a,b
bool control_pts_frontal_compare_x(cv::Point2f a, cv::Point2f b)
{
    return (a.x < b.x);
}


/// @brief 对角点排序
/// @param ellipses 椭圆拟合结果
/// @param corners 排序后的角点
/// @param pattern_size 标定图案尺寸
void order_corners(const std::vector<cv::RotatedRect>& ellipses, std::vector<cv::Point2f>& corners, const cv::Size& pattern_size)
{
    // 1. 计算所有圆心的中点
    cv::Point2f center(0,0);
#if 0
    for (auto ellipse : ellipses)
    {
        center += ellipse.center;
    }

    center /= (int)ellipses.size();
#else
    std::vector<cv::Point2f> centers;
    for (auto e:ellipses)
    {
        centers.push_back(e.center);
    }
    cv::Rect rect_outlayer = cv::boundingRect(centers);
    center.x = rect_outlayer.x + rect_outlayer.width/2.0;
    center.y = rect_outlayer.y + rect_outlayer.height/2.0;
#endif

    // 2. 计算四个角点
    std::vector<std::pair<int, double>> idxs_and_dists(4);
    idxs_and_dists[0].first = idxs_and_dists[1].first = idxs_and_dists[2].first = idxs_and_dists[3].first = -1;
    idxs_and_dists[0].second = idxs_and_dists[1].second = idxs_and_dists[2].second = idxs_and_dists[3].second = -1;
    for (int i =0;i<ellipses.size();i++)
    {
        double dist = calc_dist_between_points(ellipses[i].center, center);

        cv::Point2f pt = ellipses[i].center - center;
        int area = 0; // 顺序为左上、右上、右下、左下
        if (pt.x < 0 && pt.y <= 0)
        {
            area = 0;
        }
        else if (pt.x <= 0 && pt.y > 0)
        {
            area = 3;
        }
        else if (pt.x > 0 && pt.y >= 0)
        {
            area = 2;
        }
        else
        {
            area = 1;
        }

        if (dist > idxs_and_dists[area].second)
        {
            idxs_and_dists[area].second = dist;
            idxs_and_dists[area].first = i;
        }
    }

    // 判断四个点分别是什么方位的角点，假设每个角点相对于中心点分别属于一个象限
    std::vector<cv::Point2f> corner_pts = {
        ellipses[idxs_and_dists[0].first].center,
        ellipses[idxs_and_dists[1].first].center,
        ellipses[idxs_and_dists[2].first].center,
        ellipses[idxs_and_dists[3].first].center
    };

    // 3. 计算每条边的长度，确定起始点和长短边方向
    cv::Vec2f line_up = corner_pts[1] - corner_pts[0];
    cv::Vec2f line_left = corner_pts[3] - corner_pts[0];
    int start_idx = 0;

    if ((pow(line_up[0], 2) + pow(line_up[1], 2)) > (pow(line_left[0], 2) + pow(line_left[1], 2)))
    {
        // 如果上边长度大于左边，则起始点为左上
        start_idx = 0;
    }
    else
    {
        // 否则，起始点为左下
        start_idx = 3;
    }

    // 4. 通过单应变换将所有控制点转换到正平面
    std::vector<cv::Point2f> dst_points(4);
    if (start_idx == 0)
    {
        dst_points[0] = cv::Point2f(0   ,0);
        dst_points[1] = cv::Point2f(1000,0);
        dst_points[2] = cv::Point2f(1000,1000);
        dst_points[3] = cv::Point2f(0   ,1000);
    }
    else
    {
        dst_points[3] = cv::Point2f(0   ,0);
        dst_points[0] = cv::Point2f(1000,0);
        dst_points[1] = cv::Point2f(1000,1000);
        dst_points[2] = cv::Point2f(0   ,1000);
    }

    cv::Mat H = cv::findHomography(corner_pts, dst_points);

    std::vector<cv::Point2f> control_pts_src(ellipses.size()), control_pts_frontal(ellipses.size());
    for (int i=0;i<ellipses.size();i++)
    {
        control_pts_src[i] = ellipses[i].center;
    }
    cv::perspectiveTransform(control_pts_src, control_pts_frontal, H);

    // 5. 在正平面对所有控制点进行排序
    std::sort(control_pts_frontal.begin(), control_pts_frontal.end(), control_pts_frontal_compare_y);
    for (int i=0;i<pattern_size.height;i++)
    {
        std::sort(control_pts_frontal.begin() + i * pattern_size.width, control_pts_frontal.begin() + (i + 1) * pattern_size.width, control_pts_frontal_compare_x);
    }

    // 6. 将控制点从正平面转换回来
    cv::perspectiveTransform(control_pts_frontal, corners, H.inv());
}


/// @brief 从图像中找特征点
/// @param image 图像数据
/// @param corners 角点结果
/// @param pattern_infos 标定图案信息
void high_accuracy_corner_detector::find_corners(const cv::Mat &image, std::vector<cv::Point2f> &corners, const pattern_infos_t pattern_infos)
{
    corners.clear();

    // 0. 高斯滤波预处理
    cv::Mat image_gaussian;
    cv::GaussianBlur(image, image_gaussian, cv::Size(11, 11), 0);

    // 1. 边缘检测
    cv::Mat edge;
    cv::Canny(image_gaussian, edge, 100, 150);
    // // 做一个闭运算将断开的轮廓连起来
    // cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11));
	// cv::Mat result;
	// cv::morphologyEx(image, image, cv::MORPH_CLOSE, element);

    // 2. 椭圆检测
    // 2.1. 轮廓检测
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edge, contours, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

    // 2.2. 椭圆拟合 & 滤波
    const double w_h_ratio_threshold = 0.7;          
    const double w_h_max = image.size().width / 12.0;
    const double w_h_min = image.size().width / 72.0;
    const double r_min = 0.98;
    const double angle_range_min = 0.75;
    const double dist_min = 10;
    std::vector<cv::RotatedRect> ellipses;
    for (int i = 0; i < contours.size(); i++)
    {
        if (contours[i].size() > 10)
        {
            // 椭圆拟合
            cv::RotatedRect ellipse_rect = cv::fitEllipse(contours[i]);

            // 计算长宽比
            double w_h_ratio = std::min(ellipse_rect.size.width, ellipse_rect.size.height) / std::max(ellipse_rect.size.width, ellipse_rect.size.height);

            // 计算拟合度
            double r = calc_ellipse_fit_relation(contours[i], ellipse_rect);

            // 计算圆弧角度
            double angle_range = calc_ellipse_angle_range(contours[i], ellipse_rect);

            if (w_h_ratio > w_h_ratio_threshold &&
                std::min(ellipse_rect.size.width, ellipse_rect.size.height) > w_h_min &&
                std::max(ellipse_rect.size.width, ellipse_rect.size.height) < w_h_max &&
                min_dist_from_existed_ellipses(ellipses, ellipse_rect) > dist_min &&
                r > r_min &&
                angle_range > angle_range_min)
            {
                ellipses.push_back(ellipse_rect);
                // std::cout << ellipse_rect.center << ":" << r << ", " << contours[i].size() << ", " << angle_range << std::endl;
            }
        }
    }

    std::cout << "find " << ellipses.size() << " ellipses." << std::endl;

    // 2.3 判断椭圆个数 & 返回角点
    if (ellipses.size() == pattern_infos.size.width * pattern_infos.size.height)
    {
        // 2.4 按顺序排列找到的椭圆，并将圆心坐标放入corners
        order_corners(ellipses, corners, pattern_infos.size);
    }

    // cv::Mat show;
    // cv::cvtColor(edge, show, cv::COLOR_GRAY2RGB);

    // for (int i = 0; i < ellipses.size(); i++)
    // {
    //     cv::ellipse(show, ellipses[i], cv::Scalar(0, 255, 0), 4);
    //     if (i != ellipses.size() - 1 && corners.size() == ellipses.size())
    //         cv::line(show, corners[i], corners[i+1], cv::Scalar(255, 255, 0), 4);
    //     // cv::circle(show, ellipses[i], 16, cv::Scalar(0,255,0),4);
    // }
    // cv::resize(show, show, image.size() / 4);
    // cv::imshow("show", show);
    // cv::waitKey(0);
}


/// @brief 找矩阵最大值，并进行亚像素优化，参考https://cloud.tencent.com/developer/article/2010095
/// @param matrix 矩阵数据
/// @return 最大值坐标
cv::Point2f find_mat_max_subpixel(const cv::Mat& matrix)
{
#if 1
    // 先找最大值
    cv::Point max_loc;
    cv::minMaxLoc(matrix, NULL, NULL, NULL, &max_loc);

    // 在最大值附近拟合二维二次曲线
    Eigen::MatrixXd A(9, 6);
    Eigen::VectorXd b(9);

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            int x = j - 1;
            int y = i - 1;

            A(j + 3 * i, 0) = x*x;
            A(j + 3 * i, 1) = y*y;
            A(j + 3 * i, 2) = x*y;
            A(j + 3 * i, 3) = x;
            A(j + 3 * i, 4) = y;
            A(j + 3 * i, 5) = 1;

            b(j + 3 * i) = matrix.at<float>(y + max_loc.y,x + max_loc.x);
        }
    }

    // z中为二维二次曲线的系数
    // f(x,y) = z(0) * x*x + z(1) *y*y + z(2) *x*y + z(3) *x + z(4) *y + z(5)
    Eigen::VectorXd z = (A.transpose() * A).inverse() * A.transpose() * b;

    // 对二次曲线求导
    Eigen::Matrix2d K1;
    Eigen::Vector2d K2;

    K1 << 2*z(0), z(2), z(2), 2*z(1);
    K2 << -z(3), -z(4);

    Eigen::Vector2d r = K1.inverse() * K2;
    return cv::Point2f(r(0) + max_loc.x, r(1) + max_loc.y);
    // return cv::Point2f(max_loc.x, max_loc.y);
#else
    // 拟合二维二次曲线
    Eigen::MatrixXd A(matrix.rows*matrix.cols, 6);
    Eigen::VectorXd b(matrix.rows*matrix.cols);

    for (int y = 0; y < matrix.rows; y++)
    {
        for (int x = 0; x < matrix.cols; x++)
        {
            A(x + y*matrix.cols, 0) = x*x;
            A(x + y*matrix.cols, 1) = y*y;
            A(x + y*matrix.cols, 2) = x*y;
            A(x + y*matrix.cols, 3) = x;
            A(x + y*matrix.cols, 4) = y;
            A(x + y*matrix.cols, 5) = 1;

            b(x + y*matrix.cols) = matrix.at<float>(y,x);
        }
    }

    // z中为二维二次曲线的系数
    // f(x,y) = z(0) * x*x + z(1) *y*y + z(2) *x*y + z(3) *x + z(4) *y + z(5)
    Eigen::VectorXd z = (A.transpose() * A).inverse() * A.transpose() * b;

    // 对二次曲线求导
    Eigen::Matrix2d K1;
    Eigen::Vector2d K2;

    K1 << 2*z(0), z(2), z(2), 2*z(1);
    K2 << -z(3), -z(4);

    Eigen::Vector2d r = K1.inverse() * K2;
    return cv::Point2f(r(0), r(1));
#endif
}


/// @brief 优化控制点
/// @param image 图像数据
/// @param corners 控制点结果
/// @param pattern_infos 标定图案信息
/// @param K 初步标定得到的内参
/// @param D 初步标定得到的畸变参数
/// @param rvecs 初步标定得到的标定板旋转
/// @param tvecs 初步标定得到的标定板位移
void high_accuracy_corner_detector::refine_corners(
    const cv::Mat &image, std::vector<cv::Point2f> &corners, const pattern_infos_t pattern_infos,
    const cv::Mat &K, const cv::Mat &D)
{
    // 1. 图像去畸变
    cv::Mat img_und;
    cv::undistort(image, img_und, K, D, cv::getOptimalNewCameraMatrix(K, D, image.size(), 1));

    // 2. 控制点去畸变
    std::vector<cv::Point2f> corners_und;
    cv::undistortPoints(corners, corners_und, K, D, cv::Mat(), cv::getOptimalNewCameraMatrix(K, D, image.size(), 1));

    // 3. 选取控制点的4个顶点
    std::vector<cv::Point2f> corners_of_corners =
    {
        corners_und[0],
        corners_und[pattern_infos.size.width - 1],
        corners_und[pattern_infos.size.width * (pattern_infos.size.height - 1)],
        corners_und[pattern_infos.size.width * pattern_infos.size.height - 1]
    };

    // 4. 计算校正后的图像顶点坐标
    std::vector<cv::Point2f> corners_of_corners_frontal=
    {
        cv::Point2f(
            pattern_infos.margin_size,pattern_infos.margin_size
        ) / pattern_infos.pixel_size,
        cv::Point2f(
            pattern_infos.margin_size + (pattern_infos.size.width - 1) * pattern_infos.distance, pattern_infos.margin_size
        ) / pattern_infos.pixel_size,
        cv::Point2f(
            pattern_infos.margin_size, pattern_infos.margin_size + (pattern_infos.size.height - 1) * pattern_infos.distance
        ) / pattern_infos.pixel_size,
        cv::Point2f(
            pattern_infos.margin_size + (pattern_infos.size.width - 1) * pattern_infos.distance, pattern_infos.margin_size  + (pattern_infos.size.height - 1) * pattern_infos.distance
        ) / pattern_infos.pixel_size
    };
    // 5. 计算图像到正平面的H矩阵
    cv::Mat H = cv::findHomography(corners_of_corners, corners_of_corners_frontal);

    // 6. 图像转换到正平面
    // 6.1 计算透视变换后的图像大小
    std::vector<cv::Point2f> vertex = {
        cv::Point2f(0,0), 
        cv::Point2f(img_und.cols-1, 0), 
        cv::Point2f(0, img_und.rows-1), 
        cv::Point2f(img_und.cols-1, img_und.rows-1)
    };
    std::vector<cv::Point2f> vertex_frontal(4);
    cv::perspectiveTransform(vertex, vertex_frontal, H);
    cv::Rect rect_frontal = cv::boundingRect(vertex_frontal);

    // 6.2 透视变换
    cv::Mat img_frontal;
    cv::warpPerspective(img_und, img_frontal, H, rect_frontal.size());

    // 6.3 截取标定板区域图案
    cv::Mat img_pattern = img_frontal(
        cv::Rect(
            cv::Point(0,0), 
            cv::Point(
                (pattern_infos.distance * (pattern_infos.size.width-1) + pattern_infos.margin_size * 2) / pattern_infos.pixel_size,
                (pattern_infos.distance * (pattern_infos.size.height-1) + pattern_infos.margin_size * 2) / pattern_infos.pixel_size
            )
        )
    );                

    // 7. 模板匹配优化控制点
    // 7.1 去畸变的控制点映射到正平面
    std::vector<cv::Point2f> corners_und_frontal;
    cv::perspectiveTransform(corners_und, corners_und_frontal, H);

    // 7.2 图像增强
    //cv::threshold(img_pattern, img_pattern, 0, 255, cv::THRESH_OTSU);

    // 7.3 生成单个panttern图像
    cv::Mat img_single_pattern;
    high_accuracy_corner_detector::generate_single_pattern(pattern_infos, img_single_pattern);

    // 7.4 模板匹配
    cv::Mat result;
    cv::matchTemplate(img_pattern, img_single_pattern, result, cv::TM_CCORR_NORMED);

    // 7.5 亚像素峰值提取
    std::vector<cv::Point2f> corners_frontal_refined;
    for (int i=0;i<corners_und_frontal.size();i++)
    {
        cv::Point2f pt = corners_und_frontal[i];
        cv::Rect roi(pt.x - img_single_pattern.cols/2 - 25, pt.y - img_single_pattern.rows/2 - 25, 51, 51);
        cv::Point2f corners_max = find_mat_max_subpixel(result(roi));
        corners_frontal_refined.push_back(
            cv::Point2f(
                roi.x + img_single_pattern.cols/2.0 + corners_max.x, 
                roi.y + img_single_pattern.rows/2.0 + corners_max.y));
    }

    // 8. 将正平面控制点转换回原图像中的坐标
    // 8.1 透视变换回来
    std::vector<cv::Point2f> corners_refined_without_distortion;
    cv::perspectiveTransform(corners_frontal_refined, corners_refined_without_distortion, H.inv());

    // 8.2 畸变加回来
    std::vector<cv::Point3f> corners_refined_without_distortion_3d(corners_refined_without_distortion.size());
    cv::Mat Knew =cv::getOptimalNewCameraMatrix(K, D, image.size(), 1);
    double fx = Knew.at<double>(0,0);
    double fy = Knew.at<double>(1,1);
    double cx = Knew.at<double>(0,2);
    double cy = Knew.at<double>(1,2);
    for (int i=0;i<corners_refined_without_distortion.size();i++)
    {
        double x = corners_refined_without_distortion[i].x;
        double y = corners_refined_without_distortion[i].y;
        corners_refined_without_distortion_3d[i].x = (x-cx) / fx;
        corners_refined_without_distortion_3d[i].y = (y-cy) / fy;
        corners_refined_without_distortion_3d[i].z = 1;
    }

    std::vector<cv::Point2f> corners_refined;
    cv::projectPoints(corners_refined_without_distortion_3d, cv::Vec3f(0,0,0), cv::Vec3f(0,0,0), K, D, corners_refined);

    // cv::Mat show = img_pattern.clone();
    // cv::cvtColor(show, show, cv::COLOR_GRAY2RGB);
    // for (int i=0;i<corners.size();i++)
    // {
    //     // cv::circle(show, corners[i], 16, cv::Scalar(0, 255, 0), 4);
    //     cv::circle(show, corners_und_frontal[i], 16, cv::Scalar(0, 255, 0), 4);
    //     cv::circle(show, corners_frontal_refined[i], 16, cv::Scalar(255, 0, 0), 4);
    //     //std::cout << corners[i] << " -- " << corners_refined_without_distortion[i] << " -- " << corners_refined_without_distortion_3d[i] << " -- " << corners_refined[i] << std::endl;
    // }

    // cv::resize(show, show, show.size()/2);
    // cv::imshow("image", show);
    // cv::waitKey(0);

    corners = corners_refined;
}

/// @brief 生成标定板世界坐标
/// @param pattern_infos 标定图案信息
/// @param world_corners 角点世界坐标系下坐标
void high_accuracy_corner_detector::generate_world_corners(pattern_infos_t pattern_infos, std::vector<cv::Point3f> &world_corners)
{
    world_corners.clear();

    for (int y = 0; y < pattern_infos.size.height; y++)
    {
        for (int x = 0; x < pattern_infos.size.width; x++)
        {
            cv::Point3f pt(x * pattern_infos.distance, y * pattern_infos.distance, 0);
            world_corners.push_back(pt);
        }
    }
}

