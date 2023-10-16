/**
 * Copyright @2023 Sharemon. All rights reserved.
 *
 @author: sharemon
 @date: 2023-09-07
 */

#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <fstream>
#include <numeric>

#include "../src/generate_pattern.hpp"
#include "../src/find_corners.hpp"
#include "helper.hpp"

using namespace high_accuracy_corner_detector;
using namespace std;

/// @brief 生成标定图案并保存
/// @param pattern_infos 标定图案信息
/// @param image_folder 保存文件夹
void generate_board(const pattern_infos_t &pattern_infos, const string &image_folder)
{
    cv::Mat calibration_board_image;
    high_accuracy_corner_detector::generate_calibration_board(pattern_infos, calibration_board_image);

    cv::imwrite(image_folder + "/board.png", calibration_board_image);
}

/// @brief 生成虚拟的标定图案
/// @param pattern_infos 标定图案信息
/// @param image_folder 保存文件夹
void simulate_image(const pattern_infos_t &pattern_infos, const string &image_folder)
{
    // 定义虚拟相机参数
    const int img_w = 1920;
    const int img_h = 1080;
    const float f = 1000;
    const float cx = img_w / 2;
    const float cy = img_h / 2;

    cv::Size img_size(img_w, img_h);
    cv::Mat K = (cv::Mat_<double>(3, 3) << f, 0, cx,
                 0, f, cy,
                 0, 0, 1);

    cv::Mat D = cv::Mat::zeros(1, 5, CV_64FC1);

    // 定义标定板四个角点的世界坐标和图像坐标系
    double board_w = (pattern_infos.margin_size * 2 + pattern_infos.distance * (pattern_infos.size.width - 1)) / 1000;
    double board_h = (pattern_infos.margin_size * 2 + pattern_infos.distance * (pattern_infos.size.height - 1)) / 1000;

    std::vector<cv::Vec3f> vertex_world = {
        cv::Vec3f(-board_w / 2, -board_h / 2, 0),
        cv::Vec3f(board_w / 2, -board_h / 2, 0),
        cv::Vec3f(board_w / 2, board_h / 2, 0),
        cv::Vec3f(-board_w / 2, board_h / 2, 0)};

    // 生成正平面的标定图像
    cv::Mat pattern_img;
    high_accuracy_corner_detector::generate_calibration_board(pattern_infos, pattern_img);

    std::vector<cv::Vec2f> vertex_pattern_img = {
        cv::Vec2f(0, 0),
        cv::Vec2f(pattern_img.cols - 1, 0),
        cv::Vec2f(pattern_img.cols - 1, pattern_img.rows - 1),
        cv::Vec2f(0, pattern_img.rows - 1)};

    // 定义9张标定图片的位移和旋转
    std::vector<cv::Vec3d> tvecs = {
        cv::Vec3d(0, 0, 0.5),
        cv::Vec3d(0, 0, 0.5),
        cv::Vec3d(0, 0, 0.5),
        cv::Vec3d(0, 0, 0.5),
        cv::Vec3d(0, 0, 0.5),
        cv::Vec3d(-0.2, 0, 0.5),
        cv::Vec3d(0.2, 0, 0.5),
        cv::Vec3d(0, -0.1, 0.5),
        cv::Vec3d(0, 0.1, 0.5)};

    std::vector<cv::Vec3d> rvecs = {
        cv::Vec3d(0, 0, 0),
        cv::Vec3d(-15.0 / 180 * CV_PI, 0, 0),
        cv::Vec3d(15.0 / 180 * CV_PI, 0, 0),
        cv::Vec3d(0, -15.0 / 180 * CV_PI, 0),
        cv::Vec3d(0, 15.0 / 180 * CV_PI, 0),
        cv::Vec3d(0, 0, 0),
        cv::Vec3d(0, 0, 0),
        cv::Vec3d(0, 0, 0),
        cv::Vec3d(0, 0, 0)};

    for (int i = 0; i < tvecs.size(); i++)
    {
        // 1. 3d-2d投影
        std::vector<cv::Vec2f> vertex_img;
        cv::projectPoints(vertex_world, rvecs[i], tvecs[i], K, D, vertex_img);

        // 2. 求解标定图像正平面到投影平面的H矩阵
        cv::Mat H = cv::findHomography(vertex_pattern_img, vertex_img);

        // 4. 投影生成标定图像
        cv::Mat img;
        cv::warpPerspective(pattern_img, img, H, img_size);

        cv::imwrite(image_folder + "/sim_" + std::to_string(i) + ".png", img);
    }
}

/// @brief 单目相机标定
/// @param pattern_infos 标定图案信息
/// @param image_folder 图像文件夹
void single_calibrate(const pattern_infos_t &pattern_infos, const string &image_folder)
{
    // 1. 在文件夹下找出所有图片
    std::vector<std::string> image_paths;
    find_all_images(image_folder, image_paths);

    // 2. 加载所有图片 & 寻找角点
    std::vector<cv::Mat> images;
    std::vector<std::vector<cv::Point3f>> object_pts;
    std::vector<std::vector<cv::Point2f>> corner_pts;
    cv::Size img_size;
    int valid_image_num = 0;

    for (auto image_path : image_paths)
    {
        cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
        img_size = image.size();

        std::vector<cv::Point2f> corners;
        high_accuracy_corner_detector::find_corners(image, corners, pattern_infos);
        // cv::findCirclesGrid(image, pattern_infos.size, corners);

        if (corners.size() == pattern_infos.size.width * pattern_infos.size.height)
        {
            corner_pts.push_back(corners);
            std::vector<cv::Point3f> points_3d;
            high_accuracy_corner_detector::generate_world_corners(pattern_infos, points_3d);

            object_pts.push_back(points_3d);

            images.push_back(image);

            valid_image_num++;
        }
    }

    printf("get %d/%d valid images with corners.\n", valid_image_num, (int)image_paths.size());
    if (valid_image_num < CALIB_IMAGE_NUM_MIN)
    {
        printf("there is no enough valid images\n");
        return;
    }

    // 3. 第一次标定
    cv::Mat K, D, r, t;
    int flag = 0;
    double rms = cv::calibrateCamera(object_pts, corner_pts, img_size, K, D, r, t, flag);

    std::cout << "1st calibration result: " << std::endl;
    std::cout << "rms: " << rms << std::endl;
    std::cout << "K: " << std::endl;
    std::cout << K << std::endl;
    std::cout << "D: " << std::endl;
    std::cout << D << std::endl;

#if 1
    // 4. 迭代标定3次
    const int max_iterations = 10;
    // 4.1. 左相机
    for (int iter = 0; iter < max_iterations; iter++)
    {
        // 控制点迭代
        for (int i = 0; i < images.size(); i++)
        {
            high_accuracy_corner_detector::refine_corners(images[i], corner_pts[i], pattern_infos, K, D);
        }

        // 第二次标定
        flag = flag | cv::CALIB_USE_INTRINSIC_GUESS;
        rms = cv::calibrateCamera(object_pts, corner_pts, img_size, K, D, r, t, flag);

        std::cout << iter + 1 << "th calibration result: " << std::endl;
        std::cout << "rms: " << rms << std::endl;
        std::cout << "K: " << std::endl;
        std::cout << K << std::endl;
        std::cout << "D: " << std::endl;
        std::cout << D << std::endl;
    }
#endif
}

/// @brief 双目相机标定
/// @param pattern_infos 标定图案信息
/// @param image_folder 图像文件夹
void stereo_calibrate(const pattern_infos_t &pattern_infos, const string &image_folder)
{
    // 1. 在文件夹下找出所有图片
    std::vector<std::string> image_paths;
    find_all_images(image_folder, image_paths);

    // 2. 加载所有图片 & 寻找角点
    std::vector<cv::Mat> left_images;
    std::vector<cv::Mat> right_images;
    std::vector<std::vector<cv::Point3f>> object_pts;
    std::vector<std::vector<cv::Point3f>> object_pts_for_left_opt;
    std::vector<std::vector<cv::Point3f>> object_pts_for_right_opt;
    std::vector<std::vector<cv::Point2f>> left_corner_pts;
    std::vector<std::vector<cv::Point2f>> left_corner_pts_for_opt;
    std::vector<std::vector<cv::Point2f>> right_corner_pts;
    std::vector<std::vector<cv::Point2f>> right_corner_pts_for_opt;
    cv::Size img_size;
    int valid_image_num = 0;
    int left_valid_image_num = 0;
    int right_valid_image_num = 0;

    for (auto image_path : image_paths)
    {
        // 2.1. 将图片分解为左右两副图像
        cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
        cv::Mat left_image = image.colRange(0, image.cols / 2).clone();
        cv::Mat right_image = image.colRange(image.cols / 2, image.cols).clone();

        img_size = left_image.size();

        // 2.2. 分别对左右两幅图片提取角点
        std::vector<cv::Point2f> left_corners;
        std::vector<cv::Point2f> right_corners;
        high_accuracy_corner_detector::find_corners(left_image, left_corners, pattern_infos);
        high_accuracy_corner_detector::find_corners(right_image, right_corners, pattern_infos);

        // 2.3. 如果角点数量OK，则添加一组标定板三位点
        if (left_corners.size() == right_corners.size() && right_corners.size() == pattern_infos.size.width * pattern_infos.size.height)
        {
            left_corner_pts.push_back(left_corners);
            right_corner_pts.push_back(right_corners);

            std::vector<cv::Point3f> points_3d;
            high_accuracy_corner_detector::generate_world_corners(pattern_infos, points_3d);

            object_pts.push_back(points_3d);

            valid_image_num++;
        }
        
        if (left_corners.size() == pattern_infos.size.width * pattern_infos.size.height)
        {
            left_corner_pts_for_opt.push_back(left_corners);

            std::vector<cv::Point3f> points_3d;
            high_accuracy_corner_detector::generate_world_corners(pattern_infos, points_3d);

            object_pts_for_left_opt.push_back(points_3d);

            left_images.push_back(left_image);
            left_valid_image_num ++;
        }
        
        if (right_corners.size() == pattern_infos.size.width * pattern_infos.size.height)
        {
            right_corner_pts_for_opt.push_back(right_corners);

            std::vector<cv::Point3f> points_3d;
            high_accuracy_corner_detector::generate_world_corners(pattern_infos, points_3d);

            object_pts_for_right_opt.push_back(points_3d);

            right_images.push_back(right_image);
            right_valid_image_num ++;
        }
    }

    printf("left get %d/%d valid images with corners.\n", left_valid_image_num, (int)image_paths.size());
    printf("right get %d/%d valid images with corners.\n", right_valid_image_num, (int)image_paths.size());
    printf("get %d/%d valid images with corners.\n", valid_image_num, (int)image_paths.size());

    if (valid_image_num < CALIB_IMAGE_NUM_MIN || left_valid_image_num < CALIB_IMAGE_NUM_MIN || right_valid_image_num < CALIB_IMAGE_NUM_MIN)
    {
        printf("there is no enough valid images\n");
        return;
    }

    // 3. 第一次标定
    int flag = cv::CALIB_RATIONAL_MODEL;
    double rms = 0;
    // 3.1. 左相机
    cv::Mat Kl, Dl, rvecsl, tvecsl;
    // Kl = cv::Mat::zeros(3,3, CV_64FC1);
    // Dl = cv::Mat::zeros(1,8, CV_64FC1);
    // Kl.at<double>(0,0) = 1000;
    // Kl.at<double>(1,1) = 1000;
    // Kl.at<double>(0,2) = img_size.width / 2.0;
    // Kl.at<double>(1,2) = img_size.height / 2.0;
    // Kl.at<double>(2,2) = 1;
    // rms = cv::calibrateCamera(object_pts_for_left_opt, left_corner_pts_for_opt, img_size, Kl, Dl, rvecsl, tvecsl, flag | cv::CALIB_USE_INTRINSIC_GUESS);
    rms = cv::calibrateCamera(object_pts_for_left_opt, left_corner_pts_for_opt, img_size, Kl, Dl, rvecsl, tvecsl, flag);

    std::cout << "================================================" << std::endl;
    std::cout << "initial left camera calibration result: " << std::endl;
    std::cout << "rms: " << rms << std::endl;
    std::cout << "K: " << std::endl;
    std::cout << Kl << std::endl;
    std::cout << "D: " << std::endl;
    std::cout << Dl << std::endl;

    // 3.1. 右相机
    cv::Mat Kr, Dr, rvecsr, tvecsr;
    Kr = cv::Mat::zeros(3, 3, CV_64FC1);
    Dr = cv::Mat::zeros(1,8, CV_64FC1);
    // Kr.at<double>(0,0) = 1000;
    // Kr.at<double>(1,1) = 1000;
    // Kr.at<double>(0,2) = img_size.width / 2.0;
    // Kr.at<double>(1,2) = img_size.height / 2.0;
    // Kr.at<double>(2,2) = 1;
    // rms = cv::calibrateCamera(object_pts_for_right_opt, right_corner_pts_for_opt, img_size, Kr, Dr, rvecsr, tvecsr,  flag | cv::CALIB_USE_INTRINSIC_GUESS);
    rms = cv::calibrateCamera(object_pts_for_right_opt, right_corner_pts_for_opt, img_size, Kr, Dr, rvecsr, tvecsr,  flag);

    std::cout << "================================================" << std::endl;
    std::cout << "initial right camera calibration result: " << std::endl;
    std::cout << "rms: " << rms << std::endl;
    std::cout << "K: " << std::endl;
    std::cout << Kr << std::endl;
    std::cout << "D: " << std::endl;
    std::cout << Dr << std::endl;

#if 1
    // 4. 迭代标定3次
    const int max_iterations = 10;
    // 4.1. 左相机
    for (int iter = 0; iter < max_iterations; iter++)
    {
        // 控制点迭代
        for (int i = 0; i < left_images.size(); i++)
        {
            high_accuracy_corner_detector::refine_corners(left_images[i], left_corner_pts_for_opt[i], pattern_infos, Kl, Dl);
        }

        // 第二次标定
        rms = cv::calibrateCamera(object_pts_for_left_opt, left_corner_pts_for_opt, img_size, Kl, Dl, rvecsl, tvecsl, flag | cv::CALIB_USE_INTRINSIC_GUESS);

        std::cout << "================================================" << std::endl;
        std::cout << iter << "th left camera calibration result: " << std::endl;
        std::cout << "rms: " << rms << std::endl;
        std::cout << "K: " << std::endl;
        std::cout << Kl << std::endl;
        std::cout << "D: " << std::endl;
        std::cout << Dl << std::endl;
    }

    // 4.2. 右相机
    for (int iter = 0; iter < max_iterations; iter++)
    {
        // 控制点迭代
        for (int i = 0; i < right_images.size(); i++)
        {
            high_accuracy_corner_detector::refine_corners(right_images[i], right_corner_pts_for_opt[i], pattern_infos, Kr, Dr);
        }

        // 第二次标定
        rms = cv::calibrateCamera(object_pts_for_right_opt, right_corner_pts_for_opt, img_size, Kr, Dr, rvecsl, tvecsl, flag | cv::CALIB_USE_INTRINSIC_GUESS);

        std::cout << "================================================" << std::endl;
        std::cout << iter << "th right camera calibration result: " << std::endl;
        std::cout << "rms: " << rms << std::endl;
        std::cout << "K: " << std::endl;
        std::cout << Kr << std::endl;
        std::cout << "D: " << std::endl;
        std::cout << Dr << std::endl;
    }

    // 5. 双目标定
    cv::Mat R, T, E, F;
    // rms = cv::stereoCalibrate(object_pts, left_corner_pts, right_corner_pts, Kl, Dl, Kr, Dr, img_size, R, T, E, F, flag | cv::CALIB_FIX_INTRINSIC);
    rms = cv::stereoCalibrate(object_pts, left_corner_pts, right_corner_pts, Kl, Dl, Kr, Dr, img_size, R, T, E, F, flag | cv::CALIB_USE_INTRINSIC_GUESS);
#else
    // 5. 双目标定
    cv::Mat R, T, E, F;
    rms = cv::stereoCalibrate(object_pts, left_corner_pts, right_corner_pts, Kl, Dl, Kr, Dr, img_size, R, T, E, F, cv::CALIB_USE_INTRINSIC_GUESS);
#endif

    std::cout << "================================================" << std::endl;
    std::cout << "stereo camera calibration result: " << std::endl;
    std::cout << "rms: " << rms << std::endl;
    std::cout << "Kl: " << std::endl;
    std::cout << Kl << std::endl;
    std::cout << "Dl: " << std::endl;
    std::cout << Dl << std::endl;
    std::cout << "kr: " << std::endl;
    std::cout << Kr << std::endl;
    std::cout << "Dr: " << std::endl;
    std::cout << Dr << std::endl;
    std::cout << "R: " << std::endl;
    std::cout << R << std::endl;
    std::cout << "T: " << std::endl;
    std::cout << T << std::endl;

    // 存储标定得到的结果
    cv::FileStorage param_savefile("./stereo_params.yaml", cv::FileStorage::WRITE);

    param_savefile << "rms" << rms;
    param_savefile << "Kl" << Kl;
    param_savefile << "Dl" << Dl;
    param_savefile << "Kr" << Kr;
    param_savefile << "Dr" << Dr;
    param_savefile << "R" << R;
    param_savefile << "T" << T;

    // 6. 反算标定板三维坐标
    evaluate_stereo_calib_result(Kl, Dl, Kr, Dr, R, T, left_corner_pts, right_corner_pts, pattern_infos, img_size);

    // 7. 验证双目立体匹配结果
#if 0
    for (int i=0;i<left_images.size();i++)
    {
        cv::Mat disp, rectified;
        evaluate_stereo_match(left_images[i], right_images[i], Kl, Dl, Kr, Dr, R, T, rectified, disp);

        cv::Mat disp_colormap;
        cv::applyColorMap(disp, disp_colormap, cv::COLORMAP_JET);

        cv::imwrite("rectified_" + std::to_string(i) + ".png", rectified);
        cv::imwrite("disp_" + std::to_string(i) + ".png", disp_colormap);
    }
#endif
}

int main(int argc, char *argv[])
{
    op_mode_t mode = op_mode_t::generate_calibration_board;
    pattern_infos_t infos = {};
    string image_folder = "./";

    // 解析参数
    parse_args(argc, argv, mode, infos, image_folder);

    switch (mode)
    {
    case op_mode_t::generate_calibration_board:
        generate_board(infos, image_folder);
        break;
    case op_mode_t::single_camera_calibrate:
        single_calibrate(infos, image_folder);
        break;
    case op_mode_t::stereo_camera_calibrate:
        stereo_calibrate(infos, image_folder);
        break;
    case op_mode_t::simulate_calibration_image:
        simulate_image(infos, image_folder);
    default:
        break;
    }

    return 0;
}
