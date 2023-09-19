/**
 * Copyright @2023 Sharemon. All rights reserved.
 *
 @author: sharemon
 @date: 2023-09-19
 */

#if !defined(__HELPER_H__)
#define __HELPER_H__

#include <getopt.h>
#include <iostream>
#include "../src/type.hpp"

#define CALIB_IMAGE_NUM_MIN (6)

/// @brief 应用执行模型
enum op_mode_t
{
    generate_calibration_board = 0, // 生成标定板图案模式
    single_camera_calibrate,        // 单目标定模式
    stereo_camera_calibrate,        // 双目标定模式
    simulate_calibration_image,     // 仿真标定图像
};

/// @brief 解析命令行参数
/// @param argc 命令行参数数量
/// @param argv 命令行参数内容
/// @param mode 应用模式
/// @param pattern_infos 标定图案信息
/// @param image_folder 图像文件夹
void parse_args(int argc, char *argv[], op_mode_t &mode, high_accuracy_corner_detector::pattern_infos_t &pattern_infos, std::string &image_folder)
{
    int opt = 0;
    const char *opt_string = "";
    int opt_indx = 0;
    static struct option long_options[] = {
        {"mode", required_argument, NULL, 'm'},
        {"pattern_width", required_argument, NULL, 'w'},
        {"pattern_height", required_argument, NULL, 'h'},
        {"pattern_distance", required_argument, NULL, 'd'},
        {"pattern_radius_inside", required_argument, NULL, 'i'},
        {"pattern_radius_outside", required_argument, NULL, 'o'},
        {"pattern_margin_size", required_argument, NULL, 'n'},
        {"pixel_size", required_argument, NULL, 'p'},
        {"image_folder", required_argument, NULL, 'f'}};

    while ((opt = getopt_long_only(argc, argv, opt_string, long_options, &opt_indx)) != -1)
    {
        printf("opt is %c, arg is %s\n", opt, optarg);
        switch (opt)
        {
        case 'm':
            if (std::string(optarg) == "single")
            {
                mode = op_mode_t::single_camera_calibrate;
            }
            else if (std::string(optarg) == "stereo")
            {
                mode = op_mode_t::stereo_camera_calibrate;
            }
            else if (std::string(optarg) == "generate")
            {
                mode = op_mode_t::generate_calibration_board;
            }
            else
            {
                mode = op_mode_t::simulate_calibration_image;
            }
            break;
        case 'w':
            pattern_infos.size.width = atof(optarg);
            break;
        case 'h':
            pattern_infos.size.height = atof(optarg);
            break;
        case 'd':
            pattern_infos.distance = atof(optarg);
            break;
        case 'i':
            pattern_infos.radius_inside = atof(optarg);
            break;
        case 'o':
            pattern_infos.radius_outside = atof(optarg);
            break;
        case 'n':
            pattern_infos.margin_size = atof(optarg);
            break;
        case 'p':
            pattern_infos.pixel_size = atof(optarg);
            break;
        case 'f':
            image_folder = std::string(optarg);
            break;
        default:
            printf("unknown option.\n");
            break;
        }
    }
}

/// @brief 寻找文件夹下所有标定图像
/// @param folder 文件夹
/// @param paths 标定图像
void find_all_images(const std::string &folder, std::vector<std::string> &paths)
{
    struct dirent *dptr;
    DIR *dir = opendir(folder.c_str());

    if (dir == NULL)
    {
        printf("the folder %s does not exist.\n", folder.c_str());
        return;
    }

    while ((dptr = readdir(dir)) != NULL)
    {
        // 过滤父目录和当前目录
        if (strcmp(dptr->d_name, ".") == 0 || strcmp(dptr->d_name, "..") == 0)
        {
            continue;
        }

        if (strstr(dptr->d_name, ".jpg") != NULL || strstr(dptr->d_name, ".png") != NULL || strstr(dptr->d_name, ".bmp") != NULL)
        {
            paths.push_back(folder + "/" + std::string(dptr->d_name));
        }
    }
}

/// @brief 计算两个点之间的距离
/// @param p1 点1
/// @param p2 点2
/// @return 距离
inline double calc_dist_between_3d_points(cv::Point3f p1, cv::Point3f p2)
{
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2));
}


/// @brief 计算均值和标准差
/// @param data 数据
/// @param mean 均值
/// @param stdev 标准差
void calc_mean_and_stdev(const std::vector<double>& data, double& mean, double& stdev)
{
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    mean = sum / data.size(); // 均值

    double accum = 0.0;
    std::for_each(data.begin(), data.end(), [&](const double d)
                  { accum += (d - mean) * (d - mean); });

    stdev = sqrt(accum / (data.size() - 1)); // 标准差
}


/// @brief 最小二乘拟合平面
/// @param pts 控制点三维坐标
/// @param plane 平面参数(a,b,c,d)
void fit_plane_LS(const std::vector<cv::Point3f> &pts, cv::Vec4d &plane)
{
    cv::Mat A = cv::Mat::ones(cv::Size(4, pts.size()), CV_64FC1);

    for (int i = 0; i < pts.size(); i++)
    {
        A.at<double>(i, 0) = pts[i].x;
        A.at<double>(i, 1) = pts[i].y;
        A.at<double>(i, 2) = pts[i].z;
    }

    cv::Mat w, u, vt;
    cv::SVDecomp(A, w, u, vt, cv::SVD::FULL_UV);

    plane[0] = vt.at<double>(vt.rows-1, 0);
    plane[1] = vt.at<double>(vt.rows-1, 1);
    plane[2] = vt.at<double>(vt.rows-1, 2);
    plane[3] = vt.at<double>(vt.rows-1, 3);
}


/// @brief 对双目标定结果进行评估
/// @param Kl 左目内参
/// @param Dl 左目畸变参数
/// @param Kr 右目内参
/// @param Dr 右目畸变
/// @param R 左右目旋转
/// @param T 左右目位移
/// @param left_img_pts 左目图像控制点
/// @param right_img_pts 右目图像控制点
/// @param pattern_infos 标定板参数
void evaluate_stereo_calib_result(
    const cv::Mat &Kl, const cv::Mat &Dl,
    const cv::Mat &Kr, const cv::Mat &Dr,
    const cv::Mat &R, const cv::Mat &T,
    const std::vector<std::vector<cv::Point2f>> &left_img_pts,
    const std::vector<std::vector<cv::Point2f>> &right_img_pts,
    const high_accuracy_corner_detector::pattern_infos_t &pattern_infos,
    const cv::Size& image_size)
{
    // 计算投影矩阵
    cv::Mat P1 = cv::Mat::zeros(cv::Size(4, 3), CV_64FC1);
    cv::Mat P2 = cv::Mat::zeros(cv::Size(4, 3), CV_64FC1);

    Kl.copyTo(P1(cv::Rect(0, 0, 3, 3)));
    cv::Mat P2_3x3 = Kr * R;
    cv::Mat P2_3x1 = Kr * T;
    P2_3x3.copyTo(P2(cv::Rect(0, 0, 3, 3)));
    P2_3x1.copyTo(P2(cv::Rect(3, 0, 1, 3)));

    std::cout << "P1:" << std::endl;
    std::cout << P1 << std::endl;
    std::cout << "P2:" << std::endl;
    std::cout << P2 << std::endl;

    // 计算标定板三维坐标
    std::vector<double> dists_between_pts;
    std::vector<double> dists_fitting_plane;
    for (int i = 0; i < left_img_pts.size(); i++)
    {
        // 去畸变
        std::vector<cv::Point2f> left_img_pts_und;
        std::vector<cv::Point2f> right_img_pts_und;
        cv::undistortImagePoints(left_img_pts[i], left_img_pts_und, Kl, Dl);
        cv::undistortImagePoints(right_img_pts[i], right_img_pts_und, Kl, Dl);

        // 三角化计算控制点三维坐标
        cv::Mat points_homo;
        cv::triangulatePoints(P1, P2, left_img_pts_und, right_img_pts_und, points_homo);

        // 存储三维坐标
        std::vector<cv::Point3f> pts3d(points_homo.cols);
        for (int j = 0; j < points_homo.cols; j++)
        {
            pts3d[j].x = points_homo.at<float>(0, j) / points_homo.at<float>(3, j);
            pts3d[j].y = points_homo.at<float>(1, j) / points_homo.at<float>(3, j);
            pts3d[j].z = points_homo.at<float>(2, j) / points_homo.at<float>(3, j);
        }

        // 计算一下每个点与相邻点距离
        for (int j = 0; j < pattern_infos.size.height; j++)
        {
            for (int k = 0; k < pattern_infos.size.width; k++)
            {
                if (k != pattern_infos.size.width - 1)
                {
                    double dist = abs(calc_dist_between_3d_points(pts3d[k + j * pattern_infos.size.width], pts3d[k + 1 + j * pattern_infos.size.width]) - 32);
                    dists_between_pts.push_back(dist);
                }

                if (j != pattern_infos.size.height - 1)
                {
                    double dist = abs(calc_dist_between_3d_points(pts3d[k + j * pattern_infos.size.width], pts3d[k + (j + 1) * pattern_infos.size.width]) - 32);
                    dists_between_pts.push_back(dist);
                }
            }
        }

        // 拟合平面
        cv::Vec4d plane_param;
        fit_plane_LS(pts3d, plane_param);

        // 计算每个点的偏差
        std::ofstream of("./calib_points_" + std::to_string(i) + ".txt", std::ios_base::out);
        for (int j = 0; j < pts3d.size(); j++)
        {
            double dist = plane_param[0] * pts3d[j].x + plane_param[1] * pts3d[j].y + plane_param[2] * pts3d[j].z + plane_param[3];
            dists_fitting_plane.push_back(abs(dist));

            of << pts3d[j].x << "," << pts3d[j].y << "," << pts3d[j].z << "," << dist << std::endl;
        }
        of.close();
    }

    // 计算距离均值和标准差
    double mean_between_pts = 0, stdev_between_pts = 0;
    calc_mean_and_stdev(dists_between_pts, mean_between_pts, stdev_between_pts);

    std::cout << "distance size: " << dists_between_pts.size() << std::endl;
    std::cout << "distance average: " << mean_between_pts << std::endl;
    std::cout << "distance stdev: " << stdev_between_pts << std::endl;

    double mean_fitting_plane = 0, stdev_fitting_plane = 0;
    calc_mean_and_stdev(dists_fitting_plane, mean_fitting_plane, stdev_fitting_plane);

    std::cout << "distance size: " << dists_fitting_plane.size() << std::endl;
    std::cout << "distance average: " << mean_fitting_plane << std::endl;
    std::cout << "distance stdev: " << stdev_fitting_plane << std::endl;

    // 计算eame误差
    cv::Mat R1, R2, Q;
    cv::stereoRectify(Kl, Dl, Kr, Dr, image_size, R, T, R1, R2, P1, P2, Q);

    std::vector<double> eames;
    for (int i = 0; i < left_img_pts.size(); i++) 
    {
        std::vector<cv::Point2f> left_img_pts_und;
        std::vector<cv::Point2f> right_img_pts_und;
        cv::undistortPoints(left_img_pts[i], left_img_pts_und, Kl, Dl, R1, P1);
        cv::undistortPoints(right_img_pts[i], right_img_pts_und, Kr, Dr, R2, P2);

        for (int j = 0; j < left_img_pts_und.size(); j++)
        {
            eames.push_back(abs(left_img_pts_und[j].y - right_img_pts_und[j].y));
        }
    }

    double mean_eame = 0, stdev_eame = 0;
    calc_mean_and_stdev(eames, mean_eame, stdev_eame);

    std::cout << "distance size: " << eames.size() << std::endl;
    std::cout << "distance average: " << mean_eame << std::endl;
    std::cout << "distance stdev: " << stdev_eame << std::endl;
}


/// @brief 评估双目参数用于立体匹配的结果
/// @param left_image 左图像
/// @param right_image 右图像
/// @param Kl 左相机内参
/// @param Dl 左相机畸变
/// @param Kr 右相机内参
/// @param Dr 右相机畸变
/// @param R 左右相机旋转
/// @param T 左右相机位移
void evaluate_stereo_match(
    const cv::Mat& left_image, 
    const cv::Mat& right_image, 
    const cv::Mat& Kl, const cv::Mat& Dl, 
    const cv::Mat& Kr, const cv::Mat& Dr, 
    const cv::Mat& R, const cv::Mat& T,
    cv::Mat& image_rectified,
    cv::Mat& disp)
{
    cv::Mat P1, P2, R1, R2, Q;
    cv::stereoRectify(Kl, Dl ,Kr, Dr, left_image.size(), R, T, R1, R2, P1, P2, Q);

    cv::Mat left_mapx, left_mapy, right_mapx, right_mapy;
    cv::initUndistortRectifyMap(Kl, Dl, R1, P1, left_image.size(), CV_32FC1, left_mapx, left_mapy);
    cv::initUndistortRectifyMap(Kr, Dr, R2, P2, right_image.size(), CV_32FC1, right_mapx, right_mapy);

    cv::Mat left_image_rectified, right_image_rectified;
    cv::remap(left_image, left_image_rectified, left_mapx, left_mapy, cv::INTER_LINEAR);
    cv::remap(right_image, right_image_rectified, right_mapx, right_mapy, cv::INTER_LINEAR);

    image_rectified = cv::Mat::zeros(cv::Size(left_image.cols * 2, left_image.rows), CV_8UC1);
    left_image_rectified.copyTo(image_rectified.colRange(0, image_rectified.cols/2));
    right_image_rectified.copyTo(image_rectified.colRange(image_rectified.cols/2, image_rectified.cols));

    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 256, 5, 8*5*5, 32*5*5, 1, 63, 10, 100, 1, cv::StereoSGBM::MODE_HH);
    cv::Mat disp_s16;
    sgbm->compute(left_image_rectified, right_image_rectified, disp_s16);

    disp_s16.convertTo(disp, CV_8UC1, 1.0/16);
}

#endif // __HELPER_H__
