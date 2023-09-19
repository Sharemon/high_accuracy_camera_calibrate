/**
 * Copyright @2023 Sharemon. All rights reserved.
 *
 @author: sharemon
 @date: 2023-09-07
 */

#include "generate_pattern.hpp"

using namespace high_accuracy_corner_detector;

#define CIRCLE_SUBPIX_NUM   (17)

/// @brief 生成标定板图像
/// @param pattern_infos 标定图案信息
/// @param board_image 生成图像结果
void high_accuracy_corner_detector::generate_calibration_board(pattern_infos_t pattern_infos, cv::Mat &board_image)
{
    int width = (pattern_infos.margin_size * 2 + pattern_infos.distance * (pattern_infos.size.width - 1)) / pattern_infos.pixel_size;
    int height = (pattern_infos.margin_size * 2 + pattern_infos.distance * (pattern_infos.size.height - 1)) / pattern_infos.pixel_size;

    board_image = cv::Mat(height, width, CV_8UC1, cv::Scalar(255));

#if 0
    // 直接画圆，结果比较粗糙
    for (int r=0; r < pattern_infos.size.height; r++)
    {
        for (int c=0; c < pattern_infos.size.width; c++)
        {
            cv::circle( board_image, 
                        cv::Point(
                            (pattern_infos.margin_size + c * pattern_infos.distance) / pattern_infos.pixel_size, 
                            (pattern_infos.margin_size + r * pattern_infos.distance) / pattern_infos.pixel_size),
                        pattern_infos.radius_outside / pattern_infos.pixel_size,
                        cv::Scalar(0),
                        -1);
        }
    }
#else
    // 精细化圆点图像生成
    for (int r=0; r < pattern_infos.size.height; r++)
    {
        for (int c=0; c < pattern_infos.size.width; c++)
        {
            // 获取每个圆圆心位置
            double cx = (pattern_infos.margin_size + c * pattern_infos.distance) / pattern_infos.pixel_size;
            double cy = (pattern_infos.margin_size + r * pattern_infos.distance) / pattern_infos.pixel_size;
            double radius_in_pixel = pattern_infos.radius_outside / pattern_infos.pixel_size;

            // 获取每个圆点附近的区域
            int x_start = floor(cx - radius_in_pixel);
            int x_end = ceil(cx + radius_in_pixel);
            int y_start = floor(cy - radius_in_pixel);
            int y_end = ceil(cy + radius_in_pixel);

            // 对区域内每个像素细分处理
            for (int y = y_start; y < y_end; y++)
            {
                for(int x = x_start; x < x_end; x++)
                {
                    int sum = 0;

                    // 1. 将每个像素分成5x5=25个小区域
                    for (int y_divide = -(CIRCLE_SUBPIX_NUM/2); y_divide <= (CIRCLE_SUBPIX_NUM/2); y_divide++)
                    {
                        for (int x_divide = -(CIRCLE_SUBPIX_NUM/2); x_divide <= (CIRCLE_SUBPIX_NUM/2); x_divide++)
                        {
                            // 2. 判断25个区域中有多少区域位于圆内
                            double subpix_x = (x + (double)x_divide / CIRCLE_SUBPIX_NUM);
                            double subpix_y = (y + (double)y_divide / CIRCLE_SUBPIX_NUM);

                            if (pow((subpix_x - cx), 2) + pow(subpix_y - cy, 2) < pow(radius_in_pixel, 2))
                            {
                                sum ++;
                            } 
                        }
                    }

                    // 3. 按比例对该像素的灰度值进行赋值
                    board_image.at<uchar>(y, x) = (uchar)(255 - (double)sum / (CIRCLE_SUBPIX_NUM * CIRCLE_SUBPIX_NUM) * 255);
                }
            }
        }
    }
#endif
}

/// @brief 生成单个标定图案，用于模板匹配
/// @param pattern_infos 标定图案信息
/// @param pattern_image 生成图像结果
void high_accuracy_corner_detector::generate_single_pattern(pattern_infos_t pattern_infos, cv::Mat &pattern_image)
{
    const int margin = 10;
    int width = (int)(2 * (margin + pattern_infos.radius_outside / pattern_infos.pixel_size));
    int height = width;
    pattern_image = cv::Mat(height, width, CV_8UC1);

    // 获取每个圆圆心位置
    double cx = width / 2.0;
    double cy = height / 2.0;
    double radius_in_pixel = pattern_infos.radius_outside / pattern_infos.pixel_size;

    // 对区域内每个像素细分处理
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int sum = 0;

            // 1. 将每个像素分成5x5=25个小区域
            for (int y_divide = -(CIRCLE_SUBPIX_NUM / 2); y_divide <= (CIRCLE_SUBPIX_NUM / 2); y_divide++)
            {
                for (int x_divide = -(CIRCLE_SUBPIX_NUM / 2); x_divide <= (CIRCLE_SUBPIX_NUM / 2); x_divide++)
                {
                    // 2. 判断25个区域中有多少区域位于圆内
                    double subpix_x = (x + (double)x_divide / CIRCLE_SUBPIX_NUM);
                    double subpix_y = (y + (double)y_divide / CIRCLE_SUBPIX_NUM);

                    if (pow((subpix_x - cx), 2) + pow(subpix_y - cy, 2) < pow(radius_in_pixel, 2))
                    {
                        sum++;
                    }
                }
            }

            // 3. 按比例对该像素的灰度值进行赋值
            pattern_image.at<uchar>(y, x) = (uchar)(255 - (double)sum / (CIRCLE_SUBPIX_NUM * CIRCLE_SUBPIX_NUM) * 255);
        }
    }
}