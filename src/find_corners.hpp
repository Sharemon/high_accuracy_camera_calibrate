/**
 * Copyright @2023 Sharemon. All rights reserved.
 * 
 @author: sharemon
 @date: 2023-09-07
 */

#if !defined(__FIND_CORNERS_H__)
#define __FIND_CORNERS_H__

#include "type.hpp"
#include <vector>

namespace high_accuracy_corner_detector
{
    /// @brief 从图像中找控制点
    /// @param image 图像数据
    /// @param corners 控制点结果
    /// @param pattern_infos 标定图案信息
    void find_corners(const cv::Mat &image, std::vector<cv::Point2f> &corners, const pattern_infos_t pattern_infos);

    /// @brief 优化控制点
    /// @param image 图像数据
    /// @param corners 控制点结果
    /// @param pattern_infos 标定图案信息
    /// @param K 初步标定得到的内参
    /// @param D 初步标定得到的畸变参数
    /// @param rvecs 初步标定得到的标定板旋转
    /// @param tvecs 初步标定得到的标定板位移
    void refine_corners(const cv::Mat &image, std::vector<cv::Point2f> &corners, const pattern_infos_t pattern_infos, 
                        const cv::Mat& K, const cv::Mat& D);

    /// @brief 生成标定板世界坐标
    /// @param pattern_infos 标定图案信息
    /// @param world_corners 角点世界坐标系下坐标
    void generate_world_corners(pattern_infos_t pattern_infos, std::vector<cv::Point3f>& world_corners);

} // namespace high_accuracy_corner_detector


#endif // __FIND_CORNERS_H__
