/**
 * Copyright @2023 Sharemon. All rights reserved.
 * 
 @author: sharemon
 @date: 2023-09-07
 */

#if !defined(__TYPE_H__)
#define __TYPE_H__

#include <opencv2/opencv.hpp>

namespace high_accuracy_corner_detector
{
    /// @brief 标定图案样式
    enum pattern_type_t
    {
        chessboard = 0,         // 棋盘格
        circle,                 // 圆点
        ring                    // 圆环
    };

    /// @brief 标定图案信息
    struct pattern_infos_t
    {
        pattern_type_t type;    // 标定图案样式
        cv::Size size;          // 控制点尺寸(单位：个)
        double distance;        // 控制点间距(单位：mm)
        double radius_inside;   // 对于环形控制点图像：圆环内径(单位：mm)
        double radius_outside;  // 对于环形控制点图像：圆环外径，对于圆点图形：圆点半径(单位：mm)
        double margin_size;     // 图形边缘尺寸(单位：mm)
        double pixel_size;      // 显示器像素尺寸，用于将真实尺寸转换为像素尺寸(单位：mm)
    };
}

#endif // __TYPE_H__

