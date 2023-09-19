/**
 * Copyright @2023 Sharemon. All rights reserved.
 * 
 @author: sharemon
 @date: 2023-09-07
 */

#if !defined(__GENERATE_PATTERN_H__)
#define __GENERATE_PATTERN_H__

#include "type.hpp"

namespace high_accuracy_corner_detector
{
    /// @brief 生成标定板图像
    /// @param pattern_infos 标定图案信息
    /// @param board_image 生成图像结果
    void generate_calibration_board(pattern_infos_t pattern_infos, cv::Mat& board_image);
    
    /// @brief 生成单个标定图案，用于模板匹配
    /// @param pattern_infos 标定图案信息
    /// @param pattern_image 生成图像结果
    void generate_single_pattern(pattern_infos_t pattern_infos, cv::Mat& pattern_image);
}

#endif // __GENERATE_PATTERN_H__


