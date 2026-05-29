#pragma once

#include <string>

// 将中面网格映射回原始 surface Face ID，并把中面中的四边形拆成三角形。
//
// 输入：
// - surfaceMeshPath: 原始带 Face ID 的三角表面网格 vtk 文件。
// - volumeMeshPath: 内部四面体体网格 vtk 文件。
// - midSurfacePath: 已生成的中面网格 vtk 文件，可包含三角形和四边形。
// - outputPath: 输出全三角形中面网格 vtk 文件。
//
// tolerance <= 0 时会按模型包围盒自动估计容差。
// projectionThreshold 为投影面积比例阈值，范围0~1
// dihedralWeight 控制二面角在边界候选面评分中的权重，范围0~1。
bool MapMidSurfaceIds(const std::string& surfaceMeshPath,
                      const std::string& volumeMeshPath,
                      const std::string& midSurfacePath,
                      const std::string& outputPath,
                      double tolerance = -1.0,
                      double projectionThreshold = 0.70,
                      double dihedralWeight = 0.35);
