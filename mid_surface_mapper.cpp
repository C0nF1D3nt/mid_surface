#include "mid_surface_mapper.h"

#include <vtkCell.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkDataSet.h>
#include <vtkDataSetReader.h>
#include <vtkIdList.h>
#include <vtkIntArray.h>
#include <vtkMath.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataWriter.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkUnstructuredGridReader.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

// -1 是需求里约定的“没有匹配到另一个面”的特殊 Face ID。
// kUnsetFaceId 只在内部表示“还没有被赋值”，最终不会写到结果里。
constexpr int kUnmatchedFaceId = -1;
constexpr int kUnsetFaceId = std::numeric_limits<int>::min();

// 这里不用 vtkVector3d，是为了让几何计算更直接、也减少 VTK 类型转换噪声。
struct Vec3 {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

Vec3 operator+(const Vec3& a, const Vec3& b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 operator-(const Vec3& a, const Vec3& b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 operator*(const Vec3& a, double s) {
  return {a.x * s, a.y * s, a.z * s};
}

double Dot(const Vec3& a, const Vec3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 Cross(const Vec3& a, const Vec3& b) {
  return {a.y * b.z - a.z * b.y,
          a.z * b.x - a.x * b.z,
          a.x * b.y - a.y * b.x};
}

double Norm(const Vec3& a) {
  return std::sqrt(Dot(a, a));
}

Vec3 Normalize(const Vec3& a) {
  const double n = Norm(a);
  if (n <= 0.0) {
    return {};
  }
  return a * (1.0 / n);
}

double TriangleArea(const Vec3& a, const Vec3& b, const Vec3& c) {
  return 0.5 * Norm(Cross(b - a, c - a));
}

double Clamp(double value, double low, double high) {
  return std::max(low, std::min(value, high));
}

Vec3 GetPoint(vtkPoints* points, vtkIdType id) {
  double p[3];
  points->GetPoint(id, p);
  return {p[0], p[1], p[2]};
}

Vec3 GetPoint(vtkDataSet* dataSet, vtkIdType id) {
  double p[3];
  dataSet->GetPoint(id, p);
  return {p[0], p[1], p[2]};
}

struct QuantKey {
  long long x = 0;
  long long y = 0;
  long long z = 0;

  bool operator==(const QuantKey& other) const {
    return x == other.x && y == other.y && z == other.z;
  }

  bool operator<(const QuantKey& other) const {
    if (x != other.x) {
      return x < other.x;
    }
    if (y != other.y) {
      return y < other.y;
    }
    return z < other.z;
  }
};

struct QuantKeyHash {
  std::size_t operator()(const QuantKey& key) const {
    std::size_t h = 1469598103934665603ull;
    auto mix = [&](long long value) {
      const auto v = static_cast<std::uint64_t>(value);
      h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    };
    mix(key.x);
    mix(key.y);
    mix(key.z);
    return h;
  }
};

class Quantizer {
 public:
  explicit Quantizer(double tolerance) : tolerance_(std::max(tolerance, 1e-12)) {}

  // 把浮点坐标按 tolerance 量化成整数格点。
  // 这样可以用哈希表匹配“理论上相等，但有微小浮点误差”的点。
  QuantKey operator()(const Vec3& p) const {
    return {static_cast<long long>(std::llround(p.x / tolerance_)),
            static_cast<long long>(std::llround(p.y / tolerance_)),
            static_cast<long long>(std::llround(p.z / tolerance_))};
  }

  double tolerance() const { return tolerance_; }

 private:
  double tolerance_;
};

struct TriangleKey {
  std::array<QuantKey, 3> points;

  bool operator==(const TriangleKey& other) const {
    return points == other.points;
  }
};

struct TriangleKeyHash {
  std::size_t operator()(const TriangleKey& key) const {
    QuantKeyHash hash;
    std::size_t h = 0;
    for (const auto& point : key.points) {
      h ^= hash(point) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    }
    return h;
  }
};

TriangleKey MakeTriangleKey(std::array<QuantKey, 3> points) {
  std::sort(points.begin(), points.end());
  return {points};
}

struct EdgeKey {
  vtkIdType a = -1;
  vtkIdType b = -1;

  bool operator==(const EdgeKey& other) const {
    return a == other.a && b == other.b;
  }
};

struct EdgeKeyHash {
  std::size_t operator()(const EdgeKey& key) const {
    return std::hash<vtkIdType>{}(key.a) ^
           (std::hash<vtkIdType>{}(key.b) + 0x9e3779b97f4a7c15ull);
  }
};

EdgeKey MakeEdgeKey(vtkIdType a, vtkIdType b) {
  if (b < a) {
    std::swap(a, b);
  }
  return {a, b};
}

struct FacePair {
  int first = kUnsetFaceId;
  int second = kUnsetFaceId;

  bool operator==(const FacePair& other) const {
    return first == other.first && second == other.second;
  }

  bool operator!=(const FacePair& other) const {
    return !(*this == other);
  }

  bool operator<(const FacePair& other) const {
    if (first != other.first) {
      return first < other.first;
    }
    return second < other.second;
  }
};

struct FacePairHash {
  std::size_t operator()(const FacePair& pair) const {
    return std::hash<int>{}(pair.first) ^
           (std::hash<int>{}(pair.second) + 0x9e3779b97f4a7c15ull);
  }
};

FacePair NormalizePair(FacePair pair) {
  if (pair.first == kUnsetFaceId || pair.second == kUnsetFaceId) {
    return {kUnmatchedFaceId, kUnmatchedFaceId};
  }
  // Face ID pair 表示的是“两个面的关系”，不关心顺序。
  // 统一排序后，(3, 7) 和 (7, 3) 会映射到同一个 surface_id。
  if (pair.second < pair.first) {
    std::swap(pair.first, pair.second);
  }
  return pair;
}

bool IsValidPair(FacePair pair) {
  pair = NormalizePair(pair);
  return !(pair.first == kUnmatchedFaceId && pair.second == kUnmatchedFaceId);
}

// 原始 surface_mesh 中的一个三角形及其 Face ID。
// Build() 时会把这些三角形放进多个索引，后面用来查：
// 1. 某个四面体面（三个顶点）属于哪个 Face ID；
// 2. 某个单点落在哪些 surface 三角形上。
struct SurfaceTriangle {
  vtkIdType cellId = -1;
  std::array<vtkIdType, 3> pointIds{};
  std::array<QuantKey, 3> pointKeys{};
  std::array<Vec3, 3> points{};
  Vec3 normal{};
  double area = 0.0;
  int faceId = kUnmatchedFaceId;
  double bounds[6]{};
};

struct SurfaceFaceMatch {
  int faceId = kUnmatchedFaceId;
  Vec3 normal{};
};

bool IsNameMatch(const std::string& name, const std::string& expected) {
  std::string lhs = name;
  std::string rhs = expected;
  std::transform(lhs.begin(), lhs.end(), lhs.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  std::transform(rhs.begin(), rhs.end(), rhs.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  lhs.erase(std::remove(lhs.begin(), lhs.end(), '_'), lhs.end());
  rhs.erase(std::remove(rhs.begin(), rhs.end(), '_'), rhs.end());
  return lhs == rhs;
}

vtkDataArray* FindFaceIdArray(vtkPolyData* surface) {
  vtkCellData* cellData = surface->GetCellData();
  if (!cellData) {
    return nullptr;
  }

  // 优先找常见名字；如果用户文件里名字不同，则退化为第一个单分量 CellData。
  const std::array<std::string, 10> preferredNames = {
      "face_id", "faceid", "FaceID", "FaceId",
      "surface_id", "surfaceid", "SurfaceID",
      "regions", "region", "id"};

  for (const auto& preferred : preferredNames) {
    for (int i = 0; i < cellData->GetNumberOfArrays(); ++i) {
      vtkDataArray* array = cellData->GetArray(i);
      if (!array || !array->GetName()) {
        continue;
      }
      if (array->GetNumberOfComponents() == 1 &&
          IsNameMatch(array->GetName(), preferred)) {
        return array;
      }
    }
  }

  for (int i = 0; i < cellData->GetNumberOfArrays(); ++i) {
    vtkDataArray* array = cellData->GetArray(i);
    if (array && array->GetNumberOfComponents() == 1) {
      return array;
    }
  }

  return nullptr;
}

bool PointInTriangle(const Vec3& p,
                     const SurfaceTriangle& tri,
                     double tolerance,
                     std::array<double, 3>* barycentric = nullptr) {
  // 使用重心坐标判断点是否在三角形内部，同时检查到三角形平面的距离。
  const Vec3 v0 = tri.points[1] - tri.points[0];
  const Vec3 v1 = tri.points[2] - tri.points[0];
  const Vec3 v2 = p - tri.points[0];
  const double d00 = Dot(v0, v0);
  const double d01 = Dot(v0, v1);
  const double d11 = Dot(v1, v1);
  const double d20 = Dot(v2, v0);
  const double d21 = Dot(v2, v1);
  const double denom = d00 * d11 - d01 * d01;
  if (std::abs(denom) <= tolerance * tolerance) {
    return false;
  }

  const double v = (d11 * d20 - d01 * d21) / denom;
  const double w = (d00 * d21 - d01 * d20) / denom;
  const double u = 1.0 - v - w;

  const double distanceToPlane = std::abs(Dot(p - tri.points[0], tri.normal));
  const double scale = std::max({Norm(v0), Norm(v1), 1.0});
  const bool inside = distanceToPlane <= tolerance * scale &&
                      u >= -tolerance && v >= -tolerance && w >= -tolerance &&
                      u <= 1.0 + tolerance && v <= 1.0 + tolerance &&
                      w <= 1.0 + tolerance;

  if (inside && barycentric) {
    *barycentric = {u, v, w};
  }
  return inside;
}

struct Vec2 {
  double x = 0.0;
  double y = 0.0;
};

Vec2 operator+(const Vec2& a, const Vec2& b) {
  return {a.x + b.x, a.y + b.y};
}

Vec2 operator-(const Vec2& a, const Vec2& b) {
  return {a.x - b.x, a.y - b.y};
}

Vec2 operator*(const Vec2& a, double s) {
  return {a.x * s, a.y * s};
}

double Cross2D(const Vec2& a, const Vec2& b) {
  return a.x * b.y - a.y * b.x;
}

double PolygonArea2D(const std::vector<Vec2>& polygon) {
  if (polygon.size() < 3) {
    return 0.0;
  }

  double signedArea = 0.0;
  for (int i = 0; i < static_cast<int>(polygon.size()); ++i) {
    const Vec2& a = polygon[i];
    const Vec2& b = polygon[(i + 1) % polygon.size()];
    signedArea += Cross2D(a, b);
  }
  return 0.5 * std::abs(signedArea);
}

std::vector<Vec2> ClipPolygonByHalfPlane(const std::vector<Vec2>& polygon,
                                         const Vec2& clipA,
                                         const Vec2& clipB,
                                         double orientation,
                                         double tolerance) {
  std::vector<Vec2> clipped;
  if (polygon.empty()) {
    return clipped;
  }

  const Vec2 edge = clipB - clipA;
  const auto signedInside = [&](const Vec2& point) {
    return orientation * Cross2D(edge, point - clipA);
  };
  const auto inside = [&](const Vec2& point) {
    return signedInside(point) >= -tolerance;
  };

  Vec2 previous = polygon.back();
  bool previousInside = inside(previous);
  for (const Vec2& current : polygon) {
    const bool currentInside = inside(current);
    if (currentInside != previousInside) {
      const Vec2 segment = current - previous;
      const double previousDistance = signedInside(previous);
      const double currentDistance = signedInside(current);
      const double denom = previousDistance - currentDistance;
      if (std::abs(denom) > tolerance) {
        const double t = previousDistance / denom;
        clipped.push_back(previous + segment * t);
      }
    }
    if (currentInside) {
      clipped.push_back(current);
    }
    previous = current;
    previousInside = currentInside;
  }
  return clipped;
}

double ClippedProjectionAreaRatio(const std::array<Vec3, 3>& sourceTriangle,
                                  const SurfaceTriangle& targetTriangle,
                                  double sourceArea,
                                  double tolerance) {
  if (sourceArea <= tolerance * tolerance ||
      targetTriangle.area <= tolerance * tolerance) {
    return 0.0;
  }

  const Vec3 origin = targetTriangle.points[0];
  const Vec3 axisU = Normalize(targetTriangle.points[1] - origin);
  Vec3 axisV = Cross(targetTriangle.normal, axisU);
  axisV = Normalize(axisV);
  if (Norm(axisU) <= tolerance || Norm(axisV) <= tolerance) {
    return 0.0;
  }

  const auto toPlane2D = [&](const Vec3& point) {
    const Vec3 projected =
        point - targetTriangle.normal * Dot(point - origin, targetTriangle.normal);
    const Vec3 local = projected - origin;
    return Vec2{Dot(local, axisU), Dot(local, axisV)};
  };

  std::vector<Vec2> projected = {
      toPlane2D(sourceTriangle[0]),
      toPlane2D(sourceTriangle[1]),
      toPlane2D(sourceTriangle[2])};
  std::array<Vec2, 3> target = {
      toPlane2D(targetTriangle.points[0]),
      toPlane2D(targetTriangle.points[1]),
      toPlane2D(targetTriangle.points[2])};

  double targetOrientation =
      Cross2D(target[1] - target[0], target[2] - target[0]);
  if (std::abs(targetOrientation) <= tolerance * tolerance) {
    return 0.0;
  }
  targetOrientation = targetOrientation > 0.0 ? 1.0 : -1.0;

  std::vector<Vec2> clipped = projected;
  for (int i = 0; i < 3; ++i) {
    clipped = ClipPolygonByHalfPlane(
        clipped, target[i], target[(i + 1) % 3], targetOrientation, tolerance);
    if (clipped.empty()) {
      return 0.0;
    }
  }

  const double clippedArea = PolygonArea2D(clipped);
  return Clamp(clippedArea / sourceArea, 0.0, 1.0);
}

double DihedralCompatibility(const Vec3& referenceNormal,
                             const Vec3& candidateNormal) {
  const double dot = Clamp(std::abs(Dot(referenceNormal, candidateNormal)), 0.0, 1.0);
  const double angle = std::acos(dot);
  return 1.0 - angle / (0.5 * vtkMath::Pi());
}

class SurfaceIndex {
 public:
  SurfaceIndex(vtkPolyData* surface, double tolerance)
      : surface_(surface), tolerance_(tolerance), quantize_(tolerance) {}

  bool Build() {
    vtkDataArray* faceIds = FindFaceIdArray(surface_);
    if (!faceIds) {
      std::cerr << "Surface mesh has no scalar cell array for face IDs.\n";
      return false;
    }

    vtkPoints* points = surface_->GetPoints();
    if (!points) {
      std::cerr << "Surface mesh has no points.\n";
      return false;
    }

    for (vtkIdType cellId = 0; cellId < surface_->GetNumberOfCells(); ++cellId) {
      vtkCell* cell = surface_->GetCell(cellId);
      if (!cell || cell->GetNumberOfPoints() != 3) {
        ++skippedNonTriangles_;
        continue;
      }

      SurfaceTriangle tri;
      tri.cellId = cellId;
      tri.faceId = static_cast<int>(std::llround(faceIds->GetTuple1(cellId)));
      for (int i = 0; i < 3; ++i) {
        tri.pointIds[i] = cell->GetPointId(i);
        tri.points[i] = GetPoint(points, tri.pointIds[i]);
        tri.pointKeys[i] = quantize_(tri.points[i]);
      }

      tri.area = TriangleArea(tri.points[0], tri.points[1], tri.points[2]);
      if (tri.area <= tolerance_ * tolerance_) {
        ++skippedDegenerate_;
        continue;
      }
      tri.normal = Normalize(Cross(tri.points[1] - tri.points[0],
                                   tri.points[2] - tri.points[0]));

      tri.bounds[0] = tri.bounds[1] = tri.points[0].x;
      tri.bounds[2] = tri.bounds[3] = tri.points[0].y;
      tri.bounds[4] = tri.bounds[5] = tri.points[0].z;
      for (int i = 1; i < 3; ++i) {
        tri.bounds[0] = std::min(tri.bounds[0], tri.points[i].x);
        tri.bounds[1] = std::max(tri.bounds[1], tri.points[i].x);
        tri.bounds[2] = std::min(tri.bounds[2], tri.points[i].y);
        tri.bounds[3] = std::max(tri.bounds[3], tri.points[i].y);
        tri.bounds[4] = std::min(tri.bounds[4], tri.points[i].z);
        tri.bounds[5] = std::max(tri.bounds[5], tri.points[i].z);
      }

      const int index = static_cast<int>(triangles_.size());
      for (const auto& key : tri.pointKeys) {
        // 点 -> surface 三角形列表：用于快速查“单点侧”落在哪些面上。
        pointToTriangles_[key].push_back(index);
      }
      // 三个顶点组成的三角形 -> surface 三角形：用于快速查“三点侧”的 Face ID。
      triangleToSurface_[MakeTriangleKey(tri.pointKeys)].push_back(index);
      faceIdToTriangles_[tri.faceId].push_back(index);
      triangles_.push_back(tri);
    }

    if (triangles_.empty()) {
      std::cerr << "Surface mesh has no usable triangles.\n";
      return false;
    }
    if (skippedNonTriangles_ > 0) {
      std::cerr << "Skipped " << skippedNonTriangles_
                << " non-triangle surface cells.\n";
    }
    if (skippedDegenerate_ > 0) {
      std::cerr << "Skipped " << skippedDegenerate_
                << " degenerate surface triangles.\n";
    }
    return true;
  }

  SurfaceFaceMatch FindSurfaceTriangleMatch(
      const std::array<Vec3, 3>& points) const {
    std::array<QuantKey, 3> keys = {
        quantize_(points[0]), quantize_(points[1]), quantize_(points[2])};
    const auto it = triangleToSurface_.find(MakeTriangleKey(keys));
    if (it != triangleToSurface_.end() && !it->second.empty()) {
      const SurfaceTriangle& tri = triangles_[it->second.front()];
      return {tri.faceId, tri.normal};
    }

    // 如果三点刚好因为编号/容差没完全命中，就用重心做一次点落面兜底。
    const Vec3 centroid = (points[0] + points[1] + points[2]) * (1.0 / 3.0);
    return {FindFaceIdForPoint(centroid, nullptr, 0.70, nullptr, 0.0), {}};
  }

  int FindFaceIdForPoint(const Vec3& point,
                         const std::array<Vec3, 3>* midTriangle,
                         double projectionThreshold,
                         const Vec3* preferredParallelNormal,
                         double dihedralWeight) const {
    // 先通过量化后的点坐标查候选 surface 三角形；没找到再扫包围盒兜底。
    std::vector<int> candidates = CandidateTrianglesForPoint(point);
    if (candidates.empty()) {
      candidates = SlowCandidateTrianglesForPoint(point);
    }

    std::map<int, double> faceToProjectionRatio;
    std::map<int, double> faceToDihedralCompatibility;
    std::map<int, int> faceCounts;
    Vec3 midNormal{};
    double midArea = 0.0;
    if (midTriangle) {
      midArea = TriangleArea((*midTriangle)[0], (*midTriangle)[1], (*midTriangle)[2]);
      if (midArea > tolerance_ * tolerance_) {
        midNormal = Normalize(Cross((*midTriangle)[1] - (*midTriangle)[0],
                                    (*midTriangle)[2] - (*midTriangle)[0]));
      }
    }

    for (const int triIndex : candidates) {
      const SurfaceTriangle& tri = triangles_[triIndex];
      if (!PointInTriangle(point, tri, tolerance_ * 4.0)) {
        continue;
      }
      ++faceCounts[tri.faceId];
      if (midArea > tolerance_ * tolerance_) {
        // 投影面积必须裁剪到候选 surface 三角形内部，面外部分不计入比例。
        const double ratio =
            ClippedProjectionAreaRatio(*midTriangle, tri, midArea, tolerance_);
        auto& bestRatio = faceToProjectionRatio[tri.faceId];
        bestRatio = std::max(bestRatio, ratio);
      }
      if (preferredParallelNormal) {
        auto& bestCompatibility = faceToDihedralCompatibility[tri.faceId];
        bestCompatibility = std::max(
            bestCompatibility,
            DihedralCompatibility(*preferredParallelNormal, tri.normal));
      }
    }

    if (faceCounts.empty()) {
      return kUnmatchedFaceId;
    }
    if (faceCounts.size() == 1) {
      // 点只落在一个 Face ID 上：属于“点在面内部”的情况，直接使用该 Face ID。
      return faceCounts.begin()->first;
    }
    if (!midTriangle || midArea <= tolerance_ * tolerance_) {
      return kUnmatchedFaceId;
    }

    // 投影面积按整个 Face ID 区域累计，而不是只取包含边界点的单个三角形。
    for (const auto& [faceId, count] : faceCounts) {
      faceToProjectionRatio[faceId] =
          ClippedProjectionAreaRatioForFaceId(*midTriangle, midArea, faceId);
    }

    int bestFaceId = kUnmatchedFaceId;
    double bestScore = -1.0;
    double bestRatio = -1.0;
    dihedralWeight = Clamp(dihedralWeight, 0.0, 1.0);
    const double projectionWeight = 1.0 - dihedralWeight;
    for (const auto& [faceId, ratio] : faceToProjectionRatio) {
      const auto dihedralIt = faceToDihedralCompatibility.find(faceId);
      const double dihedralScore =
          dihedralIt == faceToDihedralCompatibility.end() ? 1.0 : dihedralIt->second;
      // 组合评分：投影面积比越大越好，二面角越小越好。
      // dihedralScore = 1 表示两面平行/反平行，= 0 表示近似垂直。
      const double score =
          projectionWeight * ratio + dihedralWeight * dihedralScore;
      if (score > bestScore) {
        bestScore = score;
        bestRatio = ratio;
        bestFaceId = faceId;
      }
    }

    if (bestRatio > projectionThreshold) {
      // 点落在边界上时，只有投影面积比例超过阈值才认为匹配该面。
      return bestFaceId;
    }
    return kUnmatchedFaceId;
  }

  const Quantizer& quantizer() const { return quantize_; }

 private:
  double ClippedProjectionAreaRatioForFaceId(
      const std::array<Vec3, 3>& midTriangle,
      double midArea,
      int faceId) const {
    const auto it = faceIdToTriangles_.find(faceId);
    if (it == faceIdToTriangles_.end()) {
      return 0.0;
    }

    double ratio = 0.0;
    for (const int triIndex : it->second) {
      ratio += ClippedProjectionAreaRatio(midTriangle, triangles_[triIndex],
                                          midArea, tolerance_);
      if (ratio >= 1.0) {
        return 1.0;
      }
    }
    return Clamp(ratio, 0.0, 1.0);
  }

  std::vector<int> CandidateTrianglesForPoint(const Vec3& point) const {
    std::vector<int> result;
    const auto it = pointToTriangles_.find(quantize_(point));
    if (it == pointToTriangles_.end()) {
      return result;
    }

    std::unordered_set<int> unique;
    for (const int triIndex : it->second) {
      if (unique.insert(triIndex).second) {
        result.push_back(triIndex);
      }
    }
    return result;
  }

  std::vector<int> SlowCandidateTrianglesForPoint(const Vec3& point) const {
    std::vector<int> result;
    for (int i = 0; i < static_cast<int>(triangles_.size()); ++i) {
      const SurfaceTriangle& tri = triangles_[i];
      if (point.x < tri.bounds[0] - tolerance_ ||
          point.x > tri.bounds[1] + tolerance_ ||
          point.y < tri.bounds[2] - tolerance_ ||
          point.y > tri.bounds[3] + tolerance_ ||
          point.z < tri.bounds[4] - tolerance_ ||
          point.z > tri.bounds[5] + tolerance_) {
        continue;
      }
      if (PointInTriangle(point, tri, tolerance_ * 4.0)) {
        result.push_back(i);
      }
    }
    return result;
  }

  vtkPolyData* surface_ = nullptr;
  double tolerance_ = 1e-6;
  Quantizer quantize_;
  std::vector<SurfaceTriangle> triangles_;
  std::unordered_map<QuantKey, std::vector<int>, QuantKeyHash> pointToTriangles_;
  std::unordered_map<TriangleKey, std::vector<int>, TriangleKeyHash> triangleToSurface_;
  std::map<int, std::vector<int>> faceIdToTriangles_;
  int skippedNonTriangles_ = 0;
  int skippedDegenerate_ = 0;
};

struct EdgeRef {
  vtkIdType tetId = -1;
  vtkIdType p0 = -1;
  vtkIdType p1 = -1;
  Vec3 midpoint{};
};

struct TriangleTetMatch {
  vtkIdType tetId = -1;
  vtkIdType singlePoint = -1;
  std::array<vtkIdType, 3> facePoints{};
};

class VolumeMidpointIndex {
 public:
  VolumeMidpointIndex(vtkUnstructuredGrid* volume, const Quantizer& quantize)
      : volume_(volume), quantize_(quantize) {}

  bool Build() {
    if (!volume_ || !volume_->GetPoints()) {
      std::cerr << "Volume mesh has no points.\n";
      return false;
    }

    static constexpr std::array<std::array<int, 2>, 6> kTetEdges = {
        std::array<int, 2>{0, 1}, std::array<int, 2>{0, 2},
        std::array<int, 2>{0, 3}, std::array<int, 2>{1, 2},
        std::array<int, 2>{1, 3}, std::array<int, 2>{2, 3}};

    for (vtkIdType cellId = 0; cellId < volume_->GetNumberOfCells(); ++cellId) {
      vtkCell* cell = volume_->GetCell(cellId);
      if (!cell || cell->GetNumberOfPoints() != 4) {
        continue;
      }

      std::array<vtkIdType, 4> ids{};
      std::array<Vec3, 4> points{};
      for (int i = 0; i < 4; ++i) {
        ids[i] = cell->GetPointId(i);
        points[i] = GetPoint(volume_, ids[i]);
      }

      for (const auto& edge : kTetEdges) {
        const Vec3 midpoint = (points[edge[0]] + points[edge[1]]) * 0.5;
        // 中面网格的点来自四面体边中点。这里建立“中点 -> 四面体边”的反查表。
        midpointToEdges_[quantize_(midpoint)].push_back(
            {cellId, ids[edge[0]], ids[edge[1]], midpoint});
      }
    }

    if (midpointToEdges_.empty()) {
      std::cerr << "Volume mesh has no usable tetrahedra.\n";
      return false;
    }
    return true;
  }

  bool FindTriangleMatch(const std::array<Vec3, 3>& midPoints,
                         TriangleTetMatch* match) const {
    // 一个中面三角形的三个点应当分别是同一个四面体中三条边的中点。
    std::array<std::vector<EdgeRef>, 3> edgeCandidates;
    for (int i = 0; i < 3; ++i) {
      edgeCandidates[i] = FindMidpointCandidates(midPoints[i]);
      if (edgeCandidates[i].empty()) {
        return false;
      }
    }

    std::unordered_map<vtkIdType, int> tetHits;
    for (int i = 0; i < 3; ++i) {
      std::unordered_set<vtkIdType> hitThisPoint;
      for (const EdgeRef& edge : edgeCandidates[i]) {
        hitThisPoint.insert(edge.tetId);
      }
      // 统计每个四面体命中了几个中面点；三个点都命中的四面体才可能是来源。
      for (const vtkIdType tetId : hitThisPoint) {
        ++tetHits[tetId];
      }
    }

    for (const auto& [tetId, count] : tetHits) {
      if (count != 3) {
        continue;
      }

      std::array<EdgeRef, 3> edges{};
      for (int i = 0; i < 3; ++i) {
        const auto edgeIt = std::find_if(
            edgeCandidates[i].begin(), edgeCandidates[i].end(),
            [&](const EdgeRef& edge) { return edge.tetId == tetId; });
        if (edgeIt == edgeCandidates[i].end()) {
          continue;
        }
        edges[i] = *edgeIt;
      }

      std::map<vtkIdType, int> endpointCounts;
      for (const EdgeRef& edge : edges) {
        ++endpointCounts[edge.p0];
        ++endpointCounts[edge.p1];
      }

      // 对任务描述中的三角形情况，三条边会共享一个公共端点：
      // 公共端点是“单点侧”，另外三个端点组成“三点侧”的 surface 三角形。
      vtkIdType common = -1;
      for (const auto& [pointId, endpointCount] : endpointCounts) {
        if (endpointCount == 3) {
          common = pointId;
          break;
        }
      }
      if (common < 0) {
        continue;
      }

      std::array<vtkIdType, 3> facePoints{};
      int facePointCount = 0;
      for (const EdgeRef& edge : edges) {
        const vtkIdType other = (edge.p0 == common) ? edge.p1 : edge.p0;
        if (std::find(facePoints.begin(), facePoints.begin() + facePointCount,
                      other) == facePoints.begin() + facePointCount) {
          facePoints[facePointCount++] = other;
        }
      }
      if (facePointCount != 3) {
        continue;
      }

      if (match) {
        match->tetId = tetId;
        match->singlePoint = common;
        match->facePoints = facePoints;
      }
      return true;
    }

    return false;
  }

 private:
  std::vector<EdgeRef> FindMidpointCandidates(const Vec3& midpoint) const {
    const QuantKey base = quantize_(midpoint);
    std::vector<EdgeRef> candidates;
    double bestDistance = std::numeric_limits<double>::max();
    constexpr int kSearchRadius = 1;
    for (int dx = -kSearchRadius; dx <= kSearchRadius; ++dx) {
      for (int dy = -kSearchRadius; dy <= kSearchRadius; ++dy) {
        for (int dz = -kSearchRadius; dz <= kSearchRadius; ++dz) {
          const QuantKey key{base.x + dx, base.y + dy, base.z + dz};
          const auto it = midpointToEdges_.find(key);
          if (it == midpointToEdges_.end()) {
            continue;
          }
          for (const EdgeRef& edge : it->second) {
            const double distance = Norm(edge.midpoint - midpoint);
            if (distance > quantize_.tolerance() * 1.5) {
              continue;
            }
            if (distance + quantize_.tolerance() * 0.1 < bestDistance) {
              bestDistance = distance;
              candidates.clear();
            }
            if (distance <= bestDistance + quantize_.tolerance() * 0.1) {
              candidates.push_back(edge);
            }
          }
        }
      }
    }
    return candidates;
  }

  vtkUnstructuredGrid* volume_ = nullptr;
  Quantizer quantize_;
  std::unordered_map<QuantKey, std::vector<EdgeRef>, QuantKeyHash> midpointToEdges_;
};

struct OutputTriangle {
  std::array<vtkIdType, 3> pointIds{};
  FacePair pair{};
};

double OutputTriangleArea(const OutputTriangle& triangle, vtkPoints* points) {
  return TriangleArea(GetPoint(points, triangle.pointIds[0]),
                      GetPoint(points, triangle.pointIds[1]),
                      GetPoint(points, triangle.pointIds[2]));
}

// 已处理三角形的边 -> 这条边两侧/周围出现过的 Face ID pair。
// 四边形拆分时用它判断每条边附近应继承哪个 pair。
using EdgePairMap = std::unordered_map<EdgeKey, std::vector<FacePair>, EdgeKeyHash>;
using VertexPairMap = std::unordered_map<vtkIdType, std::vector<FacePair>>;

void AddTriangleEdgesToMap(const OutputTriangle& triangle, EdgePairMap* edgePairs) {
  const FacePair pair = NormalizePair(triangle.pair);
  if (!IsValidPair(pair)) {
    return;
  }
  for (int i = 0; i < 3; ++i) {
    const vtkIdType a = triangle.pointIds[i];
    const vtkIdType b = triangle.pointIds[(i + 1) % 3];
    (*edgePairs)[MakeEdgeKey(a, b)].push_back(pair);
  }
}

void AddTriangleVerticesToMap(const OutputTriangle& triangle,
                              VertexPairMap* vertexPairs) {
  const FacePair pair = NormalizePair(triangle.pair);
  if (!IsValidPair(pair)) {
    return;
  }
  for (const vtkIdType pointId : triangle.pointIds) {
    (*vertexPairs)[pointId].push_back(pair);
  }
}

std::optional<FacePair> DominantPair(const std::vector<FacePair>& pairs) {
  if (pairs.empty()) {
    return std::nullopt;
  }
  std::map<FacePair, int> counts;
  for (const FacePair& pair : pairs) {
    const FacePair normalized = NormalizePair(pair);
    if (IsValidPair(normalized)) {
      ++counts[normalized];
    }
  }
  if (counts.empty()) {
    return std::nullopt;
  }
  return std::max_element(
             counts.begin(), counts.end(),
             [](const auto& a, const auto& b) { return a.second < b.second; })
      ->first;
}

std::optional<FacePair> PairForEdge(const EdgePairMap& edgePairs,
                                    vtkIdType a,
                                    vtkIdType b) {
  const auto it = edgePairs.find(MakeEdgeKey(a, b));
  if (it == edgePairs.end()) {
    return std::nullopt;
  }
  return DominantPair(it->second);
}

FacePair PairFromOptionalList(const std::vector<std::optional<FacePair>>& pairs,
                              FacePair fallback) {
  std::vector<FacePair> present;
  for (const auto& pair : pairs) {
    if (pair) {
      if (IsValidPair(*pair)) {
        present.push_back(*pair);
      }
    }
  }
  const auto dominant = DominantPair(present);
  return dominant.value_or(fallback);
}

FacePair FallbackPairForQuad(
    const std::array<std::optional<FacePair>, 4>& edgePairs) {
  std::vector<FacePair> present;
  for (const auto& pair : edgePairs) {
    if (pair) {
      if (IsValidPair(*pair)) {
        present.push_back(*pair);
      }
    }
  }
  const auto dominant = DominantPair(present);
  return dominant.value_or(FacePair{kUnmatchedFaceId, kUnmatchedFaceId});
}

FacePair VotePairForTriangle(const std::array<vtkIdType, 3>& triangle,
                             const EdgePairMap& edgePairMap,
                             const VertexPairMap& vertexPairMap,
                             FacePair fallback) {
  std::map<FacePair, double> scores;

  for (int i = 0; i < 3; ++i) {
    const auto edgeIt =
        edgePairMap.find(MakeEdgeKey(triangle[i], triangle[(i + 1) % 3]));
    if (edgeIt != edgePairMap.end()) {
      for (const FacePair& pair : edgeIt->second) {
        const FacePair normalized = NormalizePair(pair);
        if (IsValidPair(normalized)) {
          scores[normalized] += 3.0;
        }
      }
    }
  }

  for (const vtkIdType pointId : triangle) {
    const auto vertexIt = vertexPairMap.find(pointId);
    if (vertexIt != vertexPairMap.end()) {
      for (const FacePair& pair : vertexIt->second) {
        const FacePair normalized = NormalizePair(pair);
        if (IsValidPair(normalized)) {
          scores[normalized] += 1.0;
        }
      }
    }
  }

  if (scores.empty()) {
    return fallback;
  }
  return std::max_element(
             scores.begin(), scores.end(),
             [](const auto& a, const auto& b) { return a.second < b.second; })
      ->first;
}

double VoteStrengthForTriangle(const std::array<vtkIdType, 3>& triangle,
                               const EdgePairMap& edgePairMap,
                               const VertexPairMap& vertexPairMap) {
  std::map<FacePair, double> scores;
  for (int i = 0; i < 3; ++i) {
    const auto edgeIt =
        edgePairMap.find(MakeEdgeKey(triangle[i], triangle[(i + 1) % 3]));
    if (edgeIt != edgePairMap.end()) {
      for (const FacePair& pair : edgeIt->second) {
        const FacePair normalized = NormalizePair(pair);
        if (IsValidPair(normalized)) {
          scores[normalized] += 3.0;
        }
      }
    }
  }
  for (const vtkIdType pointId : triangle) {
    const auto vertexIt = vertexPairMap.find(pointId);
    if (vertexIt != vertexPairMap.end()) {
      for (const FacePair& pair : vertexIt->second) {
        const FacePair normalized = NormalizePair(pair);
        if (IsValidPair(normalized)) {
          scores[normalized] += 1.0;
        }
      }
    }
  }
  if (scores.empty()) {
    return 0.0;
  }
  return std::max_element(
             scores.begin(), scores.end(),
             [](const auto& a, const auto& b) { return a.second < b.second; })
      ->second;
}

std::array<OutputTriangle, 2> SplitQuad(
    const std::array<vtkIdType, 4>& quad,
    const EdgePairMap& edgePairMap,
    const VertexPairMap& vertexPairMap) {
  // 四边形四条边依次为 01、12、23、30。每条边尝试从相邻三角形继承 pair。
  std::array<std::optional<FacePair>, 4> edgePairs = {
      PairForEdge(edgePairMap, quad[0], quad[1]),
      PairForEdge(edgePairMap, quad[1], quad[2]),
      PairForEdge(edgePairMap, quad[2], quad[3]),
      PairForEdge(edgePairMap, quad[3], quad[0])};

  const FacePair fallback = FallbackPairForQuad(edgePairs);

  const auto vertexPair = [&](int vertex) -> std::optional<FacePair> {
    std::vector<FacePair> pairs;
    const int previousEdge = (vertex + 3) % 4;
    const int nextEdge = vertex;
    if (edgePairs[previousEdge]) {
      pairs.push_back(*edgePairs[previousEdge]);
    }
    if (edgePairs[nextEdge]) {
      pairs.push_back(*edgePairs[nextEdge]);
    }
    return DominantPair(pairs);
  };

  const auto v0 = vertexPair(0);
  const auto v1 = vertexPair(1);
  const auto v2 = vertexPair(2);
  const auto v3 = vertexPair(3);

  // 如果一组对角顶点周围的 pair 不同，说明这条对角线更可能是两个关系区的分界。
  // 两组都不明显时，默认走 0-2 对角线，保证输出稳定可复现。
  const bool split02 = v0 && v2 && NormalizePair(*v0) != NormalizePair(*v2);
  const bool split13 = v1 && v3 && NormalizePair(*v1) != NormalizePair(*v3);

  const std::array<vtkIdType, 3> tri02A = {quad[0], quad[1], quad[2]};
  const std::array<vtkIdType, 3> tri02B = {quad[0], quad[2], quad[3]};
  const std::array<vtkIdType, 3> tri13A = {quad[0], quad[1], quad[3]};
  const std::array<vtkIdType, 3> tri13B = {quad[1], quad[2], quad[3]};

  const double score02 = VoteStrengthForTriangle(tri02A, edgePairMap, vertexPairMap) +
                         VoteStrengthForTriangle(tri02B, edgePairMap, vertexPairMap);
  const double score13 = VoteStrengthForTriangle(tri13A, edgePairMap, vertexPairMap) +
                         VoteStrengthForTriangle(tri13B, edgePairMap, vertexPairMap);

  const bool use13 = (split13 && !split02) || (!split02 && score13 > score02) ||
                     (split13 && score13 >= score02);

  if (use13) {
    const FacePair pairA =
        VotePairForTriangle(tri13A, edgePairMap, vertexPairMap, fallback);
    const FacePair pairB =
        VotePairForTriangle(tri13B, edgePairMap, vertexPairMap, fallback);
    return {OutputTriangle{tri13A, pairA}, OutputTriangle{tri13B, pairB}};
  }

  const FacePair pairA =
      VotePairForTriangle(tri02A, edgePairMap, vertexPairMap, fallback);
  const FacePair pairB =
      VotePairForTriangle(tri02B, edgePairMap, vertexPairMap, fallback);
  return {OutputTriangle{tri02A, pairA}, OutputTriangle{tri02B, pairB}};
}

bool CanSplitQuadFromKnownNeighbors(const std::array<vtkIdType, 4>& quad,
                                    const EdgePairMap& edgePairMap,
                                    const VertexPairMap& vertexPairMap) {
  double score = 0.0;
  for (int i = 0; i < 4; ++i) {
    const auto edgeIt =
        edgePairMap.find(MakeEdgeKey(quad[i], quad[(i + 1) % 4]));
    if (edgeIt != edgePairMap.end()) {
      for (const FacePair& pair : edgeIt->second) {
        if (IsValidPair(pair)) {
          score += 3.0;
        }
      }
    }
  }
  for (const vtkIdType pointId : quad) {
    const auto vertexIt = vertexPairMap.find(pointId);
    if (vertexIt != vertexPairMap.end()) {
      for (const FacePair& pair : vertexIt->second) {
        if (IsValidPair(pair)) {
          score += 1.0;
        }
      }
    }
  }
  return score > 0.0;
}

int SmoothSmallLabelComponents(std::vector<OutputTriangle>* triangles,
                               vtkPoints* points,
                               int maxSmallComponentTriangles = 24,
                               double minNeighborDominance = 1.50) {
  if (!triangles || !points || triangles->empty()) {
    return 0;
  }

  std::unordered_map<EdgeKey, std::vector<int>, EdgeKeyHash> edgeToTriangles;
  edgeToTriangles.reserve(triangles->size() * 3);
  for (int triIndex = 0; triIndex < static_cast<int>(triangles->size());
       ++triIndex) {
    const auto& ids = (*triangles)[triIndex].pointIds;
    for (int i = 0; i < 3; ++i) {
      edgeToTriangles[MakeEdgeKey(ids[i], ids[(i + 1) % 3])].push_back(triIndex);
    }
  }

  std::vector<std::vector<int>> adjacency(triangles->size());
  for (const auto& [edge, incidentTriangles] : edgeToTriangles) {
    if (incidentTriangles.size() < 2) {
      continue;
    }
    for (int i = 0; i < static_cast<int>(incidentTriangles.size()); ++i) {
      for (int j = i + 1; j < static_cast<int>(incidentTriangles.size()); ++j) {
        adjacency[incidentTriangles[i]].push_back(incidentTriangles[j]);
        adjacency[incidentTriangles[j]].push_back(incidentTriangles[i]);
      }
    }
  }

  std::vector<char> visited(triangles->size(), 0);
  std::vector<FacePair> newPairs(triangles->size());
  for (int i = 0; i < static_cast<int>(triangles->size()); ++i) {
    newPairs[i] = NormalizePair((*triangles)[i].pair);
  }

  int changed = 0;
  std::vector<int> stack;
  std::vector<int> component;
  for (int start = 0; start < static_cast<int>(triangles->size()); ++start) {
    if (visited[start]) {
      continue;
    }

    const FacePair componentPair = NormalizePair((*triangles)[start].pair);
    stack.clear();
    component.clear();
    stack.push_back(start);
    visited[start] = 1;

    while (!stack.empty()) {
      const int triIndex = stack.back();
      stack.pop_back();
      component.push_back(triIndex);

      for (const int neighbor : adjacency[triIndex]) {
        if (visited[neighbor]) {
          continue;
        }
        if (NormalizePair((*triangles)[neighbor].pair) != componentPair) {
          continue;
        }
        visited[neighbor] = 1;
        stack.push_back(neighbor);
      }
    }

    if (static_cast<int>(component.size()) > maxSmallComponentTriangles) {
      continue;
    }

    std::map<FacePair, double> neighborScores;
    for (const int triIndex : component) {
      const double area = OutputTriangleArea((*triangles)[triIndex], points);
      for (const int neighbor : adjacency[triIndex]) {
        const FacePair neighborPair = NormalizePair((*triangles)[neighbor].pair);
        if (neighborPair == componentPair) {
          continue;
        }
        neighborScores[neighborPair] += std::max(area, 1e-12);
      }
    }
    if (neighborScores.empty()) {
      continue;
    }

    auto best = std::max_element(
        neighborScores.begin(), neighborScores.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });
    double secondScore = 0.0;
    for (const auto& [pair, score] : neighborScores) {
      if (pair != best->first) {
        secondScore = std::max(secondScore, score);
      }
    }
    if (secondScore > 0.0 &&
        best->second < secondScore * minNeighborDominance) {
      continue;
    }

    for (const int triIndex : component) {
      newPairs[triIndex] = best->first;
      ++changed;
    }
  }

  for (int i = 0; i < static_cast<int>(triangles->size()); ++i) {
    (*triangles)[i].pair = newPairs[i];
  }
  return changed;
}

int SmoothBoundaryJaggies(std::vector<OutputTriangle>* triangles,
                          int maxIterations = 4) {
  if (!triangles || triangles->empty()) {
    return 0;
  }

  std::unordered_map<EdgeKey, std::vector<int>, EdgeKeyHash> edgeToTriangles;
  edgeToTriangles.reserve(triangles->size() * 3);
  for (int triIndex = 0; triIndex < static_cast<int>(triangles->size());
       ++triIndex) {
    const auto& ids = (*triangles)[triIndex].pointIds;
    for (int i = 0; i < 3; ++i) {
      edgeToTriangles[MakeEdgeKey(ids[i], ids[(i + 1) % 3])].push_back(triIndex);
    }
  }

  std::vector<std::vector<int>> edgeNeighbors(triangles->size());
  for (const auto& [edge, incidentTriangles] : edgeToTriangles) {
    if (incidentTriangles.size() != 2) {
      continue;
    }
    edgeNeighbors[incidentTriangles[0]].push_back(incidentTriangles[1]);
    edgeNeighbors[incidentTriangles[1]].push_back(incidentTriangles[0]);
  }

  int totalChanged = 0;
  for (int iteration = 0; iteration < maxIterations; ++iteration) {
    std::vector<FacePair> newPairs(triangles->size());
    std::vector<char> shouldChange(triangles->size(), 0);
    for (int i = 0; i < static_cast<int>(triangles->size()); ++i) {
      newPairs[i] = NormalizePair((*triangles)[i].pair);
    }

    int changedThisIteration = 0;
    for (int triIndex = 0; triIndex < static_cast<int>(triangles->size());
         ++triIndex) {
      const FacePair current = NormalizePair((*triangles)[triIndex].pair);
      std::map<FacePair, int> neighborCounts;
      for (const int neighbor : edgeNeighbors[triIndex]) {
        const FacePair neighborPair = NormalizePair((*triangles)[neighbor].pair);
        if (neighborPair != current) {
          ++neighborCounts[neighborPair];
        }
      }

      if (neighborCounts.empty()) {
        continue;
      }
      const auto best = std::max_element(
          neighborCounts.begin(), neighborCounts.end(),
          [](const auto& a, const auto& b) { return a.second < b.second; });

      // 锯齿错误的典型形态：一个三角形伸进另一侧，至少两条共享边邻居同属另一 ID。
      if (best->second >= 2) {
        newPairs[triIndex] = best->first;
        shouldChange[triIndex] = 1;
        ++changedThisIteration;
      }
    }

    if (changedThisIteration == 0) {
      break;
    }
    for (int i = 0; i < static_cast<int>(triangles->size()); ++i) {
      if (shouldChange[i]) {
        (*triangles)[i].pair = newPairs[i];
      }
    }
    totalChanged += changedThisIteration;
  }

  return totalChanged;
}

struct BoundaryLoss {
  double total = 0.0;
  double boundaryLength = 0.0;
  double turnLoss = 0.0;
  double irregularVertexPenalty = 0.0;
};

BoundaryLoss ComputeBoundaryLoss(const std::vector<OutputTriangle>& triangles,
                                 vtkPoints* points,
                                 double lengthWeight = 0.25,
                                 double turnWeight = 2.0,
                                 double irregularWeight = 4.0) {
  std::unordered_map<EdgeKey, std::vector<int>, EdgeKeyHash> edgeToTriangles;
  edgeToTriangles.reserve(triangles.size() * 3);
  for (int triIndex = 0; triIndex < static_cast<int>(triangles.size());
       ++triIndex) {
    const auto& ids = triangles[triIndex].pointIds;
    for (int i = 0; i < 3; ++i) {
      edgeToTriangles[MakeEdgeKey(ids[i], ids[(i + 1) % 3])].push_back(triIndex);
    }
  }

  std::map<FacePair, std::map<vtkIdType, std::vector<vtkIdType>>> boundaryGraph;
  BoundaryLoss loss;
  for (const auto& [edge, incidentTriangles] : edgeToTriangles) {
    std::vector<FacePair> labels;
    labels.reserve(incidentTriangles.size());
    for (const int triIndex : incidentTriangles) {
      labels.push_back(NormalizePair(triangles[triIndex].pair));
    }

    bool isBoundary = incidentTriangles.size() != 2;
    if (!isBoundary && labels[0] != labels[1]) {
      isBoundary = true;
    }
    if (!isBoundary) {
      continue;
    }

    const double len = Norm(GetPoint(points, edge.a) - GetPoint(points, edge.b));
    loss.boundaryLength += len;
    for (const FacePair& label : labels) {
      auto& graph = boundaryGraph[label];
      graph[edge.a].push_back(edge.b);
      graph[edge.b].push_back(edge.a);
    }
  }

  for (const auto& [label, graph] : boundaryGraph) {
    for (const auto& [pointId, neighbors] : graph) {
      std::vector<vtkIdType> uniqueNeighbors = neighbors;
      std::sort(uniqueNeighbors.begin(), uniqueNeighbors.end());
      uniqueNeighbors.erase(std::unique(uniqueNeighbors.begin(), uniqueNeighbors.end()),
                            uniqueNeighbors.end());

      if (uniqueNeighbors.size() != 2) {
        loss.irregularVertexPenalty +=
            1.0 + std::abs(static_cast<int>(uniqueNeighbors.size()) - 2);
        continue;
      }

      const Vec3 center = GetPoint(points, pointId);
      Vec3 a = Normalize(GetPoint(points, uniqueNeighbors[0]) - center);
      Vec3 b = Normalize(GetPoint(points, uniqueNeighbors[1]) - center);
      const double dot = Clamp(Dot(a, b), -1.0, 1.0);
      const double angle = std::acos(dot);
      const double turn = vtkMath::Pi() - angle;
      loss.turnLoss += turn * turn;
    }
  }

  loss.total = lengthWeight * loss.boundaryLength +
               turnWeight * loss.turnLoss +
               irregularWeight * loss.irregularVertexPenalty;
  return loss;
}

std::vector<std::vector<int>> BuildEdgeNeighbors(
    const std::vector<OutputTriangle>& triangles) {
  std::unordered_map<EdgeKey, std::vector<int>, EdgeKeyHash> edgeToTriangles;
  edgeToTriangles.reserve(triangles.size() * 3);
  for (int triIndex = 0; triIndex < static_cast<int>(triangles.size());
       ++triIndex) {
    const auto& ids = triangles[triIndex].pointIds;
    for (int i = 0; i < 3; ++i) {
      edgeToTriangles[MakeEdgeKey(ids[i], ids[(i + 1) % 3])].push_back(triIndex);
    }
  }

  std::vector<std::vector<int>> neighbors(triangles.size());
  for (const auto& [edge, incidentTriangles] : edgeToTriangles) {
    if (incidentTriangles.size() != 2) {
      continue;
    }
    neighbors[incidentTriangles[0]].push_back(incidentTriangles[1]);
    neighbors[incidentTriangles[1]].push_back(incidentTriangles[0]);
  }
  return neighbors;
}

int OptimizeBoundaryByLoss(std::vector<OutputTriangle>* triangles,
                           vtkPoints* points,
                           double minImprovement = 0.50,
                           int maxSinglePasses = 1,
                           int maxSmallComponentTriangles = 36,
                           double mergeMinImprovement = 3.00) {
  if (!triangles || !points || triangles->empty()) {
    return 0;
  }

  int changed = 0;
  auto neighbors = BuildEdgeNeighbors(*triangles);
  BoundaryLoss currentLoss = ComputeBoundaryLoss(*triangles, points);

  for (int pass = 0; pass < maxSinglePasses; ++pass) {
    int passChanged = 0;
    for (int triIndex = 0; triIndex < static_cast<int>(triangles->size());
         ++triIndex) {
      const FacePair current = NormalizePair((*triangles)[triIndex].pair);
      std::set<FacePair> candidates;
      for (const int neighbor : neighbors[triIndex]) {
        const FacePair neighborPair = NormalizePair((*triangles)[neighbor].pair);
        if (neighborPair != current) {
          candidates.insert(neighborPair);
        }
      }
      if (candidates.empty()) {
        continue;
      }

      FacePair bestPair = current;
      BoundaryLoss bestLoss = currentLoss;
      for (const FacePair& candidate : candidates) {
        (*triangles)[triIndex].pair = candidate;
        const BoundaryLoss trialLoss = ComputeBoundaryLoss(*triangles, points);
        if (trialLoss.total + minImprovement < bestLoss.total) {
          bestLoss = trialLoss;
          bestPair = candidate;
        }
      }
      (*triangles)[triIndex].pair = current;

      if (bestPair != current) {
        (*triangles)[triIndex].pair = bestPair;
        currentLoss = bestLoss;
        ++changed;
        ++passChanged;
      }
    }
    if (passChanged == 0) {
      break;
    }
  }

  // 集体投降：小连通片整体并入相邻阵营，只有 loss 有足够改善才接受。
  neighbors = BuildEdgeNeighbors(*triangles);
  std::vector<char> visited(triangles->size(), 0);
  for (int start = 0; start < static_cast<int>(triangles->size()); ++start) {
    if (visited[start]) {
      continue;
    }

    const FacePair label = NormalizePair((*triangles)[start].pair);
    std::vector<int> component;
    std::vector<int> stack = {start};
    visited[start] = 1;
    while (!stack.empty()) {
      const int triIndex = stack.back();
      stack.pop_back();
      component.push_back(triIndex);
      for (const int neighbor : neighbors[triIndex]) {
        if (!visited[neighbor] &&
            NormalizePair((*triangles)[neighbor].pair) == label) {
          visited[neighbor] = 1;
          stack.push_back(neighbor);
        }
      }
    }

    if (component.empty() ||
        static_cast<int>(component.size()) > maxSmallComponentTriangles) {
      continue;
    }

    std::set<FacePair> candidates;
    for (const int triIndex : component) {
      for (const int neighbor : neighbors[triIndex]) {
        const FacePair neighborPair = NormalizePair((*triangles)[neighbor].pair);
        if (neighborPair != label) {
          candidates.insert(neighborPair);
        }
      }
    }
    if (candidates.empty()) {
      continue;
    }

    FacePair bestPair = label;
    BoundaryLoss bestLoss = currentLoss;
    for (const FacePair& candidate : candidates) {
      for (const int triIndex : component) {
        (*triangles)[triIndex].pair = candidate;
      }
      const BoundaryLoss trialLoss = ComputeBoundaryLoss(*triangles, points);
      if (trialLoss.total + mergeMinImprovement < bestLoss.total) {
        bestLoss = trialLoss;
        bestPair = candidate;
      }
    }
    for (const int triIndex : component) {
      (*triangles)[triIndex].pair = label;
    }

    if (bestPair != label) {
      for (const int triIndex : component) {
        (*triangles)[triIndex].pair = bestPair;
      }
      currentLoss = bestLoss;
      changed += static_cast<int>(component.size());
    }
  }

  return changed;
}

int MergeTinyRegions(std::vector<OutputTriangle>* triangles,
                     vtkPoints* points,
                     int maxTinyComponentTriangles = 48,
                     double maxTinyComponentArea = 0.05,
                     double allowedLossIncrease = -0.25) {
  if (!triangles || !points || triangles->empty()) {
    return 0;
  }

  auto neighbors = BuildEdgeNeighbors(*triangles);
  BoundaryLoss currentLoss = ComputeBoundaryLoss(*triangles, points);
  std::vector<char> visited(triangles->size(), 0);
  int changed = 0;

  for (int start = 0; start < static_cast<int>(triangles->size()); ++start) {
    if (visited[start]) {
      continue;
    }

    const FacePair label = NormalizePair((*triangles)[start].pair);
    std::vector<int> component;
    std::vector<int> stack = {start};
    visited[start] = 1;
    double componentArea = 0.0;
    while (!stack.empty()) {
      const int triIndex = stack.back();
      stack.pop_back();
      component.push_back(triIndex);
      componentArea += OutputTriangleArea((*triangles)[triIndex], points);
      for (const int neighbor : neighbors[triIndex]) {
        if (!visited[neighbor] &&
            NormalizePair((*triangles)[neighbor].pair) == label) {
          visited[neighbor] = 1;
          stack.push_back(neighbor);
        }
      }
    }

    if (static_cast<int>(component.size()) > maxTinyComponentTriangles &&
        componentArea > maxTinyComponentArea) {
      continue;
    }

    std::map<FacePair, int> neighborVotes;
    for (const int triIndex : component) {
      for (const int neighbor : neighbors[triIndex]) {
        const FacePair neighborPair = NormalizePair((*triangles)[neighbor].pair);
        if (neighborPair != label) {
          ++neighborVotes[neighborPair];
        }
      }
    }
    if (neighborVotes.empty()) {
      continue;
    }

    std::vector<std::pair<FacePair, int>> orderedVotes(neighborVotes.begin(),
                                                       neighborVotes.end());
    std::sort(orderedVotes.begin(), orderedVotes.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    FacePair bestPair = label;
    BoundaryLoss bestLoss = currentLoss;
    for (const auto& [candidate, votes] : orderedVotes) {
      for (const int triIndex : component) {
        (*triangles)[triIndex].pair = candidate;
      }
      const BoundaryLoss trialLoss = ComputeBoundaryLoss(*triangles, points);
      if (trialLoss.total <= bestLoss.total + allowedLossIncrease) {
        bestLoss = trialLoss;
        bestPair = candidate;
        break;
      }
    }
    for (const int triIndex : component) {
      (*triangles)[triIndex].pair = label;
    }

    if (bestPair != label) {
      for (const int triIndex : component) {
        (*triangles)[triIndex].pair = bestPair;
      }
      currentLoss = bestLoss;
      changed += static_cast<int>(component.size());
    }
  }

  return changed;
}

int SmoothBoundaryPatches(std::vector<OutputTriangle>* triangles,
                          vtkPoints* points,
                          int maxPatchTriangles = 3,
                          double minImprovement = 0.80,
                          int maxAccepted = 50) {
  if (!triangles || !points || triangles->empty()) {
    return 0;
  }

  auto neighbors = BuildEdgeNeighbors(*triangles);
  BoundaryLoss currentLoss = ComputeBoundaryLoss(*triangles, points);
  int changed = 0;
  std::set<std::vector<int>> tested;

  for (int seed = 0; seed < static_cast<int>(triangles->size()) &&
                     changed < maxAccepted;
       ++seed) {
    const FacePair label = NormalizePair((*triangles)[seed].pair);
    std::set<FacePair> seedNeighborLabels;
    for (const int neighbor : neighbors[seed]) {
      const FacePair neighborPair = NormalizePair((*triangles)[neighbor].pair);
      if (neighborPair != label) {
        seedNeighborLabels.insert(neighborPair);
      }
    }
    if (seedNeighborLabels.empty()) {
      continue;
    }

    std::vector<int> patch = {seed};
    std::set<int> patchSet = {seed};
    std::vector<int> frontier = {seed};
    while (!frontier.empty() &&
           static_cast<int>(patch.size()) < maxPatchTriangles) {
      const int triIndex = frontier.back();
      frontier.pop_back();
      for (const int neighbor : neighbors[triIndex]) {
        if (patchSet.count(neighbor) == 0 &&
            NormalizePair((*triangles)[neighbor].pair) == label) {
          patchSet.insert(neighbor);
          patch.push_back(neighbor);
          frontier.push_back(neighbor);
          if (static_cast<int>(patch.size()) >= maxPatchTriangles) {
            break;
          }
        }
      }
    }
    std::sort(patch.begin(), patch.end());
    if (!tested.insert(patch).second) {
      continue;
    }

    std::map<FacePair, int> candidateVotes;
    for (const int triIndex : patch) {
      for (const int neighbor : neighbors[triIndex]) {
        if (patchSet.count(neighbor) != 0) {
          continue;
        }
        const FacePair neighborPair = NormalizePair((*triangles)[neighbor].pair);
        if (neighborPair != label) {
          ++candidateVotes[neighborPair];
        }
      }
    }
    if (candidateVotes.empty()) {
      continue;
    }

    FacePair bestPair = label;
    BoundaryLoss bestLoss = currentLoss;
    for (const auto& [candidate, votes] : candidateVotes) {
      if (votes < 2) {
        continue;
      }
      for (const int triIndex : patch) {
        (*triangles)[triIndex].pair = candidate;
      }
      const BoundaryLoss trialLoss = ComputeBoundaryLoss(*triangles, points);
      if (trialLoss.total + minImprovement < bestLoss.total) {
        bestLoss = trialLoss;
        bestPair = candidate;
      }
    }
    for (const int triIndex : patch) {
      (*triangles)[triIndex].pair = label;
    }

    if (bestPair != label) {
      for (const int triIndex : patch) {
        (*triangles)[triIndex].pair = bestPair;
      }
      currentLoss = bestLoss;
      changed += static_cast<int>(patch.size());
    }
  }

  return changed;
}

double ComputeDefaultTolerance(vtkPolyData* surface,
                               vtkUnstructuredGrid* volume,
                               vtkPolyData* midSurface) {
  // 默认容差按整体模型包围盒对角线设置。模型越大，允许的绝对误差也稍大。
  double bounds[6] = {std::numeric_limits<double>::max(),
                      std::numeric_limits<double>::lowest(),
                      std::numeric_limits<double>::max(),
                      std::numeric_limits<double>::lowest(),
                      std::numeric_limits<double>::max(),
                      std::numeric_limits<double>::lowest()};

  const auto mergeBounds = [&](vtkDataSet* dataSet) {
    if (!dataSet || dataSet->GetNumberOfPoints() == 0) {
      return;
    }
    double local[6];
    dataSet->GetBounds(local);
    bounds[0] = std::min(bounds[0], local[0]);
    bounds[1] = std::max(bounds[1], local[1]);
    bounds[2] = std::min(bounds[2], local[2]);
    bounds[3] = std::max(bounds[3], local[3]);
    bounds[4] = std::min(bounds[4], local[4]);
    bounds[5] = std::max(bounds[5], local[5]);
  };

  mergeBounds(surface);
  mergeBounds(volume);
  mergeBounds(midSurface);

  if (bounds[0] > bounds[1]) {
    return 1e-6;
  }
  const double dx = bounds[1] - bounds[0];
  const double dy = bounds[3] - bounds[2];
  const double dz = bounds[5] - bounds[4];
  const double diagonal = std::sqrt(dx * dx + dy * dy + dz * dz);
  return std::max(diagonal * 1e-6, 1e-4);
}

vtkSmartPointer<vtkDataSet> ReadDataSet(const std::string& path) {
  vtkNew<vtkDataSetReader> reader;
  reader->SetFileName(path.c_str());
  reader->Update();

  vtkDataSet* output = reader->GetOutput();
  if (!output) {
    return nullptr;
  }

  vtkSmartPointer<vtkDataSet> data;
  if (auto* polyData = vtkPolyData::SafeDownCast(output)) {
    vtkSmartPointer<vtkPolyData> copy = vtkSmartPointer<vtkPolyData>::New();
    copy->DeepCopy(polyData);
    data = copy;
  } else if (auto* grid = vtkUnstructuredGrid::SafeDownCast(output)) {
    vtkSmartPointer<vtkUnstructuredGrid> copy =
        vtkSmartPointer<vtkUnstructuredGrid>::New();
    copy->DeepCopy(grid);
    data = copy;
  }
  return data;
}

vtkSmartPointer<vtkUnstructuredGrid> ReadUnstructuredGrid(const std::string& path) {
  vtkNew<vtkUnstructuredGridReader> reader;
  reader->SetFileName(path.c_str());
  reader->Update();

  vtkSmartPointer<vtkUnstructuredGrid> data =
      vtkSmartPointer<vtkUnstructuredGrid>::New();
  data->DeepCopy(reader->GetOutput());
  return data;
}

vtkSmartPointer<vtkPolyData> ExtractFaceCells(vtkDataSet* input,
                                              bool allowQuads,
                                              const std::string& roleName) {
  if (!input || input->GetNumberOfPoints() == 0) {
    return nullptr;
  }
  if (auto* polyData = vtkPolyData::SafeDownCast(input)) {
    vtkSmartPointer<vtkPolyData> copy = vtkSmartPointer<vtkPolyData>::New();
    copy->DeepCopy(polyData);
    return copy;
  }

  vtkNew<vtkPolyData> output;
  vtkNew<vtkPoints> points;
  points->SetDataTypeToDouble();
  points->SetNumberOfPoints(input->GetNumberOfPoints());
  for (vtkIdType pointId = 0; pointId < input->GetNumberOfPoints(); ++pointId) {
    double point[3];
    input->GetPoint(pointId, point);
    points->SetPoint(pointId, point);
  }
  output->SetPoints(points);
  output->GetPointData()->ShallowCopy(input->GetPointData());

  vtkNew<vtkCellArray> polys;
  output->GetCellData()->CopyAllocate(input->GetCellData());

  vtkIdType copiedCells = 0;
  vtkIdType skippedCells = 0;
  for (vtkIdType cellId = 0; cellId < input->GetNumberOfCells(); ++cellId) {
    vtkCell* cell = input->GetCell(cellId);
    if (!cell) {
      ++skippedCells;
      continue;
    }

    const vtkIdType numberOfPoints = cell->GetNumberOfPoints();
    if (numberOfPoints != 3 && !(allowQuads && numberOfPoints == 4)) {
      ++skippedCells;
      continue;
    }

    vtkNew<vtkIdList> ids;
    ids->SetNumberOfIds(numberOfPoints);
    for (vtkIdType i = 0; i < numberOfPoints; ++i) {
      ids->SetId(i, cell->GetPointId(i));
    }

    const vtkIdType newCellId = polys->InsertNextCell(ids);
    output->GetCellData()->CopyData(input->GetCellData(), cellId, newCellId);
    ++copiedCells;
  }

  output->SetPolys(polys);
  if (skippedCells > 0) {
    std::cerr << "Skipped " << skippedCells << " non-face cells while reading "
              << roleName << ".\n";
  }
  if (copiedCells == 0) {
    std::cerr << "No triangle" << (allowQuads ? "/quad" : "")
              << " cells found in " << roleName << ".\n";
    return nullptr;
  }
  return output;
}

bool WritePolyData(vtkPolyData* data, const std::string& path) {
  vtkNew<vtkPolyDataWriter> writer;
  writer->SetFileName(path.c_str());
  writer->SetInputData(data);
  writer->SetFileTypeToASCII();
  return writer->Write() == 1;
}

}  // namespace

bool MapMidSurfaceIds(const std::string& surfaceMeshPath,
                      const std::string& volumeMeshPath,
                      const std::string& midSurfacePath,
                      const std::string& outputPath,
                      double tolerance,
                      double projectionThreshold,
                      double dihedralWeight) {
  // 入口函数：读入三个 VTK 文件，输出全三角形且带 surface_id 的中面网格。
  vtkSmartPointer<vtkDataSet> surfaceDataSet = ReadDataSet(surfaceMeshPath);
  vtkSmartPointer<vtkUnstructuredGrid> volume = ReadUnstructuredGrid(volumeMeshPath);
  vtkSmartPointer<vtkDataSet> midSurfaceDataSet = ReadDataSet(midSurfacePath);
  vtkSmartPointer<vtkPolyData> surface =
      ExtractFaceCells(surfaceDataSet, false, "surface mesh");
  vtkSmartPointer<vtkPolyData> midSurface =
      ExtractFaceCells(midSurfaceDataSet, true, "mid-surface mesh");

  if (!surface || surface->GetNumberOfPoints() == 0 || surface->GetNumberOfCells() == 0) {
    std::cerr << "Failed to read a non-empty surface mesh: " << surfaceMeshPath << "\n";
    return false;
  }
  if (!volume || volume->GetNumberOfPoints() == 0 || volume->GetNumberOfCells() == 0) {
    std::cerr << "Failed to read a non-empty volume mesh: " << volumeMeshPath << "\n";
    return false;
  }
  if (!midSurface || midSurface->GetNumberOfPoints() == 0 ||
      midSurface->GetNumberOfCells() == 0) {
    std::cerr << "Failed to read a non-empty mid-surface mesh: "
              << midSurfacePath << "\n";
    return false;
  }

  if (tolerance <= 0.0) {
    tolerance = ComputeDefaultTolerance(surface, volume, midSurface);
  }
  projectionThreshold = std::clamp(projectionThreshold, 0.0, 1.0);
  dihedralWeight = std::clamp(dihedralWeight, 0.0, 1.0);

  SurfaceIndex surfaceIndex(surface, tolerance);
  if (!surfaceIndex.Build()) {
    return false;
  }

  // surfaceIndex 负责“查 Face ID”；volumeIndex 负责“中面三角形 -> 原四面体”。
  VolumeMidpointIndex volumeIndex(volume, surfaceIndex.quantizer());
  if (!volumeIndex.Build()) {
    return false;
  }

  std::vector<OutputTriangle> outputTriangles;
  std::vector<std::array<vtkIdType, 4>> quads;
  EdgePairMap edgePairMap;
  VertexPairMap vertexPairMap;
  int unmappedTriangles = 0;
  int skippedCells = 0;

  // 第一遍只处理原本就是三角形的中面单元。
  // 四边形先缓存起来，因为它们需要依赖周围三角形已经确定的 pair。
  for (vtkIdType cellId = 0; cellId < midSurface->GetNumberOfCells(); ++cellId) {
    vtkCell* cell = midSurface->GetCell(cellId);
    if (!cell) {
      ++skippedCells;
      continue;
    }

    const vtkIdType numberOfPoints = cell->GetNumberOfPoints();
    if (numberOfPoints == 3) {
      std::array<vtkIdType, 3> pointIds{};
      std::array<Vec3, 3> midPoints{};
      for (int i = 0; i < 3; ++i) {
        pointIds[i] = cell->GetPointId(i);
        midPoints[i] = GetPoint(midSurface, pointIds[i]);
      }

      TriangleTetMatch match;
      FacePair pair{kUnmatchedFaceId, kUnmatchedFaceId};
      if (volumeIndex.FindTriangleMatch(midPoints, &match)) {
        std::array<Vec3, 3> surfaceFacePoints{};
        for (int i = 0; i < 3; ++i) {
          surfaceFacePoints[i] = GetPoint(volume, match.facePoints[i]);
        }

        // 三点侧：直接由四面体中非公共端点组成的 surface 三角形确定。
        const SurfaceFaceMatch firstFace =
            surfaceIndex.FindSurfaceTriangleMatch(surfaceFacePoints);
        // 单点侧：判断公共端点落在哪个 Face ID 上；边界点按投影比例筛选。
        const int secondFaceId = surfaceIndex.FindFaceIdForPoint(
            GetPoint(volume, match.singlePoint), &midPoints, projectionThreshold,
            &firstFace.normal, dihedralWeight);
        pair = NormalizePair({firstFace.faceId, secondFaceId});
      } else {
        ++unmappedTriangles;
      }

      OutputTriangle triangle{pointIds, pair};
      outputTriangles.push_back(triangle);
      // 记录三角形边上的 pair，后续四边形拆分时用于继承局部区域 ID。
      AddTriangleEdgesToMap(triangle, &edgePairMap);
      AddTriangleVerticesToMap(triangle, &vertexPairMap);
      continue;
    }

    if (numberOfPoints == 4) {
      std::array<vtkIdType, 4> quad{};
      for (int i = 0; i < 4; ++i) {
        quad[i] = cell->GetPointId(i);
      }
      quads.push_back(quad);
      continue;
    }

    ++skippedCells;
  }

  // 第二步：把四边形全部拆成三角形，并从相邻三角形边继承 Face ID pair。
  std::vector<char> quadDone(quads.size(), 0);
  int remainingQuads = static_cast<int>(quads.size());
  bool progressed = true;
  while (remainingQuads > 0 && progressed) {
    progressed = false;
    for (int quadIndex = 0; quadIndex < static_cast<int>(quads.size());
         ++quadIndex) {
      if (quadDone[quadIndex]) {
        continue;
      }
      if (!CanSplitQuadFromKnownNeighbors(quads[quadIndex], edgePairMap,
                                          vertexPairMap)) {
        continue;
      }

      const auto split = SplitQuad(quads[quadIndex], edgePairMap, vertexPairMap);
      outputTriangles.push_back(split[0]);
      outputTriangles.push_back(split[1]);
      AddTriangleEdgesToMap(split[0], &edgePairMap);
      AddTriangleEdgesToMap(split[1], &edgePairMap);
      AddTriangleVerticesToMap(split[0], &vertexPairMap);
      AddTriangleVerticesToMap(split[1], &vertexPairMap);
      quadDone[quadIndex] = 1;
      --remainingQuads;
      progressed = true;
    }
  }

  for (int quadIndex = 0; quadIndex < static_cast<int>(quads.size());
       ++quadIndex) {
    if (quadDone[quadIndex]) {
      continue;
    }
    const auto split = SplitQuad(quads[quadIndex], edgePairMap, vertexPairMap);
    outputTriangles.push_back(split[0]);
    outputTriangles.push_back(split[1]);
  }

  const int smoothedTriangles =
      SmoothSmallLabelComponents(&outputTriangles, midSurface->GetPoints());
  const int jaggySmoothedTriangles = SmoothBoundaryJaggies(&outputTriangles);
  const bool runExpensiveLoss = outputTriangles.size() <= 80000;
  const int lossOptimizedTriangles =
      runExpensiveLoss
          ? OptimizeBoundaryByLoss(&outputTriangles, midSurface->GetPoints())
          : 0;
  const int patchSmoothedTriangles = 0;
  const int tinyMergedTriangles = 0;

  vtkNew<vtkPolyData> output;
  vtkNew<vtkPoints> points;
  points->SetDataTypeToDouble();
  points->DeepCopy(midSurface->GetPoints());
  output->SetPoints(points);
  output->GetPointData()->ShallowCopy(midSurface->GetPointData());

  vtkNew<vtkCellArray> polys;
  vtkNew<vtkIntArray> surfaceIds;
  surfaceIds->SetName("surface_id");
  surfaceIds->SetNumberOfComponents(1);

  vtkNew<vtkIntArray> faceId0;
  faceId0->SetName("face_id_0");
  faceId0->SetNumberOfComponents(1);

  vtkNew<vtkIntArray> faceId1;
  faceId1->SetName("face_id_1");
  faceId1->SetNumberOfComponents(1);

  // 第三步：Face ID pair -> 连续 surface_id。
  // 例如第一次遇到 (2, 8) 记为 0，第一次遇到 (5, 9) 记为 1。
  std::unordered_map<FacePair, int, FacePairHash> pairToSurfaceId;
  int nextSurfaceId = 0;
  for (const OutputTriangle& triangle : outputTriangles) {
    vtkIdType ids[3] = {triangle.pointIds[0], triangle.pointIds[1],
                        triangle.pointIds[2]};
    polys->InsertNextCell(3, ids);

    const FacePair pair = NormalizePair(triangle.pair);
    auto [it, inserted] = pairToSurfaceId.emplace(pair, nextSurfaceId);
    if (inserted) {
      ++nextSurfaceId;
    }

    surfaceIds->InsertNextValue(it->second);
    faceId0->InsertNextValue(pair.first);
    faceId1->InsertNextValue(pair.second);
  }

  output->SetPolys(polys);
  output->GetCellData()->AddArray(surfaceIds);
  output->GetCellData()->SetActiveScalars("surface_id");
  output->GetCellData()->AddArray(faceId0);
  output->GetCellData()->AddArray(faceId1);

  if (!WritePolyData(output, outputPath)) {
    std::cerr << "Failed to write output mesh: " << outputPath << "\n";
    return false;
  }

  std::cerr << "Wrote " << outputTriangles.size() << " triangles to "
            << outputPath << ". Surface relation count: " << nextSurfaceId
            << ".\n";
  if (smoothedTriangles > 0) {
    std::cerr << "Relabeled " << smoothedTriangles
              << " small boundary/component triangles by neighborhood voting.\n";
  }
  if (jaggySmoothedTriangles > 0) {
    std::cerr << "Relabeled " << jaggySmoothedTriangles
              << " jaggy boundary triangles by edge-neighborhood voting.\n";
  }
  if (patchSmoothedTriangles > 0) {
    std::cerr << "Relabeled " << patchSmoothedTriangles
              << " small boundary patch triangles by boundary-loss optimization.\n";
  }
  if (lossOptimizedTriangles > 0) {
    std::cerr << "Relabeled " << lossOptimizedTriangles
              << " boundary triangles/components by boundary-loss optimization.\n";
  }
  if (!runExpensiveLoss) {
    std::cerr << "Skipped expensive boundary-loss optimization for large mesh.\n";
  }
  if (tinyMergedTriangles > 0) {
    std::cerr << "Merged " << tinyMergedTriangles
              << " tiny-region triangles into neighboring regions.\n";
  }
  if (unmappedTriangles > 0) {
    std::cerr << "Warning: " << unmappedTriangles
              << " mid-surface triangles could not be matched to a tetrahedron.\n";
  }
  if (skippedCells > 0) {
    std::cerr << "Warning: skipped " << skippedCells
              << " mid-surface cells that were neither triangles nor quads.\n";
  }
  return true;
}

#ifdef MID_SURFACE_MAPPER_BUILD_CLI
int main(int argc, char** argv) {
  if (argc < 5 || argc > 8) {
    std::cerr << "Usage: " << argv[0]
              << " surface_mesh.vtk volume_mesh.vtk mid_surface.vtk "
                 "mid_surface_tri.vtk [tolerance] [projection_threshold] "
                 "[dihedral_weight]\n";
    return EXIT_FAILURE;
  }

  double tolerance = -1.0;
  if (argc >= 6) {
    tolerance = std::atof(argv[5]);
  }

  // 投影面积比例阈值
  double projectionThreshold = 0.30;
  if (argc >= 7) {
    projectionThreshold = std::atof(argv[6]);
  }

  // 二面角权重
  double dihedralWeight = 0.90;
  if (argc >= 8) {
    dihedralWeight = std::atof(argv[7]);
  }

  return MapMidSurfaceIds(argv[1], argv[2], argv[3], argv[4], tolerance,
                          projectionThreshold, dihedralWeight)
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
#endif
