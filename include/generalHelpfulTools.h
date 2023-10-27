#ifndef FS2D_GENERALHELPFULTOOLS_H
#define FS2D_GENERALHELPFULTOOLS_H

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <iostream>

class generalHelpfulTools {
public:
    static Eigen::Vector3d getRollPitchYaw(Eigen::Quaterniond quat);

    static Eigen::Quaterniond getQuaternionFromRPY(double roll, double pitch, double yaw);

    static Eigen::Matrix4d getTransformationMatrixFromRPY(double roll, double pitch, double yaw);

    static double angleDiff(double first, double second);//first-second

    static Eigen::Matrix4d
    interpolationTwo4DTransformations(Eigen::Matrix4d &transformation1, Eigen::Matrix4d &transformation2,
                                      double &t);// from 1 to two by t[0-1]
    static Eigen::Matrix4d getTransformationMatrix(Eigen::Vector3d &translation, Eigen::Quaterniond &rotation);

    static Eigen::Matrix4d getTransformationMatrixTF2(tf2::Vector3 &translation, tf2::Quaternion &rotation);

    static void getTF2FromTransformationMatrix(tf2::Vector3 &translation, tf2::Quaternion &rotation , Eigen::Matrix4d transformationMatrix);

    static void splitTransformationMatrixToQuadAndTrans(Eigen::Vector3d &translation, Eigen::Quaterniond &rotation,
                                                                      Eigen::Matrix4d transformationMatrix);

    static double weighted_mean(const std::vector<double> &data);

    static void
    smooth_curve(const std::vector<double> &input, std::vector<double> &smoothedOutput, int window_half_width);

    static Eigen::Matrix4d convertMatrixFromOurSystemToOpenCV(Eigen::Matrix4d inputMatrix);

    static double normalizeAngle(double inputAngle);

    static std::vector<std::string> getNextLineAndSplitIntoTokens(std::istream &str);
};


#endif //FS2D_GENERALHELPFULTOOLS_H