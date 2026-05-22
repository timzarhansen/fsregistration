#include <cstddef>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <memory>

#include "softRegistrationClass.h"

namespace py = pybind11;

// Helper: Eigen::Vector2d -> numpy array (2,)
py::array_t<double> eigen2d_to_numpy(const Eigen::Vector2d& v) {
    auto result = py::array_t<double>(2);
    py::buffer_info buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);
    ptr[0] = v.x();
    ptr[1] = v.y();
    return result;
}

// Helper: Eigen::Vector2i -> numpy array (2,)
py::array_t<int> eigen2i_to_numpy(const Eigen::Vector2i& v) {
    auto result = py::array_t<int>(2);
    py::buffer_info buf = result.request();
    int* ptr = static_cast<int*>(buf.ptr);
    ptr[0] = v.x();
    ptr[1] = v.y();
    return result;
}

// Helper: Eigen::Matrix2d -> numpy array (2,2)
py::array_t<double> eigen2x2_to_numpy(const Eigen::Matrix2d& m) {
    py::array_t<double> result({2, 2});
    py::buffer_info buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            ptr[i * 2 + j] = m(i, j);
    return result;
}

// Helper: Eigen::Matrix3d -> numpy array (3,3)
py::array_t<double> eigen3x3_to_numpy(const Eigen::Matrix3d& m) {
    py::array_t<double> result({3, 3});
    py::buffer_info buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            ptr[i * 3 + j] = m(i, j);
    return result;
}

// Helper: Eigen::Matrix4d -> numpy array (4,4)
py::array_t<double> eigen4x4_to_numpy(const Eigen::Matrix4d& m) {
    py::array_t<double> result({4, 4});
    py::buffer_info buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            ptr[i * 4 + j] = m(i, j);
    return result;
}

// Helper: numpy array (N*N) -> double*
double* numpy_to_double_array(py::array_t<double, py::array::c_style | py::array::forcecast> arr, int expectedSize) {
    auto buf = arr.request();
    if (buf.size != expectedSize) {
        throw std::runtime_error(
            "Array size mismatch. Expected " + std::to_string(expectedSize) +
            ", got " + std::to_string(buf.size)
        );
    }
    return static_cast<double*>(buf.ptr);
}

struct RotationPeak2D {
    double angle;
    double peakCorrelation;
    double covariance;
};

struct TranslationPeak2D {
    py::array_t<double> translationSI;
    py::array_t<int> translationVoxel;
    double peakHeight;
    double persistenceValue;
    py::array_t<double> covariance;
};

struct TransformationPeak2D {
    std::vector<TranslationPeak2D> potentialTranslations;
    RotationPeak2D potentialRotation;
};

class SoftRegistrationWrapper2D {
private:
    std::unique_ptr<softRegistrationClass> reg;
    int N_;

public:
    SoftRegistrationWrapper2D(int N)
        : N_(N) {
        reg.reset(new softRegistrationClass(N, N / 2, N / 2, N / 2 - 1));
    }

    // registrationOfTwoVoxelsSOFFTAllSoluations
    std::vector<TransformationPeak2D> register_all_solutions(
        py::array_t<double, py::array::c_style | py::array::forcecast> scan1,
        py::array_t<double, py::array::c_style | py::array::forcecast> scan2,
        double cellSize,
        bool useGauss,
        bool debug,
        double potentialNecessaryForPeak,
        bool multipleRadii,
        bool useClahe,
        bool useHamming
    ) {
        double* data1 = numpy_to_double_array(scan1, N_ * N_);
        double* data2 = numpy_to_double_array(scan2, N_ * N_);

        auto results = reg->registrationOfTwoVoxelsSOFFTAllSoluations(
            data1, data2,
            cellSize, useGauss, debug,
            potentialNecessaryForPeak, multipleRadii, useClahe, useHamming
        );

        std::vector<TransformationPeak2D> out;
        out.reserve(results.size());
        for (const auto& tp : results) {
            TransformationPeak2D outTp;
            for (const auto& t : tp.potentialTranslations) {
                TranslationPeak2D outT;
                outT.translationSI = eigen2d_to_numpy(t.translationSI);
                outT.translationVoxel = eigen2i_to_numpy(t.translationVoxel);
                outT.peakHeight = t.peakHeight;
                outT.persistenceValue = t.persistenceValue;
                outT.covariance = eigen2x2_to_numpy(t.covariance);
                outTp.potentialTranslations.push_back(outT);
            }
            outTp.potentialRotation.angle = tp.potentialRotation.angle;
            outTp.potentialRotation.peakCorrelation = tp.potentialRotation.peakCorrelation;
            outTp.potentialRotation.covariance = tp.potentialRotation.covariance;
            out.push_back(outTp);
        }
        return out;
    }

    // registrationOfTwoVoxelsSOFFTFast
    std::tuple<py::array_t<double>, py::array_t<double>> register_fast(
        py::array_t<double, py::array::c_style | py::array::forcecast> scan1,
        py::array_t<double, py::array::c_style | py::array::forcecast> scan2,
        py::array_t<double, py::array::c_style | py::array::forcecast> initialGuess,
        bool useInitialAngle,
        bool useInitialTranslation,
        double cellSize,
        bool useGauss,
        bool debug,
        double potentialNecessaryForPeak
    ) {
        double* data1 = numpy_to_double_array(scan1, N_ * N_);
        double* data2 = numpy_to_double_array(scan2, N_ * N_);

        auto buf = initialGuess.request();
        double* initPtr = static_cast<double*>(buf.ptr);
        Eigen::Map<Eigen::Matrix4d> initialGuessMap(initPtr);
        Eigen::Matrix4d initialGuessMat = initialGuessMap;

        Eigen::Matrix3d covarianceMatrix;
        Eigen::Matrix4d result = reg->registrationOfTwoVoxelsSOFFTFast(
            data1, data2,
            initialGuessMat, covarianceMatrix,
            useInitialAngle, useInitialTranslation,
            cellSize, useGauss, debug,
            potentialNecessaryForPeak
        );

        return std::make_tuple(eigen4x4_to_numpy(result), eigen3x3_to_numpy(covarianceMatrix));
    }

    // registrationOfTwoVoxelsSO3
    std::vector<TransformationPeak2D> register_so3(
        py::array_t<double, py::array::c_style | py::array::forcecast> scan1,
        py::array_t<double, py::array::c_style | py::array::forcecast> scan2,
        double cellSize,
        bool useGauss,
        bool debug,
        double potentialNecessaryForPeak,
        bool multipleRadii,
        bool useClahe,
        bool useHamming,
        bool benchmark
    ) {
        double* data1 = numpy_to_double_array(scan1, N_ * N_);
        double* data2 = numpy_to_double_array(scan2, N_ * N_);

        auto results = reg->registrationOfTwoVoxelsSO3(
            data1, data2,
            cellSize, useGauss, debug,
            potentialNecessaryForPeak, multipleRadii, useClahe, useHamming, benchmark
        );

        std::vector<TransformationPeak2D> out;
        out.reserve(results.size());
        for (const auto& tp : results) {
            TransformationPeak2D outTp;
            for (const auto& t : tp.potentialTranslations) {
                TranslationPeak2D outT;
                outT.translationSI = eigen2d_to_numpy(t.translationSI);
                outT.translationVoxel = eigen2i_to_numpy(t.translationVoxel);
                outT.peakHeight = t.peakHeight;
                outT.persistenceValue = t.persistenceValue;
                outT.covariance = eigen2x2_to_numpy(t.covariance);
                outTp.potentialTranslations.push_back(outT);
            }
            outTp.potentialRotation.angle = tp.potentialRotation.angle;
            outTp.potentialRotation.peakCorrelation = tp.potentialRotation.peakCorrelation;
            outTp.potentialRotation.covariance = tp.potentialRotation.covariance;
            out.push_back(outTp);
        }
        return out;
    }

    // registrationOfTwoVoxelsDirect
    std::vector<TransformationPeak2D> register_direct(
        py::array_t<double, py::array::c_style | py::array::forcecast> scan1,
        py::array_t<double, py::array::c_style | py::array::forcecast> scan2,
        double cellSize,
        bool useGauss,
        bool debug,
        double potentialNecessaryForPeak,
        bool multipleRadii,
        bool useClahe,
        bool useHamming,
        bool benchmark
    ) {
        double* data1 = numpy_to_double_array(scan1, N_ * N_);
        double* data2 = numpy_to_double_array(scan2, N_ * N_);

        auto results = reg->registrationOfTwoVoxelsDirect(
            data1, data2,
            cellSize, useGauss, debug,
            potentialNecessaryForPeak, multipleRadii, useClahe, useHamming, benchmark
        );

        std::vector<TransformationPeak2D> out;
        out.reserve(results.size());
        for (const auto& tp : results) {
            TransformationPeak2D outTp;
            for (const auto& t : tp.potentialTranslations) {
                TranslationPeak2D outT;
                outT.translationSI = eigen2d_to_numpy(t.translationSI);
                outT.translationVoxel = eigen2i_to_numpy(t.translationVoxel);
                outT.peakHeight = t.peakHeight;
                outT.persistenceValue = t.persistenceValue;
                outT.covariance = eigen2x2_to_numpy(t.covariance);
                outTp.potentialTranslations.push_back(outT);
            }
            outTp.potentialRotation.angle = tp.potentialRotation.angle;
            outTp.potentialRotation.peakCorrelation = tp.potentialRotation.peakCorrelation;
            outTp.potentialRotation.covariance = tp.potentialRotation.covariance;
            out.push_back(outTp);
        }
        return out;
    }

    int getN() const { return N_; }
};

PYBIND11_MODULE(pybind_registration_2d, m) {
    m.doc() = "pybind11 wrapper for softRegistrationClass2D";

    py::class_<RotationPeak2D>(m, "RotationPeak2D")
        .def_readwrite("angle", &RotationPeak2D::angle)
        .def_readwrite("peakCorrelation", &RotationPeak2D::peakCorrelation)
        .def_readwrite("covariance", &RotationPeak2D::covariance);

    py::class_<TranslationPeak2D>(m, "TranslationPeak2D")
        .def_readwrite("translationSI", &TranslationPeak2D::translationSI)
        .def_readwrite("translationVoxel", &TranslationPeak2D::translationVoxel)
        .def_readwrite("peakHeight", &TranslationPeak2D::peakHeight)
        .def_readwrite("persistenceValue", &TranslationPeak2D::persistenceValue)
        .def_readwrite("covariance", &TranslationPeak2D::covariance);

    py::class_<TransformationPeak2D>(m, "TransformationPeak2D")
        .def_readwrite("potentialTranslations", &TransformationPeak2D::potentialTranslations)
        .def_readwrite("potentialRotation", &TransformationPeak2D::potentialRotation);

    py::class_<SoftRegistrationWrapper2D>(m, "SoftRegistrationWrapper2D")
        .def(py::init<int>())
        .def("register_all_solutions", &SoftRegistrationWrapper2D::register_all_solutions,
             py::arg("scan1"), py::arg("scan2"),
             py::arg("cellSize"),
             py::arg("useGauss") = false,
             py::arg("debug") = false,
             py::arg("potentialNecessaryForPeak") = 0.1,
             py::arg("multipleRadii") = false,
             py::arg("useClahe") = true,
             py::arg("useHamming") = true)
        .def("register_fast", &SoftRegistrationWrapper2D::register_fast,
             py::arg("scan1"), py::arg("scan2"), py::arg("initialGuess"),
             py::arg("useInitialAngle") = true,
             py::arg("useInitialTranslation") = true,
             py::arg("cellSize"),
             py::arg("useGauss") = false,
             py::arg("debug") = false,
             py::arg("potentialNecessaryForPeak") = 0.1)
        .def("register_so3", &SoftRegistrationWrapper2D::register_so3,
             py::arg("scan1"), py::arg("scan2"),
             py::arg("cellSize"),
             py::arg("useGauss") = false,
             py::arg("debug") = false,
             py::arg("potentialNecessaryForPeak") = 0.1,
             py::arg("multipleRadii") = false,
             py::arg("useClahe") = true,
             py::arg("useHamming") = true,
             py::arg("benchmark") = false)
        .def("register_direct", &SoftRegistrationWrapper2D::register_direct,
             py::arg("scan1"), py::arg("scan2"),
             py::arg("cellSize"),
             py::arg("useGauss") = false,
             py::arg("debug") = false,
             py::arg("potentialNecessaryForPeak") = 0.1,
             py::arg("multipleRadii") = false,
             py::arg("useClahe") = true,
             py::arg("useHamming") = true,
             py::arg("benchmark") = false)
        .def_property_readonly("N", &SoftRegistrationWrapper2D::getN);
}
