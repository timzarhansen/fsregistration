#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <memory>

#include "softRegistrationClass3D.h"

namespace py = pybind11;

class SoftRegistrationWrapper {
private:
    std::unique_ptr<softRegistrationClass3D> reg;
    int N_;

public:
    SoftRegistrationWrapper(int N, int bwOut, int bwIn, int degLim)
        : N_(N) {
        reg.reset(new softRegistrationClass3D(N, bwOut, bwIn, degLim));
    }

    std::vector<transformationPeakfs3D> registerVoxels(
        py::array_t<double, py::array::c_style | py::array::forcecast> voxelData1,
        py::array_t<double, py::array::c_style | py::array::forcecast> voxelData2,
        bool debug,
        bool useClahe,
        bool timeStuff,
        double sizeVoxel,
        double r_min,
        double r_max,
        double level_potential_rotation,
        double level_potential_translation,
        bool set_r_manual,
        int normalization
    ) {
        auto buf1 = voxelData1.request();
        auto buf2 = voxelData2.request();

        int expectedSize = N_ * N_ * N_;
        if (buf1.size != expectedSize || buf2.size != expectedSize) {
            throw std::runtime_error(
                "Voxel array size mismatch. Expected " + std::to_string(expectedSize) +
                ", got " + std::to_string(buf1.size) + " and " + std::to_string(buf2.size)
            );
        }

        double* data1 = static_cast<double*>(buf1.ptr);
        double* data2 = static_cast<double*>(buf2.ptr);

        return reg->sofftRegistrationVoxel3DListOfPossibleTransformations(
            data1, data2,
            debug, useClahe, timeStuff, sizeVoxel,
            r_min, r_max,
            level_potential_rotation, level_potential_translation,
            set_r_manual, normalization
        );
    }

    int getN() const { return N_; }
};

PYBIND11_MODULE(pybind_registration_3d, m) {
    m.doc() = "pybind11 wrapper for softRegistrationClass3D";

    py::class_<translationPeak3D>(m, "TranslationPeak3D")
        .def_readwrite("xTranslation", &translationPeak3D::xTranslation)
        .def_readwrite("yTranslation", &translationPeak3D::yTranslation)
        .def_readwrite("zTranslation", &translationPeak3D::zTranslation)
        .def_readwrite("persistence", &translationPeak3D::persistence)
        .def_readwrite("levelPotential", &translationPeak3D::levelPotential)
        .def_readwrite("correlationHeight", &translationPeak3D::correlationHeight);

    py::class_<rotationPeak4D>(m, "RotationPeak4D")
        .def_readwrite("x", &rotationPeak4D::x)
        .def_readwrite("y", &rotationPeak4D::y)
        .def_readwrite("z", &rotationPeak4D::z)
        .def_readwrite("w", &rotationPeak4D::w)
        .def_readwrite("persistence", &rotationPeak4D::persistence)
        .def_readwrite("levelPotential", &rotationPeak4D::levelPotential)
        .def_readwrite("correlationHeight", &rotationPeak4D::correlationHeight);

    py::class_<transformationPeakfs3D>(m, "TransformationPeakFS3D")
        .def_readwrite("potentialTranslations", &transformationPeakfs3D::potentialTranslations)
        .def_readwrite("potentialRotation", &transformationPeakfs3D::potentialRotation);

    py::class_<SoftRegistrationWrapper>(m, "SoftRegistrationWrapper")
        .def(py::init<int, int, int, int>(),
             py::arg("N"), py::arg("bwOut"), py::arg("bwIn"), py::arg("degLim"))
        .def("registerVoxels", &SoftRegistrationWrapper::registerVoxels,
             py::arg("voxelData1"), py::arg("voxelData2"),
             py::arg("debug") = false,
             py::arg("useClahe") = true,
             py::arg("timeStuff") = false,
             py::arg("sizeVoxel") = 1.0,
             py::arg("r_min") = 0.0,
             py::arg("r_max") = 0.0,
             py::arg("level_potential_rotation") = 0.01,
             py::arg("level_potential_translation") = 0.1,
             py::arg("set_r_manual") = false,
             py::arg("normalization") = 0)
        .def_property_readonly("N", &SoftRegistrationWrapper::getN);
}
