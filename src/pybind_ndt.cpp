#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>

#include <pcl/registration/ndt.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

namespace py = pybind11;

struct NDTResult {
    double fitness;
    bool has_converged;
    int final_num_iteration;
    py::array_t<double> transformation;
};

class PCLNDTWrapper {
public:
    NDTResult align(
        py::array_t<double, py::array::c_style | py::array::forcecast> source_pts,
        py::array_t<double, py::array::c_style | py::array::forcecast> target_pts,
        double resolution,
        double step_size,
        double transformation_epsilon,
        int max_iterations,
        py::array_t<double, py::array::c_style | py::array::forcecast> initial_guess
    ) {
        auto src_buf = source_pts.request();
        auto tgt_buf = target_pts.request();

        if (src_buf.ndim != 2 || tgt_buf.ndim != 2) {
            throw std::runtime_error("Input arrays must be 2D (N x 3)");
        }
        if (src_buf.shape[1] != 3 || tgt_buf.shape[1] != 3) {
            throw std::runtime_error("Input arrays must have shape (N, 3)");
        }

        int n_src = src_buf.shape[0];
        int n_tgt = tgt_buf.shape[0];
        double* src_ptr = static_cast<double*>(src_buf.ptr);
        double* tgt_ptr = static_cast<double*>(tgt_buf.ptr);

        pcl::PointCloud<pcl::PointXYZ>::Ptr source(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr target(new pcl::PointCloud<pcl::PointXYZ>());

        source->resize(n_src);
        target->resize(n_tgt);

        for (int i = 0; i < n_src; i++) {
            source->points[i].x = static_cast<float>(src_ptr[i * 3]);
            source->points[i].y = static_cast<float>(src_ptr[i * 3 + 1]);
            source->points[i].z = static_cast<float>(src_ptr[i * 3 + 2]);
        }
        for (int i = 0; i < n_tgt; i++) {
            target->points[i].x = static_cast<float>(tgt_ptr[i * 3]);
            target->points[i].y = static_cast<float>(tgt_ptr[i * 3 + 1]);
            target->points[i].z = static_cast<float>(tgt_ptr[i * 3 + 2]);
        }

        pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
        ndt.setResolution(resolution);
        ndt.setStepSize(step_size);
        ndt.setTransformationEpsilon(transformation_epsilon);
        ndt.setMaximumIterations(max_iterations);

        ndt.setInputSource(source);
        ndt.setInputTarget(target);

        pcl::PointCloud<pcl::PointXYZ> output;

        Eigen::Matrix4f init;
        auto guess_buf = initial_guess.request();
        if (guess_buf.size == 16) {
            double* guess_ptr = static_cast<double*>(guess_buf.ptr);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    init(i, j) = static_cast<float>(guess_ptr[i * 4 + j]);
        } else {
            init = Eigen::Matrix4f::Identity();
        }
        ndt.align(output, init);

        NDTResult result;
        result.fitness = static_cast<double>(ndt.getFitnessScore());
        result.has_converged = ndt.hasConverged();
        result.final_num_iteration = ndt.getFinalNumIteration();

        Eigen::Matrix4f T = ndt.getFinalTransformation();
        result.transformation = py::array_t<double>({4, 4});
        auto buf = result.transformation.request();
        double* ptr = static_cast<double*>(buf.ptr);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                ptr[i * 4 + j] = static_cast<double>(T(i, j));
            }
        }

        return result;
    }
};

PYBIND11_MODULE(pybind_ndt, m) {
    m.doc() = "PCL Normal Distributions Transform via pybind11";

    py::class_<NDTResult>(m, "NDTResult")
        .def_readonly("fitness", &NDTResult::fitness)
        .def_readonly("has_converged", &NDTResult::has_converged)
        .def_readonly("final_num_iteration", &NDTResult::final_num_iteration)
        .def_readonly("transformation", &NDTResult::transformation);

    py::class_<PCLNDTWrapper>(m, "PCLNDTWrapper")
        .def(py::init<>())
        .def("align", &PCLNDTWrapper::align,
            py::arg("source_points"),
            py::arg("target_points"),
            py::arg("resolution") = 1.0,
            py::arg("step_size") = 0.1,
            py::arg("transformation_epsilon") = 0.01,
            py::arg("max_iterations") = 35,
            py::arg("initial_guess") = py::array_t<double>(0));
}
