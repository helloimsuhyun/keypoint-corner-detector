// Dataset I/O only: Harris responses, NMS/refinement and normals are PCL's.
#include <pcl/features/normal_3d.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/pcl_config.h>
#include <pcl/point_types.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--version") {
            std::cout << PCL_VERSION_PRETTY << '\n';
            return 0;
        }
        if (argc < 5) throw std::runtime_error("Usage: pcl_baseline harris|normals input_xyz_m output radius_m [threshold refine]");
        std::string mode = argv[1];
        double radius = std::stod(argv[4]);
        if (!std::isfinite(radius) || radius <= 0) throw std::runtime_error("Radius must be positive");
        auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        std::ifstream input(argv[2]);
        if (!input) throw std::runtime_error("Cannot open input cloud");
        pcl::PointXYZ p;
        while (input >> p.x >> p.y >> p.z) {
            if (!pcl::isFinite(p)) throw std::runtime_error("Non-finite input point");
            cloud->push_back(p);
        }
        if (!input.eof() || cloud->empty()) throw std::runtime_error("Malformed/empty input cloud");
        std::ofstream output(argv[3]);
        if (!output) throw std::runtime_error("Cannot open output");
        output << std::setprecision(17);
        if (mode == "harris" && argc == 7) {
            double threshold = std::stod(argv[5]);
            if (!std::isfinite(threshold)) throw std::runtime_error("Threshold must be finite");
            pcl::HarrisKeypoint3D<pcl::PointXYZ, pcl::PointXYZI> detector;
            detector.setInputCloud(cloud);
            detector.setMethod(decltype(detector)::HARRIS);
            detector.setRadius(radius);
            detector.setThreshold(threshold);
            detector.setNonMaxSupression(true);
            detector.setRefine(std::stoi(argv[6]) != 0);
            detector.setNumberOfThreads(1);
            pcl::PointCloud<pcl::PointXYZI> corners;
            detector.compute(corners);
            output << "x_m,y_m,z_m,response\n";
            for (const auto& c : corners)
                output << c.x << ',' << c.y << ',' << c.z << ',' << c.intensity << '\n';
        } else if (mode == "normals" && argc == 5) {
            pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> estimator;
            estimator.setInputCloud(cloud);
            estimator.setRadiusSearch(radius);
            estimator.setViewPoint(0, 0, 0);
            pcl::PointCloud<pcl::Normal> normals;
            estimator.compute(normals);
            output << "x_m,y_m,z_m,nx,ny,nz,input_index\n";
            for (std::size_t i = 0; i < cloud->size(); ++i) {
                const auto& n = normals[i];
                if (std::isfinite(n.normal_x) && std::isfinite(n.normal_y) && std::isfinite(n.normal_z)) {
                    const auto& q = (*cloud)[i];
                    output << q.x << ',' << q.y << ',' << q.z << ','
                           << n.normal_x << ',' << n.normal_y << ',' << n.normal_z << ',' << i << '\n';
                }
            }
        } else throw std::runtime_error("Unknown mode or wrong argument count");
        if (!output) throw std::runtime_error("Output write failed");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
