#define _USE_MATH_DEFINES
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <filesystem> // C++17 feature
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <nlohmann/json.hpp>
namespace fs = std::filesystem;


void readMaskImages(const std::string& folder, std::vector<cv::Mat>& images) {

    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(folder)) {
        files.push_back(entry.path().string());
    }
    // Sort files alphabetically
    std::sort(files.begin(), files.end());
    // Print sorted files
    for (const auto& file : files) {
        cv::Mat image = cv::imread(file, cv::IMREAD_GRAYSCALE); // Read as grayscale
        if (image.empty()) {
            std::cerr << "Failed to read image: " << file << std::endl;
            continue;
        }
        std::vector<cv::Mat> channels;
        cv::split(image, channels);

        // Create a new image with only the first channel
        cv::Mat first_channel = channels[0];
        images.push_back(first_channel);

    }
}

void readImages(const std::string& folder, std::vector<cv::Mat>& images) {
    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(folder)) {
        files.push_back(entry.path().string());
    }
    // Sort files alphabetically
    std::sort(files.begin(), files.end());
    // Print sorted files
    for (const auto& file : files) {
        cv::Mat image = cv::imread(file);
        if (image.empty()) {
            std::cerr << "Failed to read image: " << file << std::endl;
            continue;
        }
        
        images.push_back(image);
    }
}

void readImagesNormales(const std::string& folder, std::vector<cv::Mat>& images) {
    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(folder)) {
        files.push_back(entry.path().string());
    }
    // Sort files alphabetically
    std::sort(files.begin(), files.end());
    // Print sorted files
    for (const auto& file : files) {
        cv::Mat bgrImage = cv::imread(file,-1);
        if (bgrImage.empty()) {
            std::cerr << "Failed to read image: " << file << std::endl;
            continue;
        }

        cv::Mat rgbImage;
        cv::cvtColor(bgrImage, rgbImage, cv::COLOR_BGR2RGB);
        images.push_back(rgbImage);
    }
}


void postprocessNormals(std::vector<cv::Mat>& normalsImages) {
    for (cv::Mat& image : normalsImages) {
        image.convertTo(image, CV_32F, 1.0 / 255.0);
        image *= 2.0f; // Scale to [0, 2]
        for (int y = 0; y < image.rows; y++){
            for (int x = 0; x < image.cols; x++){
                cv::Vec3f& color = image.at<cv::Vec3f>(y, x);
                color[0] -= 1.0f;
                color[1] -= 1.0f;
                color[2] -= 1.0f;
            }
        }
    }
}

Eigen::Vector3f cvVec3fToEigenVec3f(const cv::Vec3f& cvVec) {
    return Eigen::Vector3f(
        static_cast<float>(cvVec[0]) ,
        static_cast<float>(cvVec[1]) ,
        static_cast<float>(cvVec[2])
    );
}

Eigen::Matrix3f jsonToMatrix(nlohmann::json R) {
    Eigen::Matrix3f RR;
    for (int k = 0; k < 3; k++) {
        for (int j = 0; j < 3; j++) {
            RR(k, j) = R[k][j];
        }
    }
    return RR;
}

int main(int argc, char* argv[]) {

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <inputFolder> <outputFolder>" << std::endl;
        return 1;
    }

    std::string mainFolder = argv[1];
    std::string outputFolder = argv[2];


    auto start = std::chrono::steady_clock::now();

    //std::string mainFolder = "D:/PhD/Dropbox/CVPR_2024/data/DiLiGenT-MV_SIM/DiLiGenT-MV/buddhaPNG/Pi3-Opt/";
    std::string maskFolder = mainFolder + "mask";
    std::string albedoFolder = mainFolder + "albedo";
    std::string normalsFolder = mainFolder + "normal";

    // Open the JSON file for reading
    std::ifstream file(mainFolder+"cameras.json");
    nlohmann::json j;
    file >> j;
    auto all_R = j["R"];
    

    std::vector<cv::Mat> maskImages;
    readMaskImages(maskFolder, maskImages);

    std::vector<cv::Mat> albedoImages;
    readImages(albedoFolder, albedoImages);

    std::vector<cv::Mat> normalsImages;
    readImagesNormales(normalsFolder, normalsImages);

    // Post-process normals images
    postprocessNormals(normalsImages);

    // Now maskImages, albedoImages, and normalsImages contain the loaded and processed images
    // You can further process or use them as needed

    Eigen::Vector3f tilt(0*M_PI/180.f, 120 * M_PI / 180.f, 240 * M_PI / 180.f);
    Eigen::Vector3f slant(54.74 * M_PI / 180.f, 54.74 * M_PI / 180.f, 54.74 * M_PI / 180.f);

    Eigen::MatrixXf u(3, 3);
    u.row(0) = -((slant.array().sin()) * (tilt.array().cos())).matrix();
    u.row(1) = -((slant.array().sin()) * (tilt.array().sin())).matrix();
    u.row(2) = -slant.array().cos().matrix();

    std::vector<float> lights;
    std::vector<float> lights_60;
    
    


    //for (int k = 0; k <1; k++) {
    for (int k = 0; k < normalsImages.size(); k++) {

        auto Rotation = jsonToMatrix(all_R[k]);

        auto im_normal = normalsImages[k];
        auto im_albedo = albedoImages[k];
        auto im_mask = maskImages[k];

        std::cout << im_normal.at<cv::Vec3f>(18, 303) << std::endl;

        cv::Mat im1(im_albedo.rows, im_albedo.cols, CV_16UC3);
        cv::Mat im2(im_albedo.rows, im_albedo.cols, CV_16UC3);
        cv::Mat im3(im_albedo.rows, im_albedo.cols, CV_16UC3);

        cv::Mat im1_60(im_albedo.rows, im_albedo.cols, CV_16UC3);
        cv::Mat im2_60(im_albedo.rows, im_albedo.cols, CV_16UC3);
        cv::Mat im3_60(im_albedo.rows, im_albedo.cols, CV_16UC3);

        Eigen::MatrixXf light_directions_60_world = Rotation.transpose() * u;

        std::cout << im_normal.rows << std::endl;
        // Iterate over each row
        for (int y = 0; y < im_normal.rows; ++y) {
        //for (int y = 18; y <19; ++y) {
            // Iterate over each column
            for (int x = 0; x < im_normal.cols; ++x) {
            //for (int x = 303; x < 304; ++x) {
                // Access pixel value at (x, y)

                int mask_value = static_cast<int>(im_mask.at<uchar>(y, x));

                if (mask_value > 0) {

                    cv::Vec3f value = im_normal.at<cv::Vec3f>(y, x);
                    Eigen::Vector3f normal_value = cvVec3fToEigenVec3f(im_normal.at<cv::Vec3f>(y, x));  
                    normal_value.normalize();
                    normal_value(1) *= -1.0f;
                    normal_value(2) *= -1.0f;
                    cv::Vec3b albedo_value = im_albedo.at<cv::Vec3b>(y, x);

                    Eigen::Matrix3f outer_prod = normal_value * normal_value.transpose();

                    Eigen::JacobiSVD<Eigen::Matrix3f> svd;
                    svd.compute(outer_prod, Eigen::ComputeFullU | Eigen::ComputeFullV);
                    Eigen::Matrix3f U = svd.matrixU();
                    //std::cout << U << std::endl << std::endl;
                    float detU = U.determinant();
                    Eigen::Matrix3f R;
                    if (detU < 0) {
                        R.col(0) = -U.col(1);
                        R.col(1) = U.col(2);
                        R.col(2) = U.col(0);
                    }
                    else {
                        R.col(0) = U.col(1);
                        R.col(1) = U.col(2);
                        R.col(2) = U.col(0);
                    }
                    
                    if (R(2, 2) < 0) {
                        R.col(0) *= -1.0f;
                        R.col(2) *= -1.0f;
                    }
                    //std::cout << R << std::endl;
                    Eigen::MatrixXf light_directions = (R * u);
                    //std::cout << light_directions << std::endl ;

                    //std::cout << k << " " << y << " " << x << " || " << normal_value << " || " << normal_value.norm() << std::endl;
                    //std::cout << light_directions.col(0) << std::endl << std::endl;
                    //std::cout << light_directions.col(1) << std::endl << std::endl;
                    //std::cout << light_directions.col(2) << std::endl << std::endl;

                    float shading1 = light_directions.col(0).dot(normal_value);
                    float shading2 = light_directions.col(1).dot(normal_value);
                    float shading3 = light_directions.col(2).dot(normal_value);

                    float shading1_60 = std::max(u.col(0).dot(normal_value), 0.0f);
                    float shading2_60 = std::max(u.col(1).dot(normal_value), 0.0f);
                    float shading3_60 = std::max(u.col(2).dot(normal_value), 0.0f);

                    if (shading1 < 0.0f){
                        shading1 *= -1.0f;
                        light_directions.col(0) *= -1.0f;
                    }

                    if (shading2 < 0.0f) {
                        shading2 *= -1.0f;
                        light_directions.col(1) *= -1.0f;
                    }

                    if (shading3 < 0.0f) {
                        shading3 *= -1.0f;
                        light_directions.col(2) *= -1.0f;
                    }


                    //std::cout << shading1 << " " << shading2 << " " << shading3 << " " << std::endl << std::endl << std::endl << std::endl << std::endl << std::endl;

                    Eigen::MatrixXf light_directions_world = Rotation.transpose() * light_directions;

                    

                    lights.push_back(light_directions_world.col(0)(0));
                    lights.push_back(light_directions_world.col(0)(1));
                    lights.push_back(light_directions_world.col(0)(2));

                    lights.push_back(light_directions_world.col(1)(0));
                    lights.push_back(light_directions_world.col(1)(1));
                    lights.push_back(light_directions_world.col(1)(2));

                    lights.push_back(light_directions_world.col(2)(0));
                    lights.push_back(light_directions_world.col(2)(1));
                    lights.push_back(light_directions_world.col(2)(2));
                    
                    im1.at<cv::Vec3w>(y, x) = cv::Vec3w(shading1 * static_cast<uint16_t>(albedo_value[0]) * 65535.0f / 255.0f, shading1 * static_cast<uint16_t>(albedo_value[1]) * 65535.0f / 255.0f, shading1 * static_cast<uint16_t>(albedo_value[2]) * 65535.0f / 255.0f) ;
                    im2.at<cv::Vec3w>(y, x) = cv::Vec3w(shading2 * static_cast<uint16_t>(albedo_value[0]) * 65535.0f / 255.0f, shading2 * static_cast<uint16_t>(albedo_value[1]) * 65535.0f / 255.0f, shading2 * static_cast<uint16_t>(albedo_value[2]) * 65535.0f / 255.0f) ;
                    im3.at<cv::Vec3w>(y, x) = cv::Vec3w(shading3 * static_cast<uint16_t>(albedo_value[0]) * 65535.0f / 255.0f, shading3 * static_cast<uint16_t>(albedo_value[1]) * 65535.0f / 255.0f, shading3 * static_cast<uint16_t>(albedo_value[2]) * 65535.0f / 255.0f) ;

                    im1_60.at<cv::Vec3w>(y, x) = cv::Vec3w(shading1_60 * static_cast<uint16_t>(albedo_value[0]) * 65535.0f / 255.0f, shading1_60 * static_cast<uint16_t>(albedo_value[1]) * 65535.0f / 255.0f, shading1_60 * static_cast<uint16_t>(albedo_value[2]) * 65535.0f / 255.0f);
                    im2_60.at<cv::Vec3w>(y, x) = cv::Vec3w(shading2_60 * static_cast<uint16_t>(albedo_value[0]) * 65535.0f / 255.0f, shading2_60 * static_cast<uint16_t>(albedo_value[1]) * 65535.0f / 255.0f, shading2_60 * static_cast<uint16_t>(albedo_value[2]) * 65535.0f / 255.0f);
                    im3_60.at<cv::Vec3w>(y, x) = cv::Vec3w(shading3_60 * static_cast<uint16_t>(albedo_value[0]) * 65535.0f / 255.0f, shading3_60 * static_cast<uint16_t>(albedo_value[1]) * 65535.0f / 255.0f, shading3_60 * static_cast<uint16_t>(albedo_value[2]) * 65535.0f / 255.0f);


                    int e = 0;

                }
                else {
                    im1.at<cv::Vec3w>(y, x) = cv::Vec3w(0, 0, 0);
                    im2.at<cv::Vec3w>(y, x) = cv::Vec3w(0, 0, 0);
                    im3.at<cv::Vec3w>(y, x) = cv::Vec3w(0, 0, 0);

                    im1_60.at<cv::Vec3w>(y, x) = cv::Vec3w(0, 0, 0);
                    im2_60.at<cv::Vec3w>(y, x) = cv::Vec3w(0, 0, 0);
                    im3_60.at<cv::Vec3w>(y, x) = cv::Vec3w(0, 0, 0);

                    lights.push_back(0.0f);
                    lights.push_back(0.0f);
                    lights.push_back(0.0f);

                    lights.push_back(0.0f);
                    lights.push_back(0.0f);
                    lights.push_back(0.0f);

                    lights.push_back(0.0f);
                    lights.push_back(0.0f);
                    lights.push_back(0.0f);
                }

            }
        }

        std::ostringstream oss;
        oss << std::setfill('0') << std::setw(3) << k;
        cv::imwrite(outputFolder + "/image/" + oss.str() + "_000.png", im1);
        cv::imwrite(outputFolder + "/image/" + oss.str() + "_001.png", im2);
        cv::imwrite(outputFolder + "/image/" + oss.str() + "_002.png", im3);


        cv::imwrite(outputFolder + "/image_60/" + oss.str() + "_000.png", im1_60);
        cv::imwrite(outputFolder + "/image_60/" + oss.str() + "_001.png", im2_60);
        cv::imwrite(outputFolder + "/image_60/" + oss.str() + "_002.png", im3_60);

        lights_60.push_back(light_directions_60_world.col(0)(0));
        lights_60.push_back(light_directions_60_world.col(0)(1));
        lights_60.push_back(light_directions_60_world.col(0)(2));

        lights_60.push_back(light_directions_60_world.col(1)(0));
        lights_60.push_back(light_directions_60_world.col(1)(1));
        lights_60.push_back(light_directions_60_world.col(1)(2));

        lights_60.push_back(light_directions_60_world.col(2)(0));
        lights_60.push_back(light_directions_60_world.col(2)(1));
        lights_60.push_back(light_directions_60_world.col(2)(2));
        
        std::cout << "Images saved !" << std::endl;
        

    }

    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;

    std::cout << "Elapsed time: " << std::chrono::duration <double, std::milli>(diff).count() << " ms" << std::endl;


    // Create a JSON object and assign the vector to it
    nlohmann::json json_data = lights;


    // Write the JSON to a file
    std::ofstream ff(outputFolder+"/lights.json");
    if (ff.is_open()) {
        ff << json_data; // Dump JSON with indentation for readability
        ff.close();
        std::cout << "JSON data has been written to output.json" << std::endl;
    }
    else {
        std::cerr << "Error: Unable to open file for writing." << std::endl;
    }

    // Create a JSON object and assign the vector to it
    //nlohmann::json json_data_60 = lights_60;
    nlohmann::json json_data_60 = lights_60;

    // Write the JSON to a file
    std::ofstream ff_60(outputFolder + "/lights_60.json");
    if (ff_60.is_open()) {
        ff_60 << json_data_60; // Dump JSON with indentation for readability
        ff_60.close();
        std::cout << "JSON data has been written to output.json" << std::endl;
    }
    else {
        std::cerr << "Error: Unable to open file for writing." << std::endl;
    }

    
    return 0;
}
