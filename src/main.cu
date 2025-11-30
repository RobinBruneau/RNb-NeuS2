/*
* Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

/** @file   main.cu
 *  @author Thomas Müller, NVIDIA
 */

#include <neural-graphics-primitives/testbed.h>

#include <tiny-cuda-nn/common.h>

#include <args/args.hxx>

#include <filesystem/path.h>

#ifdef __linux__
#include <unistd.h>
#include <linux/limits.h>
#elif _WIN32
#include <windows.h>
#endif

using namespace args;
using namespace ngp;
using namespace std;
using namespace tcnn;
namespace fs = ::filesystem;

// Global variable to store the executable directory path
static fs::path g_executable_dir;

// Function to set the executable directory (called once at startup)
void set_executable_dir(const fs::path& dir) {
    g_executable_dir = dir;
}

// Function to get the executable directory (can be called from anywhere)
fs::path get_executable_dir_global() {
    return g_executable_dir;
}

// Function to get the directory containing the executable
fs::path get_executable_dir() {
#ifdef __linux__
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    if (count != -1) {
        result[count] = '\0';
        fs::path exe_path(result);
        return exe_path.parent_path();
    }
#elif _WIN32
    char result[MAX_PATH];
    DWORD count = GetModuleFileNameA(NULL, result, MAX_PATH);
    if (count != 0) {
        fs::path exe_path(result);
        return exe_path.parent_path();
    }
#endif
    // Fallback to current directory if we can't determine executable path
    tlog::warning() << "Could not determine executable directory, using current directory";
    return fs::path(".");
}


int main(int argc, char** argv) {
    // Set the executable directory for use throughout the application
    set_executable_dir(get_executable_dir());

    ArgumentParser parser{
            "neural graphics primitives\n"
            "version " NGP_VERSION,
            "",
    };

    HelpFlag help_flag{
            parser,
            "HELP",
            "Display this help menu.",
            {'h', "help"},
    };

    Flag lone_flag{
            parser,
            "L_ONE",
            "Activate l_one between colors !",
            {"lone"},
    };

    Flag supernormal_flag{
            parser,
            "SUPERNORMAL",
            "Activate Supernormal loss function",
            {"supernormal"},
    };

    Flag no_rgbplus_flag{
            parser,
            "NO_RGB_PLUS",
            "Deactivate rgb normalisation",
            {"no-rgbplus"},
    };

    Flag disable_snap_to_center_flag{
            parser,
            "DISABLE_SNAP_TO_CENTER",
            "Disable snap to center for the camera !",
            {"disable-snap-to-center"},
    };

    Flag relu_flag{
            parser,
            "RELU",
            "Activate ReLU for shading !",
            {"relu"},
    };

    Flag bce_flag{
            parser,
            "BCE",
            "Apply BCE mask loss instead of Sigmoid BCE !",
            {"bce"},
    };

    ValueFlag<string> network_config_flag{
            parser,
            "CONFIG",
            "Path to the network config. Uses the scene's default if unspecified.",
            {'n', 'c', "network", "config"},
    };

    Flag no_gui_flag{
            parser,
            "NO_GUI",
            "Disables the GUI and instead reports training progress on the command line.",
            {"no-gui"},
    };

    Flag save_mesh_flag{
            parser,
            "SAVE_MESH",
            "Save as a mesh when it's done.",
            {"save-mesh"},
    };

    Flag save_snapshot_flag{
            parser,
            "SAVE_SNAPSHOT",
            "Save as a snapshot when it's done.",
            {"save-snapshot"},
    };

    Flag opti_lights_flag{
            parser,
            "OPTI-LIGHTS",
            "Use optimal lights per pixels",
            {"opti-lights"},
    };

    Flag no_albedo_flag{
            parser,
            "no-albedo",
            "To use when you don't want to optimize the albedo",
            {"no-albedo"},
    };

    Flag no_train_flag{
            parser,
            "NO_TRAIN",
            "Disables training on startup.",
            {"no-train"},
    };

    Flag free_memory_flag{
        parser,
        "FREE-MEMORY",
        "Free images from GPU memory",
        {"free-memory"},
};

    ValueFlag<string> scene_flag{
            parser,
            "SCENE",
            "The scene to load. Can be NeRF dataset, a *.obj mesh for training a SDF, an image, or a *.nvdb volume.",
            {'s', "scene"},
    };

    ValueFlag<string> snapshot_flag{
            parser,
            "SNAPSHOT",
            "Optional snapshot to load upon startup.",
            {"snapshot"},
    };

    ValueFlag<uint32_t> width_flag{
            parser,
            "WIDTH",
            "Resolution width of the GUI.",
            {"width"},
    };

    ValueFlag<uint32_t> save_each_flag{
            parser,
            "SAVE_EACH",
            "Save mesh each X number of iterations",
            {"save-each"},
    };

    ValueFlag<uint32_t> resolution_flag{
            parser,
            "RESOLUTION",
            "Resolution used for marching cube",
            {"resolution"},
    };

    ValueFlag<uint32_t> height_flag{
            parser,
            "HEIGHT",
            "Resolution height of the GUI.",
            {"height"},
    };


    ValueFlag<uint32_t> max_iter_flag{
            parser,
            "MAXITER",
            "Maximum number of iterations.",
            {"maxiter"},
    };

    Flag version_flag{
            parser,
            "VERSION",
            "Display the version of neural graphics primitives.",
            {'v', "version"},
    };

    ValueFlag<float> mask_weight_flag{
            parser,
            "MASK_WEIGHT",
            "Mask weight.",
            {"mask-weight"},
    };

    // Add the fractional-training parameter
    ValueFlag<int> fractional_training_flag{
            parser,
            "FRACTIONAL_TRAINING",
            "Step for fractional training",
            {"fractional-training"},
    };


    // Parse command line arguments and react to parsing
    // errors using exceptions.
    try {
        parser.ParseCLI(argc, argv);
    } catch (const Help&) {
        cout << parser;
        return 0;
    } catch (const ParseError& e) {
        cerr << e.what() << endl;
        cerr << parser;
        return -1;
    } catch (const ValidationError& e) {
        cerr << e.what() << endl;
        cerr << parser;
        return -2;
    }

    if (version_flag) {
        tlog::none() << "neural graphics primitives version " NGP_VERSION;
        return 0;
    }

    ETestbedMode mode = ETestbedMode::Nerf;

    Testbed testbed{mode};
    if (max_iter_flag){
        testbed.set_max_iter(get(max_iter_flag));
    }

    
    tlog::info() << "Number of iterations : " << testbed.get_max_iter();

    if (scene_flag) {
        fs::path scene_path = get(scene_flag);
        if (!scene_path.exists()) {
            tlog::error() << "Scene path " << scene_path << " does not exist.";
            return 1;
        }
        testbed.load_training_data(scene_path.str());
    }

    std::string mode_str = "nerf";

    if (snapshot_flag) {
        // Load network from a snapshot if one is provided
        fs::path snapshot_path = get(snapshot_flag);
        if (!snapshot_path.exists()) {
            tlog::error() << "Snapshot path " << snapshot_path << " does not exist.";
            return 1;
        }

        testbed.load_snapshot(snapshot_path.str());
        testbed.m_train = true;
        printf("*******Loaded snapshot succeed!\n");
    } else {
        // Otherwise, load the network config and prepare for training
        fs::path network_config_path = get_executable_dir().parent_path() / "configs" / mode_str;
        if (network_config_flag) {
            auto network_config_str = get(network_config_flag);
            if ((network_config_path/network_config_str).exists()) {
                network_config_path = network_config_path/network_config_str;
            } else {
                network_config_path = network_config_str;
            }
        } else {
            network_config_path = network_config_path/"base.json";
        }

        if (!network_config_path.exists()) {
            tlog::error() << "Network config path " << network_config_path << " does not exist.";
            return 1;
        }
        else{
            tlog::success() << "Network config path " << network_config_path << " found!";
        }
        testbed.reload_network_from_file(network_config_path.str());
        testbed.m_train = !no_train_flag;
    }

    bool gui = !no_gui_flag;
#ifndef NGP_GUI
    gui = false;
#endif

    if (gui) {
        testbed.init_window(width_flag ? get(width_flag) : 1920, height_flag ? get(height_flag) : 1080);
    }

    if (mask_weight_flag){
        testbed.set_mask_weight(get(mask_weight_flag));
    }
    
    


    if (fractional_training_flag){

        if(max_iter_flag){
            uint32_t value_frac = get(fractional_training_flag);
            
            if (value_frac < get(max_iter_flag)){
                testbed.apply_fractional_training(value_frac);
            }
            else{
                tlog::error() << "The integer must be lower than max-iter!";
                return 1;
            }
        
            
        }
        else{
            tlog::error() << "fractional-training works with max-iter.";
            return 1; 
        }
        
    }

    if (! lone_flag){
        testbed.apply_L2();
    }

    if (supernormal_flag){
        testbed.apply_supernormal();
    }
    if (! no_rgbplus_flag){
        testbed.apply_rgbplus();
    }
    if (disable_snap_to_center_flag){
        testbed.disable_snap_to_center();
    }


    if (bce_flag){
        testbed.apply_bce();
    }
    if (relu_flag){
        testbed.apply_relu();
    }

    if (opti_lights_flag){
        testbed.apply_light_opti();
    }

    if (no_albedo_flag){
        testbed.apply_no_albedo(true);
    }

    if (save_each_flag){
        testbed.save_each(get(save_each_flag));
    }

    // Get the scene path and construct output path
    std::string scene_path_str = get(scene_flag);
    // Remove trailing slash if present
    if (scene_path_str.back() == '/') {
        scene_path_str.pop_back();
    }
    fs::path output_path = fs::path(scene_path_str) / "output";

    // Build mesh filename in output folder
    static char obj_filename_buf[256] = "";
    if (obj_filename_buf[0] == '\0') {
        snprintf(obj_filename_buf, sizeof(obj_filename_buf), "%s/mesh_%d.obj", 
                output_path.str().c_str(), testbed.get_max_iter());
    }

    Eigen::Vector3i resMesh(512, 512, 512);
    if (resolution_flag){
        resMesh[0] = get(resolution_flag);
        resMesh[1] = get(resolution_flag);
        resMesh[2] = get(resolution_flag);
    }

    // This parameter is used for save_each functionality - also use output folder
    static char objmesh_prefix[256] = "";
    if (objmesh_prefix[0] == '\0') {
        snprintf(objmesh_prefix, sizeof(objmesh_prefix), "%s/mesh_", output_path.str().c_str());
    }

    testbed.add_mesh_save_params(resMesh,objmesh_prefix);

    
    // Render/training loop
    if (!no_train_flag){
        while (testbed.frame()) {
            if (!gui) {
                if (testbed.m_training_step % 100 == 0){
                    tlog::info() << "iteration=" << testbed.m_training_step << " loss=" << testbed.m_loss_scalar.val();
                // tlog::info() << "iteration=" << testbed.m_training_step << " loss=" << testbed.m_loss_scalar.val() << " lr=" << testbed.m_optimizer.learning_rate();
                }
            }
        }
    }

    if (save_mesh_flag){
        if (free_memory_flag){
            testbed.free_unnecessary_gpu_memory();
            std::this_thread::sleep_for(std::chrono::seconds(10));
        }
        testbed.compute_and_save_marching_cubes_mesh(obj_filename_buf,resMesh,{},0.0f,false);
    }

    // Save snapshot in output folder
    std::string snapshot_filename = output_path.str() + "/snapshot_" + to_string(testbed.get_max_iter()) + ".msgpack";
    if(save_snapshot_flag){
        tlog::info() << "Saving Snapshot !";
        tlog::info() << snapshot_filename;
        testbed.save_snapshot(snapshot_filename,false);
    }


}
