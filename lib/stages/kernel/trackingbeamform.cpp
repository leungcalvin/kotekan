#include "trackingbeamform.hpp"

#include "Config.hpp"             // for Config
#include "chimeMetadata.hpp"      // for MAX_NUM_BEAMS
#include "gpuCommand.hpp"         // for gpuCommandType, gpuCommandType::KERNEL
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface, Config

#include "fmt.hpp" // for format, fmt

#include <cstdint>   // for int32_t
#include <exception> // for exception
#include <regex>     // for match_results<>::_Base_type
#include <stdexcept> // for runtime_error
#include <string.h>  // for memcpy, memset
#include <vector>    // for vector

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_HSA_COMMAND(trackingbeamform);

trackingbeamform::trackingbeamform(Config& config, const std::string& unique_name,
                                         bufferContainer& host_buffers,
                                         hsaDeviceInterface& device) :
    hsaCommand(config, unique_name, host_buffers, device, "trackingbf_float" KERNEL_EXT,
               "tracking_beamformer_nbeam.hsaco") {
    command_type = gpuCommandType::KERNEL;

    _num_elements = config.get<int32_t>(unique_name, "num_elements");
    _num_beams = config.get<int32_t>(unique_name, "num_beams");
    _samples_per_data_set = config.get<int32_t>(unique_name, "samples_per_data_set");
    _num_pol = config.get<int32_t>(unique_name, "num_pol");

    // input frame contains num_elements * time_samples (packed words)
    input_frame_len = _num_elements * _samples_per_data_set;

    // output layout is logically [time, freq_group, element=(beam*pol + pol)],
    // where the kernel tiles time across the z-dimension with TS samples per work-item.
    // Total output bytes remains samples_per_data_set * num_beams * num_pol (one uchar per time/element).
    output_frame_len = _samples_per_data_set * _num_beams * _num_pol;

    // phase_len: per-element per-beam complex floats (2 floats per element)
    // scaling floats are stored right after the phase buffer (one float per beam).
    phase_len = _num_elements * _num_beams * 2 * sizeof(float);

    if (_num_beams > MAX_NUM_BEAMS)
        throw std::runtime_error(
            fmt::format(fmt("Too many beams (_num_beams: {:d}). Max allowed is: {:d}"), _num_beams,
                        MAX_NUM_BEAMS));
}

trackingbeamform::~trackingbeamform() {}

hsa_signal_t trackingbeamform::execute(int gpu_frame_id, hsa_signal_t precede_signal) {
    // Unused parameter, suppress warning
    (void)precede_signal;

    struct __attribute__((aligned(16))) args_t {
        void* input_buffer;
        void* phase_buffer;
        void* output_buffer;
        void* scaling_buffer;
    } args;
    memset(&args, 0, sizeof(args));
    args.input_buffer = device.get_gpu_memory("input_reordered", input_frame_len);
    args.phase_buffer = device.get_gpu_memory_array("beamform_phase", gpu_frame_id,
                                                    phase_len + _num_beams * sizeof(float));
    // The scaling buffer is stored at the end of the phase array.
    args.scaling_buffer = (void*)((uint8_t*)args.phase_buffer + phase_len);
    args.output_buffer =
        device.get_gpu_memory_array("bf_tracking_output", gpu_frame_id, output_frame_len);


    // Allocate the kernel argument buffer from the correct region.
    memcpy(kernel_args[gpu_frame_id], &args, sizeof(args));

    // Kernel tiling constants: TS must match the kernel (trackingbf_float uses TS=64).
    const int TS = 64;

    // Validate that samples_per_data_set is divisible by TS since the kernel tiles time into
    // z-groups of size TS. number_of_time_groups is the kernel's z-dimension (get_global_size(2))
    if (_samples_per_data_set % TS != 0) {
        throw std::runtime_error(fmt::format(
            fmt("samples_per_data_set ({:d}) must be divisible by TS ({:d})"), _samples_per_data_set, TS));
    }
    const int num_time_groups = _samples_per_data_set / TS; // kernel z-dimension

    kernelParams params;
    // keep your original x/y config but make z explicit & validated
    params.workgroup_size_x = 64; // matches kernel's expectations for local lanes
    params.workgroup_size_y = 1;
    params.workgroup_size_z = 1;

    params.grid_size_x = 128; // tuneable: number of workgroups in x (words-per-work-item tiling)
    params.grid_size_y = _num_beams; // beam dimension -> get_global_id(1)
    params.grid_size_z = num_time_groups; // time is tiled across z; get_group_id(2) * TS + t gives absolute time
    params.num_dims = 3;

    params.private_segment_size = 0;
    params.group_segment_size = 2048;

    signals[gpu_frame_id] = enqueue_kernel(params, gpu_frame_id);

    return signals[gpu_frame_id];
}

