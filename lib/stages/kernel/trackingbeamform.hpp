#include "Config.hpp"             // for Config
#include "bufferContainer.hpp"    // for bufferContainer
#include "hsa/hsa.h"              // for hsa_signal_t
#include "hsaCommand.hpp"         // for hsaCommand
#include "hsaDeviceInterface.hpp" // for hsaDeviceInterface

#include <stdint.h> // for int32_t
#include <string>   // for string

class trackingbeamform : public hsaCommand {
public:
    /// Constructor initializes internal variables from config
    trackingbeamform(kotekan::Config& config, const std::string& unique_name,
                        kotekan::bufferContainer& host_buffers, hsaDeviceInterface& device);

    /// Destructor
    virtual ~trackingbeamform();

    /// Allocate kernel argument buffer, set kernel dimensions, enqueue kernel
    hsa_signal_t execute(int gpu_frame_id, hsa_signal_t precede_signal) override;

private:
    /// Input byte length: num_elements × samples_per_data_set (packed 4+4-bit)
    int32_t input_frame_len;

    /// Output byte length: samples_per_data_set × num_beams × num_pol
    /// (each output is a packed 4+4-bit uchar)
    int32_t output_frame_len;

    /// Phase array length (float32): num_beams × num_elements × 2 (real,imag)
    /// Does not include scaling floats, which are stored immediately after.
    int32_t phase_len;

    /// Number of elements
    int32_t _num_elements;
    /// Number of beams
    int32_t _num_beams;
    /// Number of polarizations
    int32_t _num_pol;
    /// Number of time samples per data frame (must be divisible by 64)
    int32_t _samples_per_data_set;
};

#endif
