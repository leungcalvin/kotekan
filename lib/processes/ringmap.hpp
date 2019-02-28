
#ifndef RINGMAP_HPP
#define RINGMAP_HPP

#include "gsl-lite.hpp"
#include "KotekanProcess.hpp"
#include "visUtil.hpp"
#include "restServer.hpp"
#include "datasetManager.hpp"
#include "fpga_header_functions.h"

#define XX=0
#define XY=1
#define YX=2
#define YY=3

class mapMaker : public KotekanProcess {

public:

    // Default constructor
    mapMaker(Config &config,
             const string& unique_name,
             bufferContainer &buffer_container);

    // Main loop for the process
    void main_thread() override;

    /// REST endpoint to request a map
    nlohmann::json rest_callback(connectionInstance& conn, nlohmann::json &json);

    /// Abbreviation for RingMap type
    typedef std::vector<std::vector<cfloat>> RingMap;

private:

    bool setup(size_t frame_id);

    void change_dataset_state();

    void gen_matrices();

    int64_t resolve_time(time_ctype t);

    inline float wl(uint32_t fid) {
        return 299.792458 / freq_from_bin(fid);
    };

    // Matrix from visibilities to map for every freq (same for each pol)
    std::map<uint32_t,std::vector<cfloat>> vis2map;
    // Store the maps and weight maps for every frequency
    std::map<uint32_t,RingMap> map;
    std::map<uint32_t,RingMap> wgt_map;
    std::vector<float> ns_baselines;
    std::vector<stack_ctype> stacks;
    std::vector<prod_ctype> prods;
    std::vector<input_ctype> inputs;

    // Keep track of map dimensions
    std::vector<float> sinza;
    std::vector<uint32_t> freq_id;
    std::vector<freq_ctype> freqs;
    std::vector<time_ctype> times;
    std::map<uint64_t, size_t> times_map;
    modulo<size_t> latest;
    uint64_t max_fpga, min_fpga;

    uint32_t num_time;
    uint32_t num_pix;
    uint8_t num_pol;
    uint32_t num_stack;
    uint32_t num_bl;

    dset_id_t ds_id;
    std::vector<uint32_t> excl_input;
    std::vector<std::pair<uint32_t, freq_ctype>> freq;
    std::vector<prod_ctype> prod;

    // Map buffer file
    void * map_file;

    // Buffer to read from
    Buffer* in_buf;
    dset_id_t ds_id;
};

class redundantStack : public KotekanProcess {

public:

    redundantStack(Config &config,
                   const string& unique_name,
                   bufferContainer &buffer_container);

    void main_thread();

private:

    void change_dataset_state(dset_id_t ds_id);

    // dataset states and IDs
    dset_id_t output_dset_id;
    dset_id_t input_dset_id;
    const prodState* prod_state_ptr;
    const stackState* old_stack_state_ptr;
    const stackState* new_stack_state_ptr;

    // Buffers
    Buffer* in_buf;
    Buffer* out_buf;
};

#endif