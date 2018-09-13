#include "applyGains.hpp"
#include "visBuffer.hpp"
#include "errors.h"
#include "configUpdater.hpp"
#include "visFileH5.hpp"
#include "visUtil.hpp"
#include "prometheusMetrics.hpp"

#include <algorithm>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <sys/stat.h>
#include <csignal>
#include <exception>

using namespace HighFive;
using namespace std::placeholders;



REGISTER_KOTEKAN_PROCESS(applyGains);



applyGains::applyGains(Config& config,
                           const string& unique_name,
                           bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container,
                   std::bind(&applyGains::main_thread, this)) {

    apply_config(0);

    // Setup the input buffer
    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    // Setup the output buffer
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    // FIFO for gains updates
    gains_fifo = updateQueue<std::vector<std::vector<cfloat>>>(num_kept_updates);

    // subscribe to gain timestamp updates
    configUpdater::instance().subscribe(this,
                std::bind(&applyGains::receive_update, this, _1));
}

void applyGains::apply_config(uint64_t fpga_seq) {

    // Number of gain versions kept. Default is 5.
    num_kept_updates = config.get_uint64_default(unique_name, "num_kept_updates", 5);
    if (num_kept_updates < 1)
        throw std::invalid_argument("applyGains: config: num_kept_updates has" \
                                    "to be equal or greater than one (is "
                                    + std::to_string(num_kept_updates) + ").");
    // Time to blend old and new gains in seconds. Default is 5 minutes.
    tcombine = config.get_float_default(unique_name, "combine_gains_time", 5*60);
    if (tcombine < 0)
        throw std::invalid_argument("applyGains: config: combine_gains_time has" \
                                    "to be positive (is "
                                    + std::to_string(tcombine) + ").");

    // Get the path to gains directory
    gains_dir = config.get_string(unique_name, "gains_dir");

}

bool applyGains::fexists(const std::string& filename) {
    struct stat buf;
    return (stat(filename.c_str(), &buf) == 0);
}

bool applyGains::receive_update(nlohmann::json &json) {
    double new_ts;
    std::string gains_path;
    std::string gtag;
    std::vector<std::vector<cfloat>> gain_read;
    // receive new gains timestamp ("start_time" might move to "start_time")
    try {
        if (!json.at("start_time").is_number())
            throw std::invalid_argument("applyGains: received bad gains " \
                                       "timestamp: " +
                                       json.at("start_time").dump());
        if (json.at("start_time") < 0)
            throw std::invalid_argument("applyGains: received negative gains " \
                                       "timestamp: " +
                                       json.at("start_time").dump());
        new_ts = json.at("start_time");
    } catch (std::exception& e) {
        WARN("Failure reading 'start_time' from update: %s", e.what());
        return false;
    }
    if (ts_frame > double_to_ts(new_ts)) {
            WARN("applyGains: Received update with a timestamp that is older " \
                 "than the current frame (The difference is %f s).",
                 ts_to_double(ts_frame) - new_ts);
            num_late_updates++;
    }

    // receive new gains tag
    try {
        if (!json.at("tag").is_string())
            throw std::invalid_argument("applyGains: received bad gains tag:" \
                                        + json.at("tag").dump());
        gtag = json.at("tag");
    } catch (std::exception& e) {
        WARN("Failure reading 'tag' from update: %s", e.what());
        return false;
    }
    // Get the gains for this timestamp
    // TODO: For now, assume the tag is the gain file name.
    gains_path = gains_dir + "/" + gtag + ".h5";
    //Check if file exists
    if (!fexists(gains_path)) {
        // Try a different extension
        gains_path = gains_dir + "/" + gtag + ".hdf5";
        if (!fexists(gains_path)) {
            WARN("Could not update gains. File not found: %s",\
                 gains_path.c_str())
            return false;
        }
    }

    // Read the gains from file
    HighFive::File gains_fl(gains_path, HighFive::File::ReadOnly);
    // Read the dataset and alocates it to the most recent entry of the gain vector
    HighFive::DataSet gains_ds = gains_fl.getDataSet("/gain");
    gains_ds.read(gain_read);
    gain_mtx.lock();
    gains_fifo.insert(double_to_ts(new_ts), std::move(gain_read));
    gain_mtx.unlock();
    INFO("Updated gains to %s.", gtag.c_str());

    return true;
}

void applyGains::main_thread() {

    unsigned int output_frame_id = 0;
    unsigned int input_frame_id = 0;
    unsigned int freq;
    double tpast;
    double frame_time;
    size_t num_late_frames = 0;

    num_late_updates = 0;

    while (!stop_thread) {


        // Wait for the input buffer to be filled with data
        if(wait_for_full_frame(in_buf,
                    unique_name.c_str(),input_frame_id) == nullptr) {
            break;
        }

        // Create view to input frame
        auto input_frame = visFrameView(in_buf, input_frame_id);

        // get the frames timestamp
        ts_frame = std::get<1>(input_frame.time);

        // frequency index of this frame
        freq = input_frame.freq_id;
        // Unix time
        frame_time = ts_to_double(std::get<1>(input_frame.time));
        // Vector for storing gains
        std::vector<cfloat> gain(input_frame.num_elements);
        std::vector<cfloat> gain_conj(input_frame.num_elements);
        // Vector for storing weight factors
        std::vector<float> weight_factor(input_frame.num_elements);


        std::pair< timespec, const std::vector<std::vector<cfloat>>* > gainpair_new;
        std::pair< timespec, const std::vector<std::vector<cfloat>>* > gainpair_old;

        gain_mtx.lock();
        gainpair_new = gains_fifo.get_update(double_to_ts(frame_time));
        if (gainpair_new.second == NULL) {
            WARN("No gains available.\nKilling kotekan");
            std::raise(SIGINT);
        }
        tpast = frame_time - ts_to_double(gainpair_new.first);

        // Determine if we need to combine gains:
        bool combine_gains = (tpast>=0) && (tpast<tcombine);
        if (combine_gains) {
            gainpair_old = gains_fifo.get_update(double_to_ts(frame_time - tcombine));
            // If we are not using the very first set of gains, do gains interpolation:
            combine_gains = combine_gains && \
                !(gainpair_new.first==gainpair_old.first);
        }

        // Combine gains if needed:
        if (combine_gains) {
            float coef_new = tpast/tcombine;
            float coef_old = 1 - coef_new;
            for (uint32_t ii=0; ii<input_frame.num_elements; ii++) {
                gain[ii] = coef_new * (*gainpair_new.second)[freq][ii] \
                         + coef_old * (*gainpair_old.second)[freq][ii];
            }
        } else {
            gain = (*gainpair_new.second)[freq];
            if (tpast < 0) {
                WARN("No gains update is as old as the currently processed " \
                     "frame. Using oldest gains available."\
                     "Time difference is: %f seconds.", tpast);
                num_late_frames++;
            }
        }
        gain_mtx.unlock();
        // Compute weight factors and conjugate gains
        for (uint32_t ii=0; ii<input_frame.num_elements; ii++) {
            gain_conj[ii] = std::conj(gain[ii]);
            weight_factor[ii] = pow(abs(gain[ii]), -2.0);
        }

        // Wait for the output buffer to be empty of data
        if(wait_for_empty_frame(out_buf, unique_name.c_str(),
                                        output_frame_id) == nullptr) {
            break;
        }
        allocate_new_metadata_object(out_buf, output_frame_id);

        // Copy frame and create view
        auto output_frame = visFrameView(out_buf, output_frame_id,
                                         input_frame.num_elements,
                                         input_frame.num_prod,
                                         input_frame.num_ev);

        // Copy over the data we won't modify
        output_frame.copy_nonconst_metadata(input_frame);
        output_frame.copy_nonvis_buffer(input_frame);

        cfloat * out_vis = output_frame.vis.data();
        cfloat * in_vis = input_frame.vis.data();
        float * out_weight = output_frame.weight.data();
        float * in_weight = input_frame.weight.data();


        // For now this doesn't try to do any type of check on the
        // ordering of products in vis and elements in gains.
        // Also assumes the ordering of freqs in gains is standard
        uint32_t idx = 0;
        for (uint32_t ii=0; ii<input_frame.num_elements; ii++) {
            for (uint32_t jj=ii; jj<input_frame.num_elements; jj++) {
                // Gains are to be multiplied to vis
                out_vis[idx] = in_vis[idx]
                                        * gain[ii]
                                        * gain_conj[jj];
                // Update the weights.
                out_weight[idx] = in_weight[idx]
                                           * weight_factor[ii]
                                           * weight_factor[jj];
                idx++;
            }
            // Update the gains.
            output_frame.gain[ii] = input_frame.gain[ii] * gain[ii];
        }

        // Report how old the gains being applied to the current data are.
        prometheusMetrics::instance().add_process_metric(
            "kotekan_applygains_update_age_seconds",
            unique_name, tpast);

        // Report number of updates received too late
        prometheusMetrics::instance().add_process_metric(
            "kotekan_applygains_late_update_count",
            unique_name, num_late_updates);

        // Report number of frames received late
        prometheusMetrics::instance().add_process_metric(
            "kotekan_applygains_late_frame_count",
            unique_name, num_late_frames);

        // Mark the buffers and move on
        mark_frame_full(out_buf, unique_name.c_str(), output_frame_id);
        mark_frame_empty(in_buf, unique_name.c_str(), input_frame_id);
        // Advance the current frame ids
        input_frame_id = (input_frame_id + 1) % in_buf->num_frames;
        output_frame_id = (output_frame_id + 1) % out_buf->num_frames;
    }
}