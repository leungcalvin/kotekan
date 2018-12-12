/*****************************************
@file
@brief Drops frames when its output buffer is full.
- Valve : public KotekanProcess
*****************************************/
#ifndef VALVE_HPP
#define VALVE_HPP

#include <string>

#include "Config.hpp"
#include "KotekanProcess.hpp"
#include "buffer.h"
#include "bufferContainer.hpp"


/**
 * @class Valve
 * @brief ``KotekanProcess`` that drops incoming frames when its output buffer
 * is full.
 *
 * This can have quite a small input buffer, since it drops frame from here, if
 * the output buffer is full.
 *
 * @par Buffers
 * @buffer in_buf The kotekan buffer from which frames are read, can be any size.
 *         @buffer_format GPU packed upper triangle
 *         @buffer_metadata chimeMetadata
 * @buffer out_buf The kotekan buffer which will be fed the subset of visibilities.
 *         @buffer_format GPU packed upper triangle
 *         @buffer_metadata chimeMetadata
 *
 * @par Metrics
 * @metric kotekan_valve_dropped_frames_total
 *        The number of frames dropped.
 *
 *
 * @author  Rick Nitsche
 *
 */
class Valve : public KotekanProcess {

public:
    /// Constructor.
    Valve(Config &config,
          const std::string& unique_name,
          bufferContainer &buffer_container);

    /// Primary loop.
    void main_thread() override;

private:
    /// Copy a frame from the input buffer to the output buffer.
    static void copy_frame(Buffer* buf_src, int frame_id_src,
                           Buffer* buf_dest, int frame_id_dest);

    /// Input buffer
    Buffer * _buf_in;

    /// Output buffer to receive baseline subset visibilities
    Buffer* _buf_out;


    /// Number of dropped frames.
    uint64_t _dropped_total;
};


#endif // VALVE_HPP