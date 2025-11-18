#define TS 64
#define RE x
#define IM y

__kernel void trackingbf_float(__global uint*  data,
                               __global float2* phase,
                               __global uchar* output,
                               __global float* scaling) {

    float2 sum;
    const uint F = get_global_size(2);    // number of freq groups
    const uint B = get_global_size(1);    // number of beams along global y
    const uint P = get_global_size(0);    // pol-groups (expect 2)
    const uint nsamp = F * TS;            // total time slots per "frame" (matching original)

    // ph is small local scratch of preloaded phases:
    float2 ph[4][4]; // [4-in-a-word][4-words-per-work-item]

    const uint pol = get_group_id(0);     // 0 or 1
    const uint beam_id = get_global_id(1); // global beam index
    const uint freq = get_group_id(2);    // frequency index for this work-group

    // load phase words for this beam/pol into local registers (unchanged semantics)
    for (int tt = 0; tt < 4; ++tt) {
        uint element = get_local_id(0) * 4 + tt;
        uint base = ( (beam_id * 2 + pol) * 1024 ) + element * 4;
        ph[0][tt] = phase[ base + 0 ];
        ph[1][tt] = phase[ base + 1 ];
        ph[2][tt] = phase[ base + 2 ];
        ph[3][tt] = phase[ base + 3 ];
    }

    // loop over time samples owned by this work item
    for (int t = 0; t < TS; ++t) {
        sum.RE = 0.0f;
        sum.IM = 0.0f;

        // accumulate across 4 sub-elements in word
        for (int tt = 0; tt < 4; ++tt) {
            uint element = get_local_id(0) * 4 + tt;

            // data layout: data[(t*F + freq)*512 + pol*256 + element]
            uint data_temp = data[(t * F + freq) * 512 + pol * 256 + element];

            // multiply-add using the loaded phase words (same math you had)
            sum.RE +=
                  ph[0][tt].RE * ((float)((data_temp & 0x000000f0) >>  4u) - 8.0f)
                + ph[0][tt].IM * ((float)((data_temp & 0x0000000f) >>  0u) - 8.0f)
                + ph[1][tt].RE * ((float)((data_temp & 0x0000f000) >> 12u) - 8.0f)
                + ph[1][tt].IM * ((float)((data_temp & 0x00000f00) >>  8u) - 8.0f)
                + ph[2][tt].RE * ((float)((data_temp & 0x00f00000) >> 20u) - 8.0f)
                + ph[2][tt].IM * ((float)((data_temp & 0x000f0000) >> 16u) - 8.0f)
                + ph[3][tt].RE * ((float)((data_temp & 0xf0000000) >> 28u) - 8.0f)
                + ph[3][tt].IM * ((float)((data_temp & 0x0f000000) >> 24u) - 8.0f);

            sum.IM +=
                - ph[0][tt].IM * ((float)((data_temp & 0x000000f0) >>  4u) - 8.0f)
                + ph[0][tt].RE * ((float)((data_temp & 0x0000000f) >>  0u) - 8.0f)
                - ph[1][tt].IM * ((float)((data_temp & 0x0000f000) >> 12u) - 8.0f)
                + ph[1][tt].RE * ((float)((data_temp & 0x00000f00) >>  8u) - 8.0f)
                - ph[2][tt].IM * ((float)((data_temp & 0x00f00000) >> 20u) - 8.0f)
                + ph[2][tt].RE * ((float)((data_temp & 0x000f0000) >> 16u) - 8.0f)
                - ph[3][tt].IM * ((float)((data_temp & 0xf0000000) >> 28u) - 8.0f)
                + ph[3][tt].RE * ((float)((data_temp & 0x0f000000) >> 24u) - 8.0f);
        }

        // intra-wg reductions (kept your AMD-specific intrinsics as-is)
        sum.RE += as_float(__builtin_amdgcn_ds_bpermute((16 + get_local_id(0)) * 4, as_uint(sum.RE)))
                + as_float(__builtin_amdgcn_ds_bpermute((32 + get_local_id(0)) * 4, as_uint(sum.RE)))
                + as_float(__builtin_amdgcn_ds_bpermute((48 + get_local_id(0)) * 4, as_uint(sum.RE)));
        sum.IM += as_float(__builtin_amdgcn_ds_bpermute((16 + get_local_id(0)) * 4, as_uint(sum.IM)))
                + as_float(__builtin_amdgcn_ds_bpermute((32 + get_local_id(0)) * 4, as_uint(sum.IM)))
                + as_float(__builtin_amdgcn_ds_bpermute((48 + get_local_id(0)) * 4, as_uint(sum.IM)));

        // add partial lanes (your mov_dpp magic)
        sum.RE += as_float(__builtin_amdgcn_mov_dpp(as_uint(sum.RE), 0x104, 0xf, 0xf, 0))
                + as_float(__builtin_amdgcn_mov_dpp(as_uint(sum.RE), 0x108, 0xf, 0xf, 0))
                + as_float(__builtin_amdgcn_mov_dpp(as_uint(sum.RE), 0x10c, 0xf, 0xf, 0));
        sum.IM += as_float(__builtin_amdgcn_mov_dpp(as_uint(sum.IM), 0x104, 0xf, 0xf, 0))
                + as_float(__builtin_amdgcn_mov_dpp(as_uint(sum.IM), 0x108, 0xf, 0xf, 0))
                + as_float(__builtin_amdgcn_mov_dpp(as_uint(sum.IM), 0x10c, 0xf, 0xf, 0));

        sum.RE += as_float(__builtin_amdgcn_mov_dpp(as_uint(sum.RE), 0x101, 0xf, 0xf, 0))
                + as_float(__builtin_amdgcn_mov_dpp(as_uint(sum.RE), 0x102, 0xf, 0xf, 0))
                + as_float(__builtin_amdgcn_mov_dpp(as_uint(sum.RE), 0x103, 0xf, 0xf, 0));
        sum.IM += as_float(__builtin_amdgcn_mov_dpp(as_uint(sum.IM), 0x101, 0xf, 0xf, 0))
                + as_float(__builtin_amdgcn_mov_dpp(as_uint(sum.IM), 0x102, 0xf, 0xf, 0))
                + as_float(__builtin_amdgcn_mov_dpp(as_uint(sum.IM), 0x103, 0xf, 0xf, 0));

        // only lane 0 writes the packed byte (same packing as your original)
        if (get_local_id(0) == 0) {
            // apply scaling (scaling indexed by beam)
            float s = scaling[beam_id];
            sum.RE = sum.RE / s + 8.0f;
            sum.IM = sum.IM / s + 8.0f;

            // clamp to [0,15]
            sum.RE = (sum.RE > 15.0f) ? 15.0f : ((sum.RE < 0.0f) ? 0.0f : sum.RE);
            sum.IM = (sum.IM > 15.0f) ? 15.0f : ((sum.IM < 0.0f) ? 0.0f : sum.IM);

            // Compute explicit linear index in layout [time, freq, element=(beam*2+pol)]
            // t     : 0..TS-1
            // freq  : 0..F-1  (get_group_id(2))
            // element_count = B * 2  (beams * pols)
            const uint t_global = t; // local time index; if you need absolute frame time you must offset externally
            const uint elements_per_tf = B * 2u;
            const uint linear_tf = t_global * F + freq; // (t * F + freq)
            const uint out_idx = linear_tf * elements_per_tf + beam_id * 2u + pol;

            // write packed uchar (high nibble = RE, low nibble = IM)
            output[out_idx] = (uchar)((((int)sum.RE & 0x0F) << 4) | ((int)sum.IM & 0x0F));
        }
    } // end t loop
}

