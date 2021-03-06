cmake_minimum_required( VERSION 2.8 )
project( kotekan_opencl )

include_directories(${OpenCL_INCLUDE_DIR})

include_directories (${KOTEKAN_SOURCE_DIR}/include)

include_directories (${KOTEKAN_SOURCE_DIR}/lib)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/gpu)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/opencl)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/opencl/kernels)
include_directories (${KOTEKAN_SOURCE_DIR}/kotekan/opencl/kernels)

set ( KOTEKAN_OPENCL_SOURCES
      clCommand.cpp
      clDeviceInterface.cpp
      clEventContainer.cpp
      clProcess.cpp

# Copy-in & general-purpose:
      clInputData.cpp

# CHIME N2 Kernels & copy-out:
      clOutputDataZero.cpp
      clPresumZero.cpp
      clPresumKernel.cpp
      clPreseedKernel.cpp
      clCorrelatorKernel.cpp
      clOutputData.cpp
      clKVCorr.cpp

# CHIME/Pulsar Kernels & copy-out:
      clBeamformPhaseData.cpp
      clBeamformKernel.cpp
      clOutputBeamformResult.cpp

#RFI Kernels
      #clRfiTimeSum.cpp
      #clRfiInputSum.cpp
      #clRfiOutput.cpp
    )

add_library ( kotekan_opencl ${KOTEKAN_OPENCL_SOURCES} )

add_custom_target(kernel_copy
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${KOTEKAN_SOURCE_DIR}/lib/opencl/kernels"
        "kernels/")

add_dependencies( kotekan_opencl kernel_copy kotekan_gpu )
