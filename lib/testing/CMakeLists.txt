cmake_minimum_required( VERSION 2.8 )
project( libkotekan_testing )

set ( KOTEKAN_TESTING_SOURCES
      chimeMetadataDump.cpp
      constDataCheck.cpp
      fakeGpuBuffer.cpp
      fakeVis.cpp
      gpuBeamformSimulate.cpp
      gpuBeamformPulsarSimulate.cpp
      gpuSimulate.cpp
      hexDump.cpp
      networkOutputSim.cpp
      simVdifData.cpp
      testDataCheck.cpp
      testDataGen.cpp
      testDataGenQuad.cpp
      testDataGenFloat.cpp
      test_data_generation.c
      pulsarSimProcess.cpp
      visNoise.cpp
      visDrop.cpp
    )

add_library ( kotekan_testing ${KOTEKAN_TESTING_SOURCES} )
