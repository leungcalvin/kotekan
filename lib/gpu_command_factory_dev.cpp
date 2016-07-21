#include "gpu_command_factory.h"
#include <stdio.h>
#include <stdlib.h>
#include "errors.h"
#include <errno.h>

gpu_command_factory::gpu_command_factory()
{
    currentCommandCnt = 0;
}

cl_uint gpu_command_factory::getNumCommands() const
{
    return numCommands;
}

void gpu_command_factory::initializeCommands(class device_interface & param_Device, Config * param_Config)
{
    //X-Engine EXECUTION SEQUENCE IS OFFSET, PRESEED, CORRELATE.
    //numCommands = param_Config->gpu.num_kernels;
    numCommands = param_Config->gpu.num_kernels + 4;//THRE ADDITIONAL COMMANDS - Input, Input Beamform, & Output, Output Beamform
    
    listCommands =  new gpu_command * [numCommands];

    char** gpuKernels = param_Config->gpu.kernels;

    int file_idx = 0;
    
    for (int i = 0; i < numCommands; i++){    
        if (i==0){
            listCommands[i] = new input_data_stage("input_data_stage");
        }
        else if (i==1){
            if (param_Config->gpu.use_beamforming == 1){
                listCommands[i] = new beamform_phase_data("beamform_phase_data");
            }
            else{
                listCommands[i] = new dummy_placeholder_kernel("dummy");
            }
        }
        //IN THE ORIGINAL VERSION, THE SQUENCE IS X-ENGINE --> READ, THEN BEAMFORM --> READ. HERE, X-ENGINE AND BEAMFORM HAPPEND TOGETHER AS DO THE READS. WILL THIS BE SLOWER?
        else if (i == (numCommands - 2)){ 
            if (param_Config->gpu.use_beamforming == 1){
                listCommands[i] = new output_beamform_result("output_beamform_result");
            }
            else{
                listCommands[i] = new dummy_placeholder_kernel("dummy");
            }
        }
        else if (i == (numCommands - 1)){
            listCommands[i] = new output_data_result("output_data_result");
        }
        else {
            if (std::gpuKernels[file_idx].find("offset_accumulator") != -1){
                listCommands[i] = new offset_kernel(gpuKernels[file_idx], "offset_accumulator");
            }
            else if (std::gpuKernels[file_idx].find("preseed_multifreq") != -1){
                listCommands[i] = new preseed_kernel(gpuKernels[file_idx], "preseed_multifreq");
            }
            else if (std::gpuKernels[file_idx].find("pairwise_correlator") != -1){
                listCommands[i] = new correlator_kernel(gpuKernels[file_idx], "pairwise_correlator");
            }
            else if (std::gpuKernels[file_idx].find("beamform_tree_scale") != -1){
                if (param_Config->gpu.use_beamforming == 1){
                    listCommands[i] = new beamform_kernel(gpuKernels[file_idx], "beamform_tree_scale");
                }
                else{
                    listCommands[i] = new dummy_placeholder_kernel("dummy");
                }
            }
            file_idx++;
        }
               
        listCommands[i]->build(param_Config, param_Device);
    }

    currentCommandCnt = 0;
}

gpu_command* gpu_command_factory::getNextCommand(class device_interface & param_Device, int param_BufferID)
{
  //LEAVE THIS AS IS FOR NOW, BUT LATER WILL WANT TO DYNAMICALLY REQUEST FOR MEMORY BASED ON KERNEL STATE AND SET PRE AND POST CL_EVENT BASED ON EVENTS RETURNED BY INDIVIDUAL KERNAL OBJECTS.
  //KERNELS WILL TRACK SETTING THEIR OWN PRE AND POST EVENTS, BUT WILL RETURN THOSE EVENTS TO BE PASSED TO THE NEXT KERNEL IN THE SEQUENCE
    
//  TO ADDRESS THE ISSUE OF COMMAND_OBJECTS NEEDING TO PASS BUFFERS TO EACH OTHER (IE BEAMFORM PHASE AND BEAMFORM KERNEL)
//  IT MAY BE A GOOD IDEA TO INTRODUCE AN OBJECT THAT LIVES IN COMMAND_FACTOR CALLED RESOURCE_ALLOCATION. IT WILL SERVE
//  AS AN INTERFACE BETWEEN DEVICE_INTERFACE - RESPONSIBLE FOR ALLOCATING MEMORY, AND COMMAND_OBJECTS - RESPONSIBLE FOR MANAGING
//  MEMORY OBJECTS. MEMORY WILL BE ALLOCATED BY DEVICE_INTERFACE FOR A BUFFER THAT IS STORED AND MAINTAINED BY COMMAND_OBJECT,
//  BUT THE RESOURCE_ALLOCATION OBJECT WILL RECEIVE A REFERENCE TO THESE MEMORY OBJECTS AND WILL BE RESPONSIBLE FOR DISTRIBUTING
//  THOSE MEMORY BUFFERS TO THE DIFFERENT COMMAND_OBJECTS TAHT NEED THEM.
    gpu_command* currentCommand;

    currentCommand = listCommands[currentCommandCnt];
    
    switch (currentCommand->get_name)
    {
        case "input_data_stage"://input_data_stage prep
            break;
        case "beamform_phase_data":
            break;
        case "pairwise_correlator"://THIRD KERNEL BY EVENTS SEQUENCE "corr"
            currentCommand->setKernelArg(0, param_Device.getInputBuffer(param_BufferID));
            currentCommand->setKernelArg(1, param_Device.getOutputBuffer(param_BufferID));
            break;
        case "offset_accumulator"://FIRST KERNEL BY EVENTS SEQUENCE "offsetAccumulateElements"
            currentCommand->setKernelArg(0, param_Device.getInputBuffer(param_BufferID));
            currentCommand->setKernelArg(1, param_Device.getAccumulateBuffer(param_BufferID));
            break;
        case "preseed_multifreq"://SECOND KERNEL BY EVENTS SEQUENCE "preseed"
            currentCommand->setKernelArg(0, param_Device.getAccumulateBuffer(param_BufferID));
            currentCommand->setKernelArg(1, param_Device.getOutputBuffer(param_BufferID));
            break;
        case "beamform_tree_scale":
            if (cl_data->config->gpu.use_beamforming) {


        CHECK_CL_ERROR( clSetKernelArg(cl_data->beamform_kernel,
                                       0,
                                       sizeof(void *),
                                       (void*) &device_kernel_input_data) );

        CHECK_CL_ERROR( clSetKernelArg(cl_data->beamform_kernel,
                                       1,
                                       sizeof(void *),
                                       (void*) &cl_data->device_beamform_output_buffer[buffer_id]) );

        CHECK_CL_ERROR( clSetKernelArg(cl_data->beamform_kernel,
                                       2,
                                       sizeof(cl_mem),
                                       (void*) &cl_data->device_freq_map[buffer_id % cl_data->num_links]) );

        CHECK_CL_ERROR( clSetKernelArg(cl_data->beamform_kernel,
                               3,
                               sizeof(cl_mem),
                               (void*) &cl_data->device_phases) );
            break;
        case "output_data_result":
            break;
        case"output_beamform_result":
            break;
      }

      currentCommandCnt++;
      if (currentCommandCnt >= numCommands)
	currentCommandCnt = 0;

  return currentCommand;

}
void gpu_command_factory::deallocateResources()
{
    for (int i = 0; i < numCommands; i++){
     listCommands[i]->freeMe();
    }
    DEBUG("CommandsFreed\n");

    delete[] listCommands;
    DEBUG("ListCommandsDeleted\n");
}
