/*
 * Copyright (c) 2015 <copyright holder> <email>
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 */

#ifndef KERNELPRESEED_H
#define KERNELPRESEED_H

#include "gpu_command.h"

class kernelPreseed: public gpu_command
{
public:
    kernelPreseed();
    kernelPreseed(const char param_gpuKernel);
    ~kernelPreseed();
    virtual void build(const Config& param_Config, const device_interface& param_Device);
    virtual cl_event execute(int param_bufferID, const device_interface &param_Device);
protected:
  void defineOutputDataMap(const Config & param_Config, int param_num_blocks, const device_interface & param_Device, cl_mem & id_x_map, cl_mem & id_y_map);
};

#endif // KERNELPRESEED_H
