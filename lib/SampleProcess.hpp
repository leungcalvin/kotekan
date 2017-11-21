#ifndef SAMPLEPROCESS_H
#define SAMPLEPROCESS_H

#include "KotekanProcess.hpp"

class SampleProcess : public KotekanProcess {
public:
    SampleProcess(Config &config, const string& unique_name, bufferContainer &buffer_container);
    virtual ~SampleProcess();
    void main_thread();
    void apply_config(uint64_t fpga_seq) override;
private:

};

#endif /* SAMPLEPROCESS_H */
