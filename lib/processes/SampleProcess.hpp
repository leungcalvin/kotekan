#ifndef SAMPLEPROCESS_H
#define SAMPLEPROCESS_H

#include "KotekanProcess.hpp"

class SampleProcess : public kotekan::KotekanProcess {
public:
    SampleProcess(kotekan::Config& config, const string& unique_name,
                  kotekan::bufferContainer& buffer_container);
    virtual ~SampleProcess();
    void main_thread() override;

private:
};

#endif /* SAMPLEPROCESS_H */
