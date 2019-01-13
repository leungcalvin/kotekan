#ifndef SAMPLEPROCESS_H
#define SAMPLEPROCESS_H

#include "stage.hpp"

class SampleProcess : public kotekan::Stage {
public:
    SampleProcess(kotekan::Config& config, const string& unique_name,
                  kotekan::bufferContainer& buffer_container);
    virtual ~SampleProcess();
    void main_thread() override;

private:
};

#endif /* SAMPLEPROCESS_H */
