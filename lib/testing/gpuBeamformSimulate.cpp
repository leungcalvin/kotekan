#include "gpuBeamformSimulate.hpp"
#include "errors.h"
#include <math.h>

#define SWAP(a,b) tempr=(a);(a)=(b);(b)=tempr
#define HI_NIBBLE(b)                    (((b) >> 4) & 0x0F)
#define LO_NIBBLE(b)                    ((b) & 0x0F)

#define PI 3.14159265
#define feed_sep 0.3048
#define light 3.e8
#define Freq_ref 492.125984252
#define freq1 450.

gpuBeamformSimulate::gpuBeamformSimulate(Config& config,
        const string& unique_name,
        bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container, std::bind(&gpuBeamformSimulate::main_thread, this)) {

    apply_config(0);

    input_buf = get_buffer("network_in_buf");
    register_consumer(input_buf, unique_name.c_str());
    output_buf = get_buffer("beam_out_buf");
    register_producer(output_buf, unique_name.c_str());

    input_len = _samples_per_data_set * _num_elements * 2;
    input_len_padded = input_len * 2;
    transposed_len = (_samples_per_data_set+32) * _num_elements * 2;
    output_len = _num_elements*(_samples_per_data_set/_downsample_time/_downsample_freq);


    input_unpacked = (double *)malloc(input_len * sizeof(double));
    input_unpacked_padded = (double *)malloc(input_len_padded * sizeof(double));
    clamping_output = (double *)malloc(input_len * sizeof(double));
    cpu_beamform_output = (double *)malloc(input_len * sizeof(double));
    transposed_output = (double *)malloc(transposed_len * sizeof(double));
    tmp128 = (double *)malloc(_factor_upchan*2*sizeof(double));
    cpu_final_output = (double *)malloc(output_len*sizeof(double));

    coff = (float *) malloc(16*2*sizeof(float));
    assert(coff != nullptr);
    for (int angle_iter=0; angle_iter < 4; angle_iter++){
        // NEED TO FIND OUT THE ANGLES
      double anglefrac = sin(0.1*angle_iter*PI/180.);
      for (int cylinder=0; cylinder < 4; cylinder++){
	coff[angle_iter*4*2 + cylinder*2]     = cos( 2*PI*anglefrac*cylinder*22*freq1*1.e6/light );
	coff[angle_iter*4*2 + cylinder*2 + 1] = sin( 2*PI*anglefrac*cylinder*22*freq1*1.e6/light);
      }
    }
}

gpuBeamformSimulate::~gpuBeamformSimulate() {

    free(input_unpacked_padded);
    free(input_unpacked);
    free(clamping_output);
    free(cpu_beamform_output);
    free(coff);
    free(transposed_output);
    free(tmp128);
    free(cpu_final_output);
}

void gpuBeamformSimulate::apply_config(uint64_t fpga_seq) {
    _num_elements = config.get_int(unique_name, "num_elements");
    _samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");
    _factor_upchan = config.get_int(unique_name, "factor_upchan");
    _downsample_time = config.get_int(unique_name, "downsample_time");
    _downsample_freq = config.get_int(unique_name, "downsample_freq");
}

void gpuBeamformSimulate::cpu_beamform_ns(double *data, unsigned long transform_length,  int stop_level)
{
    unsigned long n,m,j,i;
    double wr,wi,theta;
    double tempr,tempi;
    n=transform_length << 1;
    j=1;

    for (i=1;i<n;i+=2) { /* This is the bit-reversal section of the routine. */
        if (j > i) {
            SWAP(data[j-1],data[i-1]); /* Exchange the two complex numbers. */
            SWAP(data[j],data[i]);
        }
        m=transform_length;
        while (m >= 2 && j > m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
    long int step_stop;
    if (stop_level < -1) //neg values mean do the whole sequence; the last stage has pairs half the transform length apart
        step_stop = transform_length/2;
    else
        step_stop =pow(2,stop_level);

    for (long int step_size = 1; step_size <= step_stop ; step_size +=step_size) {
        theta = -3.141592654/(step_size);
        for (int index = 0; index < transform_length; index += step_size*2){
            for (int minor_index = 0; minor_index < step_size; minor_index++){
                wr = cos(minor_index*theta);
                wi = sin(minor_index*theta);
                int first_index = (index+minor_index)*2;
                int second_index = first_index + step_size*2;
                tempr = wr*data[second_index]-wi*data[second_index+1];
                tempi = wi*data[second_index]+wr*data[second_index+1];
                data[second_index    ]  = data[first_index  ]-tempr;
                data[second_index + 1]  = data[first_index+1]-tempi;
                data[first_index     ] += tempr;
                data[first_index  + 1] += tempi;
            }
        }
    }

}

void gpuBeamformSimulate::cpu_beamform_ew(double *input, double *output, float *Coeff, int nbeamsNS, int nbeamsEW, int npol, int nsamp_in)
{

    int elm_now, out_add;
    for (int t=0;t<nsamp_in;t++){
        for (int p=0;p<npol;p++){
            for (int bEW=0; bEW < nbeamsEW; bEW++){
                for (int bNS=0;bNS < nbeamsNS; bNS++){
                    out_add = (t*2048 + p*1024 + bEW*256 + bNS)*2;
                    for (int elm=0; elm<4; elm++){
                        elm_now = (t*2048 + p*1024 + elm*256 + bNS)*2;

                        //REAL
                        output[out_add]+= input[elm_now]*Coeff[2*(bEW*4+elm)]
                            +input[elm_now + 1]*Coeff[2*(bEW*4+elm)+1];
                        //IMAG
                        output[out_add+1]+=input[elm_now]*Coeff[2*(bEW*4+elm)+1]
                            - input[elm_now+1]*Coeff[2*(bEW*4+elm)];
                    }
                    output[out_add] = output[out_add]/4.;
                    output[out_add+1] = output[out_add+1]/4.;
                }
            }
        }
    }

}

void gpuBeamformSimulate::clamping(double *input, double *output, float freq, int nbeamsNS, int nbeamsEW, int nsamp_in, int npol)
{
    float t, delta_t, Beam_Ref;
    int cl_index;
    float D2R = PI/180.;
    int pad = 2 ;
    int tile=1;
    int nbeams = nbeamsEW*nbeamsNS;
    for (int b=0;b<nbeamsNS; b++){
        Beam_Ref = asin(light*(b-nbeamsNS/2.) / (Freq_ref*1.e6) / (nbeamsNS) /feed_sep) * 180./ PI;
        t = nbeamsNS*pad*(Freq_ref*1.e6)*(feed_sep/light*sin(Beam_Ref*D2R)) + 0.5;
        delta_t = nbeamsNS*pad*(freq*1e6-Freq_ref*1e6) * (feed_sep/light*sin(Beam_Ref*D2R));

        cl_index = (int) floor(t + delta_t) + nbeamsNS*tile*pad/2.;

	if (cl_index < 0)
          cl_index = 256*pad + cl_index;
        else if (cl_index > 256*pad)
          cl_index = cl_index - 256*pad;
        cl_index = cl_index - 256;
        if (cl_index < 0)
          cl_index = 256*pad + cl_index;

        for (int i=0;i<nsamp_in;i++){
	  for (int p=0;p<npol;p++){
	    for (int b2 = 0; b2< nbeamsEW; b2++){
	      output[2*(i*npol*nbeamsNS*nbeamsEW + p*nbeams + b2*nbeamsNS + b) ] = input[2*(i*2048*2 + p*1024*2+ b2*512 + cl_index)];
	      output[2*(i*npol*nbeamsNS*nbeamsEW + p*nbeams + b2*nbeamsNS + b) +1]=input[2*(i*2048*2 + p*1024*2 + b2*512 + cl_index) + 1];

                }
            }
        }
    }
}

void gpuBeamformSimulate::transpose(double *input, double *output, int n_beams, int n_samp)\
{
  for (int j=0;j<n_samp;j++){
    for (int i=0;i<n_beams;i++){
      output[(i*(n_samp+32)+j)*2] = input[(j*n_beams+i)*2];
      output[(i*(n_samp+32)+j)*2+1] = input[(j*n_beams+i)*2+1];
      for (int k=0;k<32;k++){
        output[(i*(n_samp+32)+(n_samp+k))*2] = 0.0;
        output[(i*(n_samp+32)+(n_samp+k))*2+1] =0.0;
      }
    }
  }
}

void gpuBeamformSimulate::upchannelize(double *data, int nn){
  unsigned long n,mmax,m,j,istep,i;
  double wtemp,wr,wpr,wpi,wi,theta;
  double tempr,tempi;
  n=nn << 1;
  j=1;
  for (i=1;i<n;i+=2) { /* bit-reversal section of the routine. */
    if (j > i) {
      SWAP(data[j-1],data[i-1]); /* Exchange two complex numbers. */
      SWAP(data[j],data[i]);
    }
    m=nn;
    while (m >= 2 && j > m) {
      j -= m;
      m >>= 1;
    }
    j += m;
  }
  mmax=2;
  while (n > mmax) { /* Outer loop executed log2 nn times. */
    istep=mmax << 1;
    theta=(6.28318530717959/mmax); /* Initialize the trigonometric recurr\
                                      ence. */
    wtemp=sin(0.5*theta);
    wpr = -2.0*wtemp*wtemp;
    wpi=sin(theta);
    wr=1.0;
    wi=0.0;
    for (m=1;m<mmax;m+=2) { /* two nested inner loops. */
      for (i=m;i<=n;i+=istep) {
        j=i+mmax; /* This is the Danielson-Lanczos formula. */
        tempr=wr*data[j-1]-wi*data[j];
        tempi=wr*data[j]+wi*data[j-1];
        data[j-1]=data[i-1]-tempr;
        data[j]=data[i]-tempi;
        data[i-1] += tempr;
        data[i] += tempi;
      }
      wr=(wtemp=wr)*wpr-wi*wpi+wr; /* Trigonometric recurrence. */
      wi=wi*wpr+wtemp*wpi+wi;
    }
    mmax=istep;
  }

}

void gpuBeamformSimulate::main_thread() {
    int input_buf_id = 0;
    int output_buf_id = 0;

    int npol = 2;
    int nbeamsEW = 4;
    int nbeamsNS = 256;
    int nbeams = nbeamsEW*nbeamsNS;

    while(!stop_thread) {

        unsigned char * input = (unsigned char *)wait_for_full_frame(input_buf, unique_name.c_str(), input_buf_id);
        if (input == NULL) break;
        float * output = (float *)wait_for_empty_frame(output_buf, unique_name.c_str(), output_buf_id);
        if (output == NULL) break;

        for (int i=0;i<input_len;i++){
          cpu_beamform_output[i] = 0.0; //Need this
          clamping_output[i] = 0.0;  //Maybe don't need this
        }
        for (int i=0;i<transposed_len;i++){
          transposed_output[i] = 0.0; //Maybe don't need this
        }
        for (int i=0;i<output_len;i++){
          cpu_final_output[i] = 0.0;
        }

        // TODO adjust to allow for more than one frequency.
        // TODO remove all the 32's in here with some kind of constant/define
        INFO("Simulating GPU beamform processing for %s[%d] putting result in %s[%d]",
                input_buf->buffer_name, input_buf_id,
                output_buf->buffer_name, output_buf_id);

        // Unpack and pad the input data
        int dest_idx = 0;
        for (int i = 0; i < input_buf->frame_size; ++i) {
            input_unpacked[dest_idx++] = HI_NIBBLE(input[i])-8;
            input_unpacked[dest_idx++] = LO_NIBBLE(input[i])-8;
        }

        // Pad to 512
        // TODO this can be simplified a fair bit.
        int index = 0;
        for (int j = 0; j < _samples_per_data_set*2; j++){
            for (int b = 0; b < nbeamsEW; b++){
                for (int i = 0; i < 512; i++){
                    if (i < 256){
                        input_unpacked_padded[index++] = input_unpacked[2*(j*nbeams + b*nbeamsNS + i)];
                        input_unpacked_padded[index++] = input_unpacked[2*(j*nbeams + b*nbeamsNS + i) + 1];
                    } else{
                        input_unpacked_padded[index++] = 0;
                        input_unpacked_padded[index++] = 0;
                    }
                }
            }
        }

        // Beamform north south.
        for (int i = 0; i < _samples_per_data_set*npol*nbeamsEW; i++){
            cpu_beamform_ns(&input_unpacked_padded[i*512*2], 512, 8);
        }

        // Clamp the data
        clamping(input_unpacked_padded, clamping_output, freq1, nbeamsNS, nbeamsEW, _samples_per_data_set, npol);

        //EW brute force beamform
        cpu_beamform_ew(clamping_output, cpu_beamform_output, coff, nbeamsNS, nbeamsEW, npol, _samples_per_data_set);

        //transpose
        transpose(cpu_beamform_output, transposed_output, _num_elements, _samples_per_data_set);

        //Upchannelize; re-use cpu_beamform_output
        for (int b=0; b< _num_elements; b++){
          for (int n=0;n <  _samples_per_data_set/_factor_upchan; n++){
            int index=0;
            for (int i=0;i< _factor_upchan;i++){
              tmp128[index++] = transposed_output[(b*(_samples_per_data_set+32)	+n*_factor_upchan+i)*2];
              tmp128[index++] = transposed_output[(b*(_samples_per_data_set+32) +n*_factor_upchan+i)*2+1];
            }
            upchannelize(tmp128, _factor_upchan);
            for (int i=0;i<_factor_upchan;i++){
              cpu_beamform_output[(b*_samples_per_data_set+n*_factor_upchan+i)*2] = tmp128[i*2];
              cpu_beamform_output[(b*_samples_per_data_set+n*_factor_upchan+i)*2+1] = tmp128[i*2+1];
            }
          }
        }

        //Downsample
        int nfreq_out = _factor_upchan/_downsample_freq;
        int nsamp_out = _samples_per_data_set/_factor_upchan/_downsample_time;
        for (int b=0;b< 1024; b++){
          for (int t=0;t<nsamp_out;t++){
            for (int f=0;f< nfreq_out;f++){
              int out_id = b*nsamp_out*nfreq_out + t*nfreq_out + f;
              float out_real=0.0;
              float out_imag = 0.0;
              for (int pp=0;pp<npol;pp++){
                for (int tt=0;tt<_downsample_time;tt++){
                  for (int ff=0;ff<_downsample_freq;ff++){
                    out_real += cpu_beamform_output[(pp*1024*_samples_per_data_set+b*_samples_per_data_set+(t*_downsample_time+tt)*_factor_upchan+(f*_downsample_freq+ff))*2];
                    out_imag += cpu_beamform_output[(pp*1024*_samples_per_data_set+b*_samples_per_data_set+(t*_downsample_time+tt)*_factor_upchan+(f*_downsample_freq+ff))*2 +1];
                  }
                }
              }
              cpu_final_output[out_id] = (out_real*out_real+out_imag*out_imag)/48.;
	          //output  = sqrt(out_real*out_real+out_imag*out_imag)/48.;
            }
          }
        }

        for (int i = 0; i < output_buf->frame_size/sizeof(float); i++) {
            output[i] = (float)cpu_final_output[i];
	    }

        INFO("Simulating GPU beamform processing done for %s[%d] result is in %s[%d]",
                input_buf->buffer_name, input_buf_id,
                output_buf->buffer_name, output_buf_id);

        //pass_metadata(&input_buf, input_buf_id, &output_buf, output_buf_id);
        mark_frame_empty(input_buf, unique_name.c_str(), input_buf_id);
        mark_frame_full(output_buf, unique_name.c_str(), output_buf_id);

        input_buf_id = (input_buf_id + 1) % input_buf->num_frames;
        output_buf_id = (output_buf_id + 1) % output_buf->num_frames;
    }
}

