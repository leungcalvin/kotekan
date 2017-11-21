/*********************************************************************************

Kotekan RFI Documentation Block:
By: Jacob Taylor 
Date: August 2017
File Purpose: OpenCL Kernel to flag RFI in CHIME data
Details:
	Sums power and power**2 across time
	Normalizes
	Sums power and power**2 across elements
	Compute Spectral Kurtosis
	Flags RFI

**********************************************************************************/

__kernel void
rfi_chime(
	  __global char* input, //Input buffer
	  __global unsigned int *count, //Amount of each frequency zeroed
	  __global float* in_means, //Normilization means
	  float sqrtM, //Square root of M
	  int sensitivity, //# of deviations for Kurtosis threshold
	  int time_samples, //Time samples in data block
	  int zero //Whether or no to zero data
)
{
	int gx = get_global_id(0); //Get Work Id's
	int gy = get_global_id(1);
	int gz = get_global_id(2);
	int lx = get_local_id(0);
	int gx_size = get_global_size(0); //#Elements
	int gy_size = get_global_size(1); //#Frequencies 
	int gz_size = get_global_size(2); //#Time Sample/SK_STEP
	int lx_size = get_local_size(0);
	int SK_STEP = time_samples/gz_size;

	__local float power_across_input[256]; //Local Memory
	__local float sq_power_across_input[256];
	__local bool Zero_Flag;
	
	__local unsigned int power_across_time[128]; //Local Memory
	__local unsigned int sq_power_across_time[128];

	power_across_time[gz] = 0; //Intialize Local Memory
	sq_power_across_time[gz] = 0;
	
	for(int i =0; i < SK_STEP; i++){ //Sum across time
		char data_point = input[gx + gy*gx_size + (SK_STEP*gz + i)*(gx_size*gy_size)];
	   	char real, imag;
	   	unsigned int power; //Compute power
		real = ((data_point >> 4) & 0xF)-8;
	   	imag = (data_point & 0xF)-8;
		power = real*real + imag*imag;
		in_means[gx + gy*gx_size]  += 0.01*(power-in_means[gx + gy*gx_size]); //Adjust mean
		power_across_time[gz] += power;
	   	sq_power_across_time[gz] += (power*power);
	}

	if(gx == 0 && gz == 0) count[gy] = 0; //Initialize Count array

	barrier(CLK_LOCAL_MEM_FENCE); //Wait for all to finish
	barrier(CLK_GLOBAL_MEM_FENCE);

	float Median = in_means[gx + gy*gx_size]*(512.0/729.0); //Adjust Mean to Median

	power_across_input[gx/(gx_size*gy_size/256)] = (float)power_across_time[gz]/Median; //Save Values
	sq_power_across_input[gx/(gx_size*gy_size/256)] = (float)sq_power_across_time[gz]/(Median*Median);

	barrier(CLK_LOCAL_MEM_FENCE); //Wait for all to finish

	for(int i = lx_size/2; i>0; i >>= 1){ //Reduce along Elements
        	if(lx < i){
            		power_across_input[lx] += power_across_input[lx + i];
			sq_power_across_input[lx] += sq_power_across_input[lx + i];
       		}
        	barrier(CLK_LOCAL_MEM_FENCE);
    	}

	barrier(CLK_LOCAL_MEM_FENCE); //Wait for all to finish

	if(lx == 0){ //Calculate SK and Flag
		unsigned int M = gx_size*SK_STEP;
		float SK = ((((float)M+1)/(M-1))*(M*(sq_power_across_input[0])/(power_across_input[0]*power_across_input[0])-1));
		Zero_Flag = (SK < (1 - (sensitivity*2.0)/sqrtM) || SK > (1 + (sensitivity*2.0)/sqrtM));
		if(Zero_Flag) count[gy] += SK_STEP; //Adjust count
	}

	barrier(CLK_LOCAL_MEM_FENCE); //Wait for all to finish

	for(int j = 0; j < SK_STEP; j++){ //Zero flagged values
		unsigned int data_location = gx + gy*gx_size + (SK_STEP*gz + j)*(gx_size*gy_size);
		if(Zero_Flag && zero == 1) input[data_location] = 0x88; //Zero
	}
}
