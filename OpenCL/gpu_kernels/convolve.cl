
#pragma OPENCL EXTENSION cl_intel_printf : enable




__kernel void convolve(__global float* data, __constant float* h,
__global float* output,const int NKx, const int NKy){


     int i0 = get_global_id(0);
     int j0 = get_global_id(1);
     int Nx = get_global_size(0);
     int Ny = get_global_size(1);

    float res = 0.f;

    for(int i = 0;i< NKx;i++)
        for(int j = 0;j< NKy;j++){

        int i2 = (i0+i-NKx/2);
        int j2 = (j0+j-NKy/2);

        if ((i2>=0) && (i2<Nx) &&(j2>=0) && (j2<Ny))
            res += h[i+NKx*j]*data[i2+Nx*j2];

    }

    output[i0+Nx*j0] = res;

    
    
//    output[i0+Nx*j0] = i0;

}
