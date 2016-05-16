

__kernel void add(__global float* a, __global float* b, __global float* res)
{
    unsigned int i = get_global_id(0);


    res[i] = a[i] + b[i];

    
}

__kernel void add2(__global float* a, const float val,__global float* res)
{
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);
    unsigned int Nx = get_global_size(0);


    res[i+Nx*j] = a[i+Nx*j] + val;

    

    
}