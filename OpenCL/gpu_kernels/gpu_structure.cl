__kernel void structure(__global float* data, __global float* output)
{
//    unsigned int i = get_global_id(0);


    int i0 = get_global_id(0);
    int j0 = get_global_id(1);
    int Nx = get_global_size(0);
    int Ny = get_global_size(1);

    int NKx = 3;
    int NKy = 3;

    float res = 0.f;
    float norm = 0.f;

    float a = 0.f;
    float b = 0.f;
    float c = 0.f;
    float d = 0.f;

    float first_eig = 0.f;
    float second_eig = 0.f;







  //  int i2 = (i0+i-NKx/2);
//    int j2 = (j0+j-NKy/2);

    int i; i = i0;
    int j; j = j0;

    if ((i>=1) && (i<Nx-1) &&(j>=1) && (j<Ny-1)){

      a = 0.25*pow((1.0*data[i0+Nx*(j0+1)] - 1.0*data[i0+Nx*(j0-1)]),2);
      b = (1.0*data[i0+Nx*(j0+1)] - 1.0*data[i0+Nx*(j0-1)])*(1.0*data[i0+1+Nx*(j0)] - 1.0*data[i0-1+Nx*(j0)]);
      d = 0.25*pow((1.0*data[i0+1+Nx*(j0)] - 1.0*data[i0-1+Nx*(j0)]),2);
      




         first_eig = 0.5*(a + d) + sqrt(0.5*(4*pow(b,2) + pow(a-d,2)));
         second_eig = 0.5*(a + d) - sqrt(0.5*(4*pow(b,2) + pow(a-d,2)));

        
       
        
    }       


    
  
    output[0+2*i0+2*Nx*j0] = first_eig;
    output[1+2*i0+2*Nx*j0] = second_eig;
  

    //output[i0+Nx*j0] = res/norm;

    
    
//    output[i0+Nx*j0] = i0;




    
}
