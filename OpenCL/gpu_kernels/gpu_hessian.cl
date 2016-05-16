

__kernel void hessian(__global float* data, __global float* output)
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
    float t = 0.f;
    float module = 0.f;
    float trace = 0.f;
    float determinant = 0.f;
    float first_eig = 0.f;
    float second_eig = 0.f;
    float orientation = 0.f;
    float square_diff = 0.f;
    float diff_square = 0.f;





  //  int i2 = (i0+i-NKx/2);
//    int j2 = (j0+j-NKy/2);

    int i; i = i0;
    int j; j = j0;

    if ((i>=1) && (i<Nx-1) &&(j>=1) && (j<Ny-1)){

        a = 1.0*data[i0+Nx*(j0-1)] - 2.0*data[i0+Nx*j0] + 1.0*data[i0+Nx*(j0+1)]    ;
        b = 0.25*(data[i0+1+Nx*(j0+1)] - data[i0+1+Nx*j0] - data[i0+Nx*(j0+1)] + 2.0*data[i0+Nx*j0] - data[i0-1+Nx*j0] - data[i0+Nx*(j0-1)] + data[i0-1+Nx*(j0-1)]);
        c = b;
        d = 1.0*data[i0-1+Nx*j0] - 2.0*data[i0+Nx*j0] + 1.0*data[i0+1+Nx*j0];

         t = 1.f; // check the meaning of this parameter;
         module = sqrt(pow(a,2) + b*c +pow(d,2));
         trace = a + d;
         determinant = a*d - c*b;
         first_eig = 0.5*(a + d) + sqrt(0.5*(4*pow(b,2) + pow(a-d,2)));
         second_eig = 0.5*(a + d) - sqrt(0.5*(4*pow(b,2) + pow(a-d,2)));
	 float bo = (0.5*(a-d)+sqrt(0.5*(4*pow(b,2) + pow(a-d,2))));
	 if ((bo >= 0.0)&& (bo<=0.001))
	     orientation = 3.141592/2.0;
	 else if ((bo <= 0.0)&&(bo>=-0.001))
	   orientation = -3.141592/2.0;
	 else
	   orientation = -b/bo;
        
         square_diff = pow(t,4)*(pow(a-d,2))*((pow(a-d,2)) - 4*pow(b,2));
         diff_square = (pow(t,2))*((pow(a-d,2)) + 4*pow(b,2));
        
       
        
    }       


    
    output[0+8*i0+8*Nx*j0] = module;
    output[1+8*i0+8*Nx*j0] = trace;
    output[2+8*i0+8*Nx*j0] = determinant;
    output[3+8*i0+8*Nx*j0] = first_eig;
    output[4+8*i0+8*Nx*j0] = second_eig;
    
    // if (orientation >=0.0)
    //   output[5+8*i0+8*Nx*j0] = orientation;
    // else
    output[5+8*i0+8*Nx*j0] = atan(orientation);
    
    output[6+8*i0+8*Nx*j0] = square_diff;
    output[7+8*i0+8*Nx*j0] = diff_square;

    //output[i0+Nx*j0] = res/norm;

    
    
//    output[i0+Nx*j0] = i0;




    
}
