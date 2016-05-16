__kernel void variance(__global float* data, __global float* output, const int NKx, const int NKy)
{
//    unsigned int i = get_global_id(0);


    int i0 = get_global_id(0);
    int j0 = get_global_id(1);
    int Nx = get_global_size(0);
    int Ny = get_global_size(1);

    float res = 0.f;
    float norm = 0.f;

    norm = 1.0/(NKx*NKy);

    float mean = 0.f;;
    float var = 0.f;



    float data_center = data[i0+Nx*j0];



    for(int i = 0;i< NKx;i++)
      for(int j = 0;j< NKy;j++){

        int i2 = (i0+i-NKx/2);
        int j2 = (j0+j-NKy/2);

        if ((i2>=0) && (i2<Nx) &&(j2>=0) && (j2<Ny)){
             mean += data[i2+Nx*j2];

        }    
    }

    mean = mean*norm;

    for(int i = 0;i< NKx;i++)
        for(int j = 0;j< NKy;j++){

        int i2 = (i0+i-NKx/2);
        int j2 = (j0+j-NKy/2);

        if ((i2>=0) && (i2<Nx) &&(j2>=0) && (j2<Ny)){
      
	  var += pow(data[i2+Nx*j2]-mean,2);
	  
        }    
    }


    var = var*norm;
    
    output[i0+Nx*j0] = var;

    
//    output[i0+Nx*j0] = i0;




    
}


