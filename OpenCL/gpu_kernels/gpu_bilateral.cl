__kernel void bilateral(__global float* data, __global float* output, const int NKx, const int NKy, const float sigma_int, const float sigma_dist)
{
//    unsigned int i = get_global_id(0);


    int i0 = get_global_id(0);
    int j0 = get_global_id(1);
    int Nx = get_global_size(0);
    int Ny = get_global_size(1);

    float res = 0.f;
    float norm = 0.f;


    float data_center = data[i0+Nx*j0];



    for(int i = 0;i< NKx;i++)
        for(int j = 0;j< NKy;j++){

        int i2 = (i0+i-NKx/2);
        int j2 = (j0+j-NKy/2);

   
        if ((i2>=0) && (i2<Nx) &&(j2>=0) && (j2<Ny)){

          
  		  float value = data[i2+Nx*j2];

		  float del_data = value - data_center;
		  float del_x = (i-NKx/2);
		  float del_y = (j-NKy/2);

		  float weight = exp(-0.5*(pow(del_x,2)+pow(del_y,2))/(pow(sigma_dist,2)) - 0.5*(pow(del_data,2)/(pow(sigma_int,2))));

		  
		  res += value*weight;
		  norm += weight;
        }       


    }

    output[i0+Nx*j0] = res/norm;

    
    
//    output[i0+Nx*j0] = i0;




    
}


