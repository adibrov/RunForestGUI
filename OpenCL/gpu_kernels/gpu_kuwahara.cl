__kernel void kuwahara(__global float* data, __global float* output, const int NK)
{
//    unsigned int i = get_global_id(0);


    int i0 = get_global_id(0);
    int j0 = get_global_id(1);
    int Nx = get_global_size(0);
    int Ny = get_global_size(1);

    float res = 0.f;
    float norm = 0.f;

    norm = 1.0/((NK/2+1)*(NK/2+1));

    float mean[4];
    float std[4];

    for (int i=0;i<4;i++){
      mean[i] = 0.0;
      std[i] = 0.0;
    }
 

    float data_center = data[i0+Nx*j0];

    //---------------------- mean ---------------------//
    
    for(int i = 0;i< NK/2+1;i++)
        for(int j = 0;j< NK/2+1;j++){

        int i2 = (i0+i-NK/2);
        int j2 = (j0+j-NK/2);

        if ((i2>=0) && (i2<Nx) &&(j2>=0) && (j2<Ny)){
             mean[0] += data[i2+Nx*j2];

        }    
    }

    for(int i = NK/2;i< NK;i++)
        for(int j = 0;j< NK/2+1;j++){

        int i2 = (i0+i-NK/2);
        int j2 = (j0+j-NK/2);

        if ((i2>=0) && (i2<Nx) &&(j2>=0) && (j2<Ny)){
      
            mean[1] += data[i2+Nx*j2];

        }    
    }

    for(int i = 0;i< NK/2+1;i++)
        for(int j = NK/2;j< NK;j++){

        int i2 = (i0+i-NK/2);
        int j2 = (j0+j-NK/2);

        if ((i2>=0) && (i2<Nx) &&(j2>=0) && (j2<Ny)){
      
            mean[2] += data[i2+Nx*j2];

        }    
    }

    for(int i = NK/2;i< NK;i++)
        for(int j = NK/2;j< NK;j++){

        int i2 = (i0+i-NK/2);
        int j2 = (j0+j-NK/2);

        if ((i2>=0) && (i2<Nx) &&(j2>=0) && (j2<Ny)){
      
            mean[3] += data[i2+Nx*j2];

        }    
    }

    for(int i=0;i<4;i++){
      mean[i] = norm*mean[i];
    }


    
    // ---------- std ----------------//
    for(int i = 0;i< NK/2+1;i++)
        for(int j = 0;j< NK/2+1;j++){

        int i2 = (i0+i-NK/2);
        int j2 = (j0+j-NK/2);

        if ((i2>=0) && (i2<Nx) &&(j2>=0) && (j2<Ny)){
      
	  std[0] += pow(data[i2+Nx*j2]-mean[0],2);
	  
        }    
    }

    for(int i = NK/2;i< NK;i++)
        for(int j = 0;j< NK/2+1;j++){

        int i2 = (i0+i-NK/2);
        int j2 = (j0+j-NK/2);

        if ((i2>=0) && (i2<Nx) &&(j2>=0) && (j2<Ny)){
      
	  std[1] += pow(data[i2+Nx*j2]-mean[1],2);

        }    
    }

    for(int i = 0;i< NK/2+1;i++)
        for(int j = NK/2;j< NK;j++){

        int i2 = (i0+i-NK/2);
        int j2 = (j0+j-NK/2);

        if ((i2>=0) && (i2<Nx) &&(j2>=0) && (j2<Ny)){
      
	  std[2] += pow(data[i2+Nx*j2]-mean[2],2);
	  
        }    
    }

    for(int i = NK/2;i< NK;i++)
        for(int j = NK/2;j< NK;j++){

        int i2 = (i0+i-NK/2);
        int j2 = (j0+j-NK/2);

        if ((i2>=0) && (i2<Nx) &&(j2>=0) && (j2<Ny)){
      
	  std[3] += pow(data[i2+Nx*j2]-mean[3],2);

        }    
    }

    for(int i=0;i<4;i++){
      std[i] = sqrt(norm*std[i]);
    }

    float min_sigma = 1000000;
    

    
    for(int i=0;i<4;i++)
      if (min_sigma>std[i]){
	min_sigma = std[i];
	res = mean[i];
      }
    


    output[i0+Nx*j0] = res;
    // output[1+5*i0+5*Nx*j0] = std[0];
    // output[2+5*i0+5*Nx*j0] = std[1];
    // output[3+5*i0+5*Nx*j0] = std[2];
    // output[4+5*i0+5*Nx*j0] = std[3];

    

    
//    output[i0+Nx*j0] = i0;




    
}


