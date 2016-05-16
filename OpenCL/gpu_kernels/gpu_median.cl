
// float sele(int *x, int size, int k){

//       	int left = 0;
//     	int right = size-1;
 
//     	//we stop when our indicies have crossed
//     	while (left < right){
 
//     		int pivot = (left + right)/2; //this can be whatever
//     		int pivotValue = x[pivot];
//     		int storage=left;
 
//     		x[pivot] = x[right];
//     		x[right]=pivotValue;
//     		for(int i =left; i < right; i++){//for each number, if its less than the pivot, move it to the left, otherwise leave it on the right
//     			if(x[i] < pivotValue){
//     				int temp =x[storage];
//     				x[storage] = x[i];
//     				x[i]=temp;
//     				storage++;
//     			}
//     		}
//     		x[right]=x[storage];
//     		x[storage]=pivotValue;//move the pivot to its correct absolute location in the list
 
//     		//pick the correct half of the list you need to parse through to find your K, and ignore the other half
//     		if(storage < k)
//     			left = storage+1;
//     		else//storage>= k
//     			right = storage;
 
//     	}
//     	return x[k];

// }

__kernel void median(__global float* data, __global float* output, const int NKx, const int NKy)
{
    int i0 = get_global_id(0);
    int j0 = get_global_id(1);
    int Nx = get_global_size(0);
    int Ny= get_global_size(1);

    float res = 0.f;
    
    //   res = data[i0+Nx*j0];
    // int dc = data[i0+Nx*j0];
    
    //    int i2_begin = i0 - (NKx*NKy/2);

    //    int i2_end = j0 + (NKx*NKy/2);

    // const int NKx = 10;
    // const int NKy = 10;
    
    //    float arr[NKy*NKx];
    float arr[121];
    
    for(int i = 0;i< NKx;i++)
        for(int j = 0;j< NKy;j++){

        int i2 = (i0+i-NKx/2);
        int j2 = (j0+j-NKy/2);

   
        if ((i2>=0) && (i2<Nx) &&(j2>=0) && (j2<Ny)){

	  	  arr[i+NKx*j] = data[i2+Nx*j2];
	  
        }       
	else{
	   arr[i+NKx*j] = 0.0;
	}
    }   
    

    //    int elem = sizeof(arr)/sizeof(arr[0]);
//    std::sort(arr,arr+elem);




    

// Sorting
//-----------------------------------------------------------------//
    
     //  int i, j, flag = 1;    // set flag to 1 to start first pass
     //  int temp;             // holding variable
     //  int length = 150; 
     //  for(i = 1; (i <= length) && flag; i++)
     // {
     //      flag = 0;
     //      for (j=0; j < (length -1); j++)
     //     {
     //           if (arr[j+1] > arr[j])      // ascending order simply changes to <
     //          { 
     //                temp = arr[j];             // swap elements
     //                arr[j] = arr[j+1];
     //                arr[j+1] = temp;
     //                flag = 1;               // indicates that a swap occurred.
     //           }
     //      }
     // }

     //    res = arr[NKx*NKy/2];

    int k = NKx*NKy/2;
    int left = 0;
    int right = NKx*NKy -1;
	while (left < right){

		int pivot = (left + right)/2; //this can be whatever
		int pivotValue = arr[pivot];
		int storage=left;

		arr[pivot] = arr[right];
		arr[right]=pivotValue;
		for(int i =left; i < right; i++){//for each number, if its less than the pivot, move it to the left, otherwise leave it on the right
			if(arr[i] < pivotValue){
				int temp =arr[storage];
				arr[storage] = arr[i];
				arr[i]=temp;
				storage++;
			}
		}
		arr[right]=arr[storage];
		arr[storage]=pivotValue;//move the pivot to its correct absolute location in the list

		//pick the correct half of the list you need to parse through to find your K, and ignore the other half
		if(storage < k)
			left = storage+1;
		else//storage>= k
			right = storage;

}



	
//    res = 5.0;


//--------------------------------------------------------------//




    
    int size = NKx*NKy;
    int med = size/2;
    // float aux = sele(arr,size,med);
    output[i0+Nx*j0] = arr[k];//sele(arr,121,60);
}


