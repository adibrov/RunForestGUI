#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

float partition(float lis[100],int left, int right, int pi)
    {
      float pv = lis[pi];

      lis[pi] = lis[right];
      lis[right] = pv;

      int stor = left;

      for (int i=left; i<=right-1;i++)
      {
	if (lis[i] < pv)
	{
	  float aux = lis[i];
	  lis[i] = lis[stor];
	  lis[stor] = aux;
	  stor++;	
	}
      }

      float aux1 = lis[right];
      lis[right] = lis[stor];
      lis[stor] = aux1;

      return stor;

     }


float select(float lis[100], int left, int right, int n)
    {
      if (left ==right)
      {
	return lis[left];
      }

      //   srand(time(NULL));
      int pi = (right + left)/2;
      pi = partition(lis,left,right,pi);

      if (n==pi)
	return lis[n];	
      
      else if (n<pi)
	return select(lis,left,pi-1,n);
      else
	return select(lis, pi+1, right, n);
    }

float bubble(float arr[100]){
      int i, j, flag = 1;    // set flag to 1 to start first pass
      int temp;             // holding variable
      int length = 100; 
      for(i = 1; (i <= length) && flag; i++)
     {
          flag = 0;
          for (j=0; j < (length -1); j++)
         {
               if (arr[j+1] > arr[j])      // ascending order simply changes to <
              { 
                    temp = arr[j];             // swap elements
                    arr[j] = arr[j+1];
                    arr[j+1] = temp;
                    flag = 1;               // indicates that a swap occurred.
               }
          }
     }
      return arr[50];
}


int main()
{

  using namespace std;
  cout<<"hello!"<<endl;

  float arr[100];

  srand(time(NULL));
  
  for (int i=0;i<100;i++){
    arr[i] = (rand()%100);
    //    cout<<i<<" "<<arr[i]<<endl;
   }

  //--------------------select----------//
   float se = select(arr,0,99,49);
    float bu = bubble(arr);
  
   cout<<"res from select "<<se<<endl;
   cout<<"res from bubble "<<bu<<endl;
   cout<<"arr 50 is "<<arr[50]<<endl;
   cout<<"arr 49 is "<<arr[49]<<endl;


  // for (int i=0;i<100;i++){
  //   arr[i] = (rand()%100);
  //   cout<<i<<" "<<arr[i]<<endl;
  //  }


      int i, j, flag = 1;    // set flag to 1 to start first pass
      int temp;             // holding variable
      int length = 100; 
      for(i = 1; (i <= length) && flag; i++)
     {
          flag = 0;
          for (j=0; j < (length -1); j++)
         {
               if (arr[j+1] > arr[j])      // ascending order simply changes to <
              { 
                    temp = arr[j];             // swap elements
                    arr[j] = arr[j+1];
                    arr[j+1] = temp;
                    flag = 1;               // indicates that a swap occurred.
               }
          }
     }

      cout<<"done"<<endl;

      // for (int i=0;i<100;i++){
      // 	cout<<arr[i]<<" "<<endl;
      // }


      
}
