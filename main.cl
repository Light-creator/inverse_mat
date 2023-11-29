__kernel void vectorAdd(
__global const double *A,
__global const double *B, 
__global double *C, int n) 
{ 
// Vector element index 
  int i = get_global_id(0); 

  if (i < n) C[i] = A[i] + B[i]; 
}
