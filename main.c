#include <stdio.h>
#include <stdlib.h>
 
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
 
#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x100000)
#define MAT_AT(m, i, j) m.arr[i*m.size+j]
 
typedef struct _Mat {
    int size;
    double *arr;
} Mat;

double randfrom(double min, double max) 
{
    double range = (max - min); 
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

void rand_arr(double *arr, int size) {
    for(int i=0; i<size; i++) {
        for(int j=0; j<size; j++) {
            arr[i*size+j] = randfrom(5.0, 40.0);
        }
    }
}

Mat createMat(int size) {
    Mat m;
    m.size = size;
    m.arr = (double*)malloc(sizeof(double)*size*size);
    
    return m;
}

void rand_mat(Mat m) {
    for(int i=0; i<m.size; i++) {
        for(int j=0; j<m.size; j++) {
            MAT_AT(m, i, j) = randfrom(5.0, 40.0);
        }
    }
}

void print_arr(double *arr, int size) {
    for(int i=0; i<size; i++) {
        for(int j=0; j<size; j++) {
            printf("%5.2f ", arr[i*size+j]);
        }
        printf("\n");
    } 
}

int main(int argc, char **argv)
{
    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_mem memobj = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret;
 
    char string[MEM_SIZE];
 
    FILE *fp;
    char fileName[] = "./main.cl";
    char *source_str;
    size_t source_size;
 
    /* Load the source code containing the kernel*/
    fp = fopen(fileName, "r");
    if (!fp) {
    	fprintf(stderr, "Failed to load kernel.\n");
    	exit(1);
    }

    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    printf("source_size: %d\n", (int)source_size);
    close(fp);
 
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);

    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
    printf("platform_id: %ld\n", (int)platform_id);
    printf("ret: %s\n", (int)ret);
    printf("ret_num_platforms: %u\n", (int)ret_num_platforms);
    printf("CL_DEVICE_TYPE_DEFAULT: %d\n", CL_DEVICE_TYPE_DEFAULT);
    printf("device_id: %d\n", (int)device_id);
    printf("ret_num_devices: %d\n", (int)ret_num_devices);
 
	/* Create OpenCL context */
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    printf("context: %s\n", context);
 
    /* Create Command Queue */
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
 
    /* Create Memory Buffer */
    memobj = clCreateBuffer(context, CL_MEM_READ_WRITE,MEM_SIZE * sizeof(char), NULL, &ret);
 
    /* Create Kernel Program from the source */
    program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
    /* Build Kernel Program */
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
 
    /* Create OpenCL Kernel */
    kernel = clCreateKernel(program, "vectorAdd", &ret);

    int mat_size = atoi(argv[1]);
    double *a = (double*)malloc(sizeof(double)*mat_size*mat_size);
    double *b = (double*)malloc(sizeof(double)*mat_size*mat_size);
    double *c = (double*)malloc(sizeof(double)*mat_size*mat_size);

    rand_arr(a, mat_size);
    rand_arr(b, mat_size);

    cl_mem hDeviceMemA, hDeviceMemB, hDeviceMemC; 
    hDeviceMemA = clCreateBuffer(context, CL_MEM_READ_ONLY| CL_MEM_COPY_HOST_PTR, mat_size*mat_size * sizeof(cl_float), a, &ret); 
    hDeviceMemB = clCreateBuffer(context, CL_MEM_READ_ONLY| CL_MEM_COPY_HOST_PTR, mat_size*mat_size * sizeof(cl_float), b, &ret); 
    hDeviceMemC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, mat_size*mat_size * sizeof(cl_float), c, &ret); 

    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&hDeviceMemA); 
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&hDeviceMemB); 
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&hDeviceMemC);

    size_t globalWorkSize, localWorkSize;
    globalWorkSize = mat_size*mat_size*3;
    localWorkSize =  mat_size*mat_size;
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
 
    clEnqueueReadBuffer(context, hDeviceMemC, CL_TRUE, 0, mat_size*mat_size * sizeof(cl_float), c, 0, 0, 0); 

    /* Execute OpenCL Kernel */
    ret = clEnqueueTask(command_queue, kernel, 0, NULL,NULL);
 
    /* Copy results from the memory buffer */
    ret = clEnqueueReadBuffer(command_queue, memobj, CL_TRUE, 0, MEM_SIZE * sizeof(char),string, 0, NULL, NULL);
    print_arr(c, mat_size);
    
 
    /* Finalization */
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(memobj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
	 
    free(source_str);
    free(a);
    free(b);
    free(c);

    //clReleaseMemObj(hDeviceMemA); 
    //clReleaseMemObj(hDeviceMemB); 
    //clReleaseMemObj(hDeviceMemC);
 
    return 0;
}