/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <assert.h>
#define DEBUG
#define GPUS 4

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}


void profileCopies(float        *h_a, 
                   float        **d, 
                   unsigned int  n,
                   unsigned int  iterations)
{
  unsigned int bytes = n * sizeof(float);

  // events for timing
  cudaEvent_t startEvent, stopEvent; 
  
  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );
 
  cudaStream_t stream[GPUS];

  for (int i = 0; i < GPUS; i++)
  {
    checkCuda( cudaMemcpy(d[i], h_a, bytes, cudaMemcpyHostToDevice) );
    checkCuda( cudaStreamCreate(&stream[i]) );
  }

  checkCuda( cudaEventRecord(startEvent, 0) );
  for (int i = 0; i < iterations; i++)
  {
    for (int j = 0; j < GPUS; j++)
    {
      checkCuda( cudaMemcpyAsync(d[(j+1)%GPUS], d[j%GPUS], bytes, cudaMemcpyDeviceToDevice, stream[j] ) );
    }
    checkCuda( cudaDeviceSynchronize());
  }

  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );

  float time;
  checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
  printf("Total time = %f\n", time);
  printf("Avg time per trip= %f\n", time/iterations);
  printf("Device to Device roundtrip bandwidth (GB/s): %f\n", bytes * 1e-6 * iterations / time);

  // clean up events
  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );


  float *h_b[GPUS];

  for (int i = 0; i < GPUS; i++)
  {
    h_b[i] = (float*)malloc(bytes);
    checkCuda( cudaMemcpy(h_b[i], d[i], bytes, cudaMemcpyDeviceToHost) );
  }

}

int main()
{
  unsigned int nElements = 4*1024*1024;
  const unsigned int bytes = nElements * sizeof(float);

  // host arrays
  float *h_aPageable;   

  // device array
  float *d_a[GPUS];

  // allocate and initialize
  h_aPageable = (float*)malloc(bytes);                    // host pageable

  cudaDeviceProp prop[GPUS];
  for (int i = 0; i < GPUS; i++) 
  { 
    checkCuda (cudaSetDevice(i) );
    checkCuda( cudaMalloc((void**)&d_a[i], bytes) );           // device
    checkCuda( cudaGetDeviceProperties(&prop[i], i) );
    printf("Device %d: %s\n", i, prop[i].name);
  }

  for (int i = 0; i < nElements; ++i) h_aPageable[i] = i;      

  printf("Transfer size (MB): %d\n", bytes / (1024 * 1024));

  // perform copies and report bandwidth
  profileCopies(h_aPageable, d_a, nElements, 10000);

  printf("\n");

  // cleanup
  for (int i = 0; i < GPUS; i++)  
  {
    checkCuda (cudaSetDevice(i) );
    cudaFree(d_a[i]);
  }

  free(h_aPageable);

  return 0;
}
