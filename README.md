Download Link: https://assignmentchef.com/product/solved-cs3210-lab-3-gpu-parallel-programming-in-cuda
<br>
This Lab aims to demonstrate the basic concepts in CUDA – a parallel computing platform for NVIDIA GPUs. Details on CUDA, GPGPU and architecture will be provided in the relevant sections of this Lab. This Lab serves as a precursor to Assignment 1 Part 2.

Part 1: Hello World in CUDA

This section will cover the basics on how to create, compile and execute a CUDA program. Let us do so by using the hello.cu program.

<strong>What is a CUDA program?</strong>

A CUDA program, with the extension .cu, is a modified C++ program with sections of CUDA code that are executed on the GPU, called kernels. In CUDA’s terminology, the CPU is called the <strong>host </strong>and the GPU is called the <strong>device</strong>.

<h2>CPU vs GPU Code</h2>

In CUDA, the programmer has to explicitly separate the code that will run on the CPU from the code that runs on the GPU.

<h2>Function execution space specifiers</h2>

The separation of CPU and GPU code is done on the method/function level. That is to say, a C++ function can be marked as “GPU” or “CPU” code. This is done using <strong>function execution space specifiers </strong>that are placed before the return type, as follows

__global__ void hello(char *a, int len)

{

int tid = threadIdx.x;

if (tid &gt;= len)

return;

a[tid] += ‘A’ – ‘a’;

}

1

Here are some function specifiers and their effects:

<ul>

 <li>__device__ – will execute on the device. Can only be called from the device.</li>

 <li>__global__ – will execute on the device. Callable from both the device and host. Usually serves as the entry point to the GPU kernel (CUDA kernel function).</li>

 <li>__host__ – executes on the host. Can only be called from the host. Alternatively, a function with <strong>no specifiers </strong>is deemed to be __host__.</li>

</ul>

For more information on function specifiers, refer to the <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#function-declaration-specifiers">CUDA Programming Guide Section B.1</a>



<h2>Organizing Threads in CUDA</h2>

This section explains how CUDA threads are laid out from the <strong>programmer’s perspective</strong>.

<h3>Threads, Blocks and Grid</h3>

An invocation of a CUDA kernel function (i.e. functions annotated with __global__) launches a new grid. The grid is a collection of <strong>thread blocks</strong>. A <strong>thread block </strong>contains a collection of <strong>CUDA threads</strong>. Threads in a block can be laid out in one, two or three dimensions. Similarly, blocks in a grid can be laid out in one, two or three dimensions. Figure 1 shows an example of thread organization.

Figure 1: Organization of threads, blocks within grid

The programmer has to explicitly specify the number of threads per block and the number of blocks per grid.

All blocks within the same grid will have the same number of threads, laid out in the same fashion.

<h3>Specifying thread organization in CUDA</h3>

To specify the size of the block or grid, we can either:

<ul>

 <li>Specify an integer denoting the number of threads/blocks. This is only applicable if we want a 1 dimensional block/grid.</li>

 <li>Declare a variable of type dim3 dim3 is a type defined by CUDA that stores the values that require dimensions. We can define a dim3 variable as follows:</li>

 <li>dim3 threadsPerBlock(4, 4, 4); – creates a variable called threadsPerBlock. In our use case, the value is a grid/block that is three dimensional, with 4 blocks/threads on each dimension</li>

 <li>dim3 blocksPerGrid(8, 4); – creates a variable blocksPerGrid. Two dimensional, with the x dimension having 8 elements, while the y dimension has 4 elements.</li>

</ul>

Such variables are then passed when the program makes the CUDA kernel function call, covered in the next section.

<h2>Invoking a CUDA kernel function</h2>

The block and grid dimensions have to be supplied when we call a kernel function. Recall that a grid is launched when a CUDA kernel function is invoked. Thus, this is when CUDA needs to know the grid and block sizes for that particular invocation. The kernel <strong>has to be invoked from a host function </strong>and this is done with a modified syntax as follows:

kernel_name&lt;&lt;&lt; Dg, Db, Ns, S &gt;&gt;&gt;([kernel arguments]);

<ul>

 <li>Dg is of type dim3 and specifies the dimensions and size of the grid</li>

 <li>Db is of type dim3 and specifies the dimensions and size of each thread block</li>

 <li>Ns is of type size_t and specifies the number of bytes of shared memory that is dynamically allocated per thread block for this call and addition to statically allocated memory. (<strong>Optional</strong>, defaults to 0)</li>

 <li>S is of type cudaStream_t and specifies the stream associated with this call. The stream must have been allocated in the same thread block where the call is being made. (<strong>Optional</strong>, defaults to 0)</li>

</ul>

For instance, if we have a kernel function (i.e. annotated with the __global__ specifier) called void convolve(int a, int b) and we want to invoke it with the following thread dimensions:

<ul>

 <li>The grid has dimension 3×<sub>3</sub>×<sub>3</sub></li>

 <li>A block in the grid has dimension 4<sub>×3</sub></li>

</ul>

Then, we write the following code in a <strong>host </strong>function at the point where we want to execute the kernel function:

dim3 gridDimensions(3, 3, 3);

dim3 blockDimensions(4, 3);

convolve&lt;&lt;&lt;gridDimensions, blockDimensions&gt;&gt;&gt;(5, 10);

<h2>Compiling and Execution</h2>

A CUDA program (application) is compiled using a proprietary compiler called NVIDIA CUDA Compiler (nvcc). For instance, if we want to compile hello.cu to an executable called my-hello, we can run the following command in a terminal:

&gt; nvcc -o my-hello hello.cu

In our lab, nvcc is only installed on the Jetson TX2. Thus, you need to SSH into the Jetson TX2, compile and run the CUDA programs there.

This will create an executable called my-hello in the directory where nvcc was executed. To run my-hello, we run it as though it is a normal executable as follows:

 &gt; ./my-hello

<h3>Specifying Compute Capability</h3>

Nvcc compiler defaults to target Compute Capability to 3.0 (sm_30). To target a higher Compute Capability, we should set the -arch flag as follows:

&gt; nvcc -arch sm_60 -o my-hello hello.cu

More information on nvcc can be found in the <a href="https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html">CUDA Compiler Driver NVCC</a> section of the CUDA Toolkit Documentation

• The Jetson TX2 has a Compute Capability of 6.2.                                           For more information on Compute

Capability, refer to <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities">Section H of the CUDA Programming Guide.</a>

<ul>

 <li>You may also try to run the programs on the SoC Compute Cluster. Refer to the Appendix for more information.</li>

</ul>

<table width="601">

 <tbody>

  <tr>

   <td width="60"></td>

   <td width="541"><strong><u>Exercise 1</u></strong>Compile and execute hello.cu on the Jetson TX2. Inspect the code and answer the following questions:•     How many threads per block? Blocks in the grid?•     How many threads in total?•     Draw out a diagram as per Figure 1.</td>

  </tr>

 </tbody>

</table>

<h2>Determining the thread’s location in grid/block</h2>

Once invoked, the code within the kernel runs on the device. Often computation done by the kernel is based on the thread location within the block and grid. CUDA provides the developer with <strong>built-in variables </strong>that can be called in device code to find the following:

• gridDim – the dimensions of the grid as dim3 (i.e. dimensions in X direction can be found in gridDim.x, Y in gridDim.y and Z in gridDim.z

<ul>

 <li>blockIdx – the block index the thread is running in</li>

</ul>

(blockIdx.x, blockIdx.y &amp; blockIdx.z) • blockDim – the dimensions of the block

(blockDim.x, blockDim.y &amp; blockDim.z)

<ul>

 <li>threadIdx – the index of the thread <strong>within its block</strong></li>

</ul>

(threadIdx.x, threadIdx.y &amp; threadIdx.z)

<table width="601">

 <tbody>

  <tr>

   <td width="70"></td>

   <td width="531"><strong><u>Exercise 2</u></strong>Modify hello.cu such that it runs on a 3D grid of dimension 2<sub>×2×2 </sub>containing 2D blockof 2×4.<strong>Hint: </strong>You will need to change how the tid is calculated (in the kernel function hello), so that there’s no overlapping work done.</td>

  </tr>

 </tbody>

</table>

<h2>Executing CUDA threads – hardware’s perspective</h2>

Grouping threads into blocks (and blocks into a grid) is a useful abstraction for the programmer. This section explains how these threads, which are organized by the programmer, is executed by the hardware.

<h3>Architecture of the GPU</h3>

To better understand the execution model, let us go through the architecture of the GPU. In particular, we will be using the Jetson TX2 as an example.

The GPU has a set number of Streaming Multiprocessors (SM), in the case of the Jetson TX2, it has 2 SMs. Each SM has, depending on the GPU’s architecture, a fixed number of processing blocks (SMP), 4 in the case of the Jetson TX2, as shown in Figure 2

Figure 2: Streaming Mutiprocessor (SM) for the NVIDIA Pascal Architecture with four SMPs (like the Jetson TX2)

Each <strong>thread block </strong>is assigned a <strong>Streaming Multiprocessor </strong>by the GPU. Once it is assigned, it does not migrate (i.e. change SMs). The SM will break down the thread block into <strong>groups of 32 threads each </strong>called <strong>warps</strong>. The warps execute in a Single Instruction Multiple Threads (SIMT) fashion, which is similar to an SIMD execution (i.e. lockstep execution).

If the thread block does not divide by 32 evenly, then a warp might be executing with less than 32 threads. However, the warp will be executed <strong>as though there are 32 threads</strong>.

<table width="601">

 <tbody>

  <tr>

   <td width="60"></td>

   <td width="541"><strong><u>Exercise 3</u></strong>Compile and execute slow.cu. Notice the extremely slow runtime when each block contains 1 thread. Calculate the number of warps launched when running 1024 blocks of 1 thread each, versus 1 block of 1024 threads.</td>

  </tr>

  <tr>

   <td width="60"></td>

   <td width="541"><strong><u>Exercise 4</u></strong>Compile and execute printing.cu. Notice that there is a trend in how the thread IDs are printed out. Notice any correlation with our knowledge on how the GPU executes a CUDA program?</td>

  </tr>

 </tbody>

</table>

<h1>Part 2: CUDA Memory Model</h1>

This section presents the CUDA memory model and its usage in a CUDA program.

<h2>CUDA Memory Model</h2>

CUDA offers the programmer access to various types of memory, that either resides on the host or device.

These can be further subdivided based on whether the host can access the memory in question (Refer to the Appendix for more information on memory organization for GPU). These different types of memory is akin to the memory hierarchy of computers and can be visualized in Figure 3

Figure 3: Memory Hierarchy in CUDA (or GPUs in general)

Table 1 provides a summary of the different memory types that are available in CUDA.

<table width="543">

 <tbody>

  <tr>

   <td width="66"><strong>Type</strong></td>

   <td width="62"><strong>Scope</strong></td>

   <td width="88"><strong>Access type</strong></td>

   <td width="64"><strong>Speed</strong></td>

   <td width="170"><strong>CUDA declaration syntax</strong></td>

   <td width="92"><strong>Explicit sync</strong></td>

  </tr>

  <tr>

   <td width="66">Register</td>

   <td width="62">thread</td>

   <td width="88">RW</td>

   <td width="64">fastest</td>

   <td width="170">–</td>

   <td width="92">no</td>

  </tr>

  <tr>

   <td width="66">Local</td>

   <td width="62">thread</td>

   <td width="88">RW</td>

   <td width="64">very fast</td>

   <td width="170">float x;</td>

   <td width="92">no</td>

  </tr>

  <tr>

   <td width="66">Shared</td>

   <td width="62">block</td>

   <td width="88">RW</td>

   <td width="64">fast</td>

   <td width="170">__shared__ float x;</td>

   <td width="92">yes</td>

  </tr>

  <tr>

   <td width="66">Global</td>

   <td width="62">program</td>

   <td width="88">RW</td>

   <td width="64">slow</td>

   <td width="170">__device__ float x;</td>

   <td width="92">yes</td>

  </tr>

  <tr>

   <td width="66">Constant</td>

   <td width="62">program</td>

   <td width="88">R</td>

   <td width="64">slow</td>

   <td width="170">__constant__ float x;</td>

   <td width="92">yes</td>

  </tr>

  <tr>

   <td width="66">Texture</td>

   <td width="62">program</td>

   <td width="88">R</td>

   <td width="64">slow</td>

   <td width="170">__texture__ float x;</td>

   <td width="92">yes</td>

  </tr>

 </tbody>

</table>

Table 1: Different memory types in the CUDA/GPU memory model

Note that the declaration syntax provided in Table 1 can only be used with <strong>fixed size types </strong>(i.e. primitives or structs).

Arguments to a kernel invocation (e.g. varibles ad &amp; len in hello&lt;&lt;&lt;1, N&gt;&gt;&gt;(ad, len) in

 hello.cu) are passed to the device using <strong>constant memory </strong>and are limited to 4KB <a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a>

For more details on Variable Memory Space Specifiers, refer to the <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#variable-memory-space-specifiers">CUDA Programming Guide </a> <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#variable-memory-space-specifiers">Section B.2</a>

Next, we will discuss details about selected CUDA memory types and their usage in an application.

<h2>Global Memory – Static &amp; Dynamically Allocated</h2>

The previous section showed that we can declare a variable in global memory using the __device__ specifier. From Table 1, we know that the variable has program scope, which means that it can be accessed by both the host and device.

<table width="601">

 <tbody>

  <tr>

   <td width="70"></td>

   <td width="531"><strong><u>Exercise 5</u></strong>Compile and run global_comm.cu. Observe how the variable result can be accessed by both the device and host (albeit indirectly). What’s the reason behind this indirect access?</td>

  </tr>

 </tbody>

</table>

However, there is a limitation when using __device__ variables. From the example in global_comm.cu, we can see that arrays need to have their sizes explicitly specified during compile time. As such, if we only know the size of an array during run-time, the __device__ variable is not a very efficient and scalable solution. Thus, we need a way to define and allocate memory dynamically.

<h3>cudaMalloc – malloc for CUDA</h3>

One of the ways to allocate memory dynamically is through the use of what CUDA calls <strong>linear arrays</strong>. You can think of it like malloc but on CUDA. One can declare a <strong>linear array </strong>by invoking cudaMalloc in either host or device code, as follows:

<strong><u>Method Signature for cudaMalloc </u></strong>cudaError_t cudaMalloc (void** devPtr, size_t size)

<ul>

 <li>devPtr – is a pointer to a pointer variable that will receive the memory address of the allocated linear array</li>

 <li>size – requested allocation size in <strong>bytes</strong></li>

</ul>

<table width="601">

 <tbody>

  <tr>

   <td width="60"></td>

   <td width="541"><strong><u>Exercise 6</u></strong>Compile and run cudaMalloc_comm.cu. Observe the following:•     The global memory “declaration”•     The host can only access the linear array through cudaMemcpy•     The device access to global memory•     Like in normal malloc, it is good practice to free your dynamically allocated memory. This is done using cudaFree.</td>

  </tr>

 </tbody>

</table>

<h2>Unified Memory</h2>

We observes in the previous section that global memory, whether static or dynamically allocated, cannot be accessed by the host directly. However, in CUDA SDK 6.0, there is a new component introduced to CUDA called <strong>Unified Memory</strong>. This defines a managed memory space that has a <strong>common memory addressing space</strong>, allowing both CPU and GPU to access them as though it is part of their memory space (especially for CPU).

The only difference between <strong>unified memory </strong>and the global memory declarations described previously, is how we allocate the memory in code. We can do the following to replace our global memory declarations to use <strong>unified memory</strong>:

<ul>

 <li>Replacing __device__ with __managed__</li>

 <li>Replacing cudaMalloc with cudaMallocManaged</li>

</ul>

<table width="601">

 <tbody>

  <tr>

   <td width="70"></td>

   <td width="531"><strong><u>Exercise 7</u></strong>Compile and run both global_comm_unified.cu and cudaMalloc_comm_unified.cu (Note that you need to tell nvcc to target architecture sm_30). Compare the code with their non-unified counterparts. What difference(s) do you observe?</td>

  </tr>

 </tbody>

</table>

<h2>Shared Memory</h2>

Another important memory type in CUDA is <strong>shared memory</strong>. Since this resides only in the device, the memory is accessible to the device faster than global memory. As such, it allows us to store intermediate values during computation. Unlike <strong>local memory</strong>, as the name suggests, <strong>shared memory </strong>is <strong>shared </strong>within the <strong>same thread block</strong>. Thus, a thread residing in a different <strong>thread block </strong>will not see the same values in the same shared memory location.

<table width="601">

 <tbody>

  <tr>

   <td width="70"></td>

   <td width="531"><strong><u>Exercise 8</u></strong>Compile and run shared_mem.cu. Observe/Ponder on the following:•     Are there any difference between shared and global memory?•     Are the results printed out differ between runs?</td>

  </tr>

 </tbody>

</table>

<h1>Part 3: Synchronization in CUDA</h1>

This final section presents synchronization constructs in CUDA.

<h2>Atomic Memory Accesses</h2>

As with any parallel program, there are times where we want CUDA threads to use a shared memory location, be it shared (within block) or global (within device) memory. However, this may lead to race conditions. Hence, there needs to be a way to ensure <strong>mutual exclusion </strong>when accessing these shared locations.

CUDA provides the programmer with a set of <strong>atomic functions</strong>, each performing a different operation on either a 32-bit or 64-bit memory location. These functions are atomic and this guarantees mutual exclusion. For details on the CUDA atomic functions available, refer to <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions">CUDA Programming Guide Section</a>

<a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions">B.12.</a>

<table width="601">

 <tbody>

  <tr>

   <td width="70"></td>

   <td width="531"><strong><u>Exercise 9</u></strong>Compile and run atomic.cu. Observe on the following:•     What are the values that are printed out? Are they consistent across different runs?•     How does the code resolve global memory access race condition?</td>

  </tr>

 </tbody>

</table>

<h2>Synchronization Constructs</h2>

Another synchronization construct that we have learnt in this course is <strong>barriers </strong>(such as MPI_Barrier). These prevent a group of processes/threads from proceeding past the barrier till all participating processes/threads reached the barrier.

CUDA provides a simple barrier construct with the __syncthreads() function. Threads in the same block wait on syncthreads until all threads of the block have reached syncthreads. Furthermore, it guarantees that accesses to global and shared memory made up to this point are visible to all threads. Please note that there is no synchronization among threads from different blocks when you use syncthreads.

<table width="601">

 <tbody>

  <tr>

   <td width="60"></td>

   <td width="541"><strong><u>Exercise 10</u></strong>Compile and run synchronise.cu. Observe/Ponder on the following:•     What is the significance of counter values printed out with/without syncthreads•     Why does the values vary when __syncthreads is used in a kernel launch containing multiple blocks?</td>

  </tr>

 </tbody>

</table>

For details on the CUDA synchronization constructs available, refer to <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#synchronization-functions">CUDA Programming Guide</a>

<a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#synchronization-functions">Section B.6.</a>

<h1>Appendices</h1>

<h2>SoC Compute Cluster</h2>

The SoC Compute Cluster has a cluster of compute nodes that are able to execute CUDA programs. The following nodes have been reserved for you:

<ul>

 <li>xgpe0 to xgpe4 – nodes with NVIDIA Titan RTX GPU</li>

 <li>xgpf0 to xgpf4 – nodes with NVIDIA Tesla T4 GPU</li>

</ul>

To use these nodes, you must enable SoC Compute Cluster from your <a href="https://mysoc.nus.edu.sg/~myacct/services.cgi">MySoC Account page.</a> You must use your SoC account to use these nodes. To connect to any of these nodes, simply SSH to them from the SoC network (for instance, through sunfire or the computers in the Lab), using your <strong>SoC account details </strong>(used to connect to sunfire) as follows:

&gt; ssh xgpe3 This will connect to the xgpe3 node of the Compute Cluster

                      • <a href="https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/hardware">List of nodes in SoC Compute Cluster</a>

<h2>GPU Memory Organization</h2>

GPUs can be in a system either as discrete (e.g. most consumer graphics cards) or integrated (e.g. Jetson TX2). Figure 4 shows the memory organization of integrated and discrete GPUs.


