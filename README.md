# Parallel-Programming
This project includes two seperate parts.
The ISPC part is a data-level parallel program. It uses the newton method to compute the square root of millions of numbers parallelly.
The ISPC part program needs to be ran in the Linux environment with Intel ISPC installed. 
To run the program in ISPC, firstly use the makefile to compile the program to generate the executable file.
The ISPC program used two instruction set: the SSE(Streaming SIMD Extensions) and the AVX (Advanced Vector Extension), if you need to run the AVX version, you must have a CPU that support AVX。
The CUDA matrix part implements the parallel matrix multiplication using NVIDIA GPU. To run the propram, you need have a NVIDIA GPU installed on your computer.
