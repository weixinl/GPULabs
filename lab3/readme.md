run on cuda3
module load cuda-10.2
nvcc -o solveeq solveeq.cu -arch=sm_60