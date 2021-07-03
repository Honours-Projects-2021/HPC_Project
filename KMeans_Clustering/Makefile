# INC=inc
# CUDA=./Cuda/
# SERIAL=./Serial/
# MPI=./MPI/
# INCFLAGS=-I$(CUDA)$(INC)
# MPIC=mpic++
# NVCC=nvcc
# cpp=g++

# all: Cuda_KMeans MPIKMeans Serial
# Cuda_KMeans: Cuda_KMeans.cu
# 	$(NVCC) $(INCFLAGS) $(CUDA)Cuda_KMeans.cu -o Cuda_KMeans
# MPIKMeans: MPIKMeans.cpp
# 			$(MPIC) $(MPI)MPIKMeans.cpp -o MPIKMeans
# Serial:	Serial.cpp
# 			$(cpp) $(SERIAL)Serial.cpp -o Serial
# clean:
# 	rm Cuda_KMeans MPIKMeans Serial

CUDA= "nvcc -I./Cuda/inc ./Cuda/Cuda_KMeans.cu -o ./Cuda/Cuda_KMeans"
MPI = "mpic++ ./MPI/MPIKMeans.cpp -o ./MPI/MPIKMeans"
Serial= "g++ ./Serial/Serial.cpp -o ./Serial/Serial"

all: Cuda_KMeans Serial MPI
Cuda_KMeans: Cuda/Cuda_KMeans.cu
		$(CUDA)
Serial: ./Serial/Serial.cpp
		$(Serial)
MPI: ./MPI/MPIKMeans.cpp
		$(MPI)
clean:
		rm Cuda_KMeans Serial MPI
