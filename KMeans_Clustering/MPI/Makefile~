
run: MPIKMeans.o
	mpic++ MPIKMeans.o -o run
    
MPIKMeans.o: MPIKMeans.cpp
	mpic++ -c MPIKMeans.cpp
    
clean:
	rm *.o run
