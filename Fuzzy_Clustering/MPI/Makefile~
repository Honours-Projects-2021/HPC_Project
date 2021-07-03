run: FuzzyMPI.o
	mpic++ FuzzyMPI.o -o run
    
FuzzyMPI.o: FuzzyMPI.cpp
	mpic++ -c FuzzyMPI.cpp
    
clean:
	rm *.o run
