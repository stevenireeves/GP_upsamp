CFLAGS = -std=c++11 -O3
ifdef WITH_PY
	CFLAGS += -DWITH_PY
else
	CFLAGS +=  -I/usr/local/opt/lapack/include -L/usr/local/opt/lapack/lib -lblas -llapacke
endif
ifdef USE_OMP
	OMPFLAGS= -fopenmp
else
	OMPFLAGS=
endif

platform =$(shell uname -s)
ifeq ($(platform),Darwin)
ifdef USE_OMP
	OMPINCLUDE=-I/usr/local/opt/libomp/include -L/usr/local/opt/libomp/lib
else
	OMPINCLUDE=
endif
CPP = clang++
interp.dylib: interp.cpp GP.o weights.o
	$(CPP) $(CFLAGS) $(OMPINCLUDE) $(OMPFLAGS)  -dynamiclib  -o interp.dylib interp.cpp GP.o weights.o

weights.o: weights.cpp weights.h
	$(CPP) $(CFLAGS) $(OMPINCLUDE) $(OMPFLAGS)  -c weights.cpp
GP.o: GP.cpp GP.h weights.h
	$(CPP) $(CFLAGS) $(OMPINCLUDE) $(OMPFLAGS)  -c GP.cpp

clean:
	rm GP.o weights.o interp.dylib

else ifeq ($(platform),Linux)
CPP = g++
interp.so: interp.cpp GP.o weights.o
	$(CPP) -fPIC -shared -o interp.so interp.cpp GP.o weights.o $(CFLAGS) $(OMPFLAGS) 
weights.o: weights.cpp weights.h
	$(CPP) -fPIC -c weights.cpp $(CFLAGS) $(OMPFLAGS) -o weights.o
GP.o: GP.cpp GP.h weights.h
	$(CPP) -fPIC -c GP.cpp $(CFLAGS) $(OMPFLAGS)
clean:
	rm GP.o weights.o interp.so
endif
