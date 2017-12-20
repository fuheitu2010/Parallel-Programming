CXX=clang++ -m64
CXXFLAGS=-Iobjs/ -Icommon -O2 -Wall -g -mavx
ISPC=ispc
ISPCFLAGS=-O2 --arch=x86-64 --target=avx2-i32x16
TASKSYS_SOURCE=tasksys.cpp

default: sqrt

.PHONY: dirs clean
.PRECIOUS: objs/sqrt.h

dirs:
	mkdir -p objs/

clean:
	rm -rf objs *~ sqrt

sqrt: dirs  objs/sqrt.o objs/sqrt_ispc.o objs/tasksys.o objs/sqrt_avx.o
	$(CXX) $(CXXFLAGS) -o $@ objs/sqrt.o objs/sqrt_ispc.o objs/tasksys.o objs/sqrt_avx.o

objs/sqrt.o: sqrt.cpp objs/sqrt_ispc.h 
	$(CXX) $(CXXFLAGS) -c -o $@ $<

objs/sqrt_avx.o: sqrt_avx.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

objs/tasksys.o: $(TASKSYS_SOURCE)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

objs/%_ispc.h objs/%_ispc.o: %.ispc
	$(ISPC) $(ISPCFLAGS) $< -o objs/$*_ispc.o -h objs/$*_ispc.h

