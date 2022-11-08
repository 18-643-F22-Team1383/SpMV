all:
	@cp ../SpMV/src/*.h .
	@cp ../SpMV/src/*.cpp .
	@cp ../SpMV_kernels/src/*.h .
	@cp ../SpMV_kernels/src/*.cpp .
	g++ -O3 -o spmvsw *.cpp

clean:
	-rm -rf *.cpp
	-rm -rf *.h
	-rm -rf spmvsw

.PHONY: all clean
