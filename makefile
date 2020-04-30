.PHONY : win clean
win :
	g++ -Wall -std=c++17 src\\*.cpp -o build\\program.exe -lstdc++ -pthread
	make clean
	
clean :
	del *.obj
	del *.o
