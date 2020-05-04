#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <utility>

#include "matrix.h"
#include "network.h"

using namespace std;

char* loadFile(const char* filename);
vector<vector<double>> processImages(const char* filename);
vector<vector<double>> processLabels(const char* filename);
int filesize(const char* filename);

int main() 
{
	auto data = network::spiltData(processImages("train-images"), processLabels("train-labels"), 5.0/6.0);

	printf("Data Loaded\n\n");
	//{784, 500, 150, 10}

	network nn = network({784, 500, 150, 10});//
	nn.train(600, 100, 4.5, data.first);
	// auto output = nn.predict(data.second.at(0).first, data.second.at(0).second);
	// printf("Cost: %.5f\n", output.first);
	// printf("\nPredicted:\n");
	// output.second.print();
	// printf("\nActual:\n");
	// data.second.at(0).second.print();
	nn.save("thirdnnmap.nn");

	cout << "Press enter to close...";
	getchar(); 

	return 0;
}

char* loadFile(const char* filename) {
	char* imagebytes;
	int isize;

	ifstream imagedata;
	imagedata.open(filename, ios::binary);
	if (!imagedata) {
		throw std::out_of_range("Error: File not found.");
	}

	isize = filesize(filename);
	imagebytes = new char[isize];

	imagedata.read(imagebytes, isize);

	imagedata.close();
	return imagebytes;
}

vector<vector<double>> processImages(const char* filename) {
	//Load data from file into 1d array
	char* imgbytes = loadFile(filename);

	vector<vector<double>> images;
	vector<double> image;
	int isize = filesize(filename);
	
	int length = 0;
	double pixel = 0;

	//Process data from imgbytes array into 2d vector
	//Each element in first dimension represents an image, while each element in second dimension
	//represents a list of its pixeldata
	for (int i = 16; i < isize; i++) {
		pixel = ((uint8_t) imgbytes[i])/255.0;
		image.push_back(pixel);
		length++;
		if (length == 28 * 28) {
			images.push_back(image);
			image.clear();
			length = 0;
		}
	}

	delete[] imgbytes;

	return images;
}

vector<vector<double>> processLabels(const char* filename) {
	//Load data from file into 1d array
	char* lblbytes = loadFile(filename);

	vector<vector<double>> labels;
	int lsize = filesize(filename);

	int cur = 0;

	//Process data from lblbytes into appropriately formatted 1d vector
	for (int i = 8; i < lsize; i++) {
		cur = (int) (uint8_t) lblbytes[i];
		std::vector<double> fmtvector(10);
		fmtvector.at(cur) = 1;
		labels.push_back(fmtvector);
	}

	delete[] lblbytes;

	return labels;
}

int filesize(const char* filename) {
	ifstream file;
	file.open(filename, ios::ate | ios::binary);

	if (!file) {
		throw std::out_of_range("Error: File not found.");
	}

	size_t size = file.tellg();

	//Convert size_t to int
	int finalsize = 0;
	for (size_t i = 0; i < size; i++) {
		finalsize += 1;
	}

	return finalsize;
}