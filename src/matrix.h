#pragma once

#include <vector>
#include <stdexcept>
#include <cstring>
#include <random>
#include <iomanip>
#include <initializer_list>
#include <memory>
#include <math.h>
#include <functional>

class matrix 
{
public:
	matrix() {}
	matrix(const int& rows, const int& cols);
	matrix(const std::vector<std::vector<double>>& data);
	matrix(const std::initializer_list<std::initializer_list<double>>& list);
	matrix(const matrix& mat);
	matrix(const std::vector<double>& data);
	~matrix();

	static matrix rand(const int& rows, const int& cols);

	const matrix& operator=(const matrix& mat);

	matrix operator+(const matrix& mat) const;
	matrix operator-(const matrix& mat) const;
	matrix operator-() const;

	matrix operator+(const double& scalar) const;
	matrix operator*(const double& scalar) const;
	matrix operator/(const double& scalar) const;

	matrix operator*(const matrix& mat) const;
	
	friend matrix operator+(const double& scalar, const matrix& mat);
	friend matrix operator*(const double& scalar, const matrix& mat);
	friend matrix operator/(const double& scalar, const matrix& mat);

	matrix hproduct(const matrix& mat) const;
	matrix hquotient(const matrix& mat) const;
	static matrix exp(const matrix& mat);

	void foreach(std::function<void(const double&)> op) const;
	void foreach(std::function<void(const double&, const int&)> op) const;
	matrix elementOp(std::function<double(const double&)> op) const;
	matrix elementOp(std::function<double(const double&, const int&)> op) const;

	double& at(const int& r, const int& c);
	const double& at(const int& r, const int& c) const;

	double& at(const int& i);
	const double& at(const int& i) const;

	void print() const
	{
		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < cols; c++) {
				std::cout << this->at(r,c) << " ";
			}
			std::cout << std::endl;
		}
	}
	
	int r() const;
	int c() const;
	matrix T();

private:
	double* data = nullptr;
	int rows, cols;
};

matrix::matrix(const int& rows, const int& cols)
{
	this->rows = rows;
	this->cols = cols;

	data = new double[rows*cols];
	for(int i = 0; i < rows*cols; i++) data[i] = 0.0;
}

matrix::matrix(const std::vector<std::vector<double>>& data)
{
	rows = data.size();
	cols = data.at(0).size();

	this->data = new double[rows*cols];

	int i = 0;
	for (const auto& v : data)
		for (const auto& d : v) {
			this->data[i] = d;
			i++;
		}
}

matrix::matrix(const std::initializer_list<std::initializer_list<double>>& list)
{
	rows = list.size();
	cols = (*list.begin()).size();

	data = new double[rows*cols];

	int i = 0;
	for (const auto& l : list)
		for (const auto& d : l) {
			data[i] = d;
			i++;
		}
}

matrix::matrix(const matrix& mat) : rows(mat.rows), cols(mat.cols)
{
	data = new double[rows*cols];
	for (int i = 0; i < rows*cols; i++) data[i] = mat.data[i];
}

matrix::matrix(const std::vector<double>& data)
{
	rows = data.size();
	cols = 1;
	this->data = new double[rows];
	for(int i = 0; i < rows; i++) this->data[i] = data.at(i);
}

matrix::~matrix()
{
	delete[] data;
}

matrix matrix::rand(const int& rows, const int& cols)
{
	std::random_device rd; 
    std::mt19937 gen(rd()); 
	gen.seed(time(NULL));

	matrix randMat(rows, cols);
	for(int i = 0; i < rows*cols; i++)
	{
		std::normal_distribution<double> dist(0.0, 2.0);
		randMat.data[i] = dist(gen);
	}
	
	return randMat;
}

const matrix& matrix::operator=(const matrix& mat)
{
	if(&mat == this) return *this;

	rows = mat.rows;
	cols = mat.cols;

	if(data != nullptr) delete[] data;
	data = new double[rows*cols];

	for (int i = 0; i < rows*cols; i++) data[i] = mat.data[i];

	return *this;
}

matrix matrix::operator+(const matrix& mat) const
{
	if (rows == mat.rows && cols == mat.cols)
	{
		matrix sum = matrix(mat.rows, mat.cols);
		for (int r = 0; r < rows; r++)
			for(int c = 0; c < cols; c++)
				sum.at(r,c) = this->at(r,c) + mat.at(r,c);

		return sum;
	}
	else
	{
		char* error = nullptr;
		const char* msg = "Addition dimension mismatch: (%i,%i) with (%i,%i)";
		snprintf(error, strlen(msg), msg, rows, cols, mat.rows, mat.cols);

		throw std::out_of_range(error);
		exit(0);
	}
}

matrix matrix::operator-(const matrix& mat) const
{
	if (rows == mat.rows && cols == mat.cols)
	{
		matrix diff = matrix(mat.rows, mat.cols);
		for (int r = 0; r < rows; r++)
			for(int c = 0; c < cols; c++)
				diff.at(r,c) = this->at(r,c) - mat.at(r,c);

		return diff;
	}
	else
	{
		char* error = nullptr;
		const char* msg = "Difference dimension mismatch: (%i,%i) with (%i,%i)";
		snprintf(error, strlen(msg), msg, rows, cols, mat.rows, mat.cols);

		throw std::out_of_range(error);
		exit(0);
	}
}

matrix matrix::operator-() const
{
	matrix negated(rows, cols);

	negated.data = new double[rows*cols];
	for (int i = 0; i < rows*cols; i++) negated.data[i] = -data[i];

	return negated;
}

matrix matrix::operator+(const double& scalar) const
{
	matrix sum(rows, cols);
	for (int i = 0; i < rows*cols; i++) sum.data[i] = data[i] + scalar;
	return sum;
}

matrix matrix::operator*(const double& scalar) const
{
	matrix prod(rows, cols);
	for (int i = 0; i < rows*cols; i++) prod.data[i] = data[i] * scalar;
	return prod;
}

matrix matrix::operator/(const double& scalar) const
{
	matrix quot(rows, cols);
	for (int i = 0; i < rows*cols; i++) quot.data[i] = data[i] / scalar;
	return quot;
}

matrix matrix::operator*(const matrix& mat) const
{
	if (cols == mat.rows)
	{
		matrix prod(rows, mat.cols);

		for (int r = 0; r < rows; r++)
			for (int c = 0; c < mat.cols; c++)
			{
				double dotProd = 0;
				for (int i = 0; i < mat.rows; i++) 
					dotProd += this->at(r, i) * mat.at(i, c);
				prod.at(r, c) = dotProd;
			}
		return prod;
	}
	else
	{
		char* error = nullptr;
		const char* msg = "Multiplication dimension mismatch: (%i,%i) with (%i,%i)";
		snprintf(error, strlen(msg), msg, rows, cols, mat.rows, mat.cols);

		throw std::out_of_range(error);
		exit(0);
	}
}

matrix operator+(const double& scalar, const matrix& mat)
{
	matrix sum(mat.rows, mat.cols);
	for (int i = 0; i < mat.rows*mat.cols; i++) sum.data[i] = mat.data[i] + scalar;
	return sum;
}

matrix operator*(const double& scalar, const matrix& mat)
{
	matrix prod(mat.rows, mat.cols);
	for (int i = 0; i < mat.rows*mat.cols; i++) prod.data[i] = mat.data[i] * scalar;
	return prod;
}

matrix operator/(const double& scalar, const matrix& mat)
{
	matrix quot(mat.rows, mat.cols);
	for (int i = 0; i < mat.rows*mat.cols; i++) quot.data[i] = scalar / mat.data[i];
	return quot;
}

matrix matrix::hproduct(const matrix& mat) const
{
	if(mat.rows != rows || mat.cols != cols)
	{
		exit(0);
	}

	matrix prod(rows, cols);

	for(int r = 0; r < rows; r++)
		for(int c = 0; c < cols; c++)
			prod.at(r,c) = this->at(r,c) * mat.at(r,c);

	return prod;
}

matrix matrix::hquotient(const matrix& mat) const
{
	if(mat.rows != rows || mat.cols != cols)
	{
		exit(0);
	}

	matrix quot(rows, cols);

	for(int r = 0; r < rows; r++)
		for(int c = 0; c < cols; c++)
			quot.at(r,c) = this->at(r,c) / mat.at(r,c);

	return quot;
}

matrix matrix::exp(const matrix& mat)
{
	matrix exponent(mat.rows, mat.cols);
	for(int i = 0; i < mat.rows*mat.cols; i++) exponent.data[i] = std::exp(mat.data[i]);
	return exponent;
}

void matrix::foreach(std::function<void(const double&)> op) const
{
	for(int i = 0; i < rows*cols; i++) op(data[i]);
}

void matrix::foreach(std::function<void(const double&, const int&)> op) const
{
	for(int i = 0; i < rows*cols; i++) op(data[i], i);
}

matrix matrix::elementOp(std::function<double(const double&)> op) const
{
	matrix mat(rows, cols);
	for(int i = 0; i < rows*cols; i++) mat.data[i] = op(data[i]);
	return mat;
}

matrix matrix::elementOp(std::function<double(const double&, const int&)> op) const
{
	matrix mat(rows, cols);
	for(int i = 0; i < rows*cols; i++) mat.data[i] = op(data[i], i);
	return mat;
}

double& matrix::at(const int& r, const int& c)
{
	if(r >= rows || c >= cols || r < 0 || c < 0) 
	{
		std::cout << "Rows:" << r << " Cols:" << c << std::endl;
		exit(0);
	}
	return data[(c*rows) + r];
}

const double& matrix::at(const int& r, const int& c) const
{
	if(r >= rows || c >= cols || r < 0 || c < 0) 
	{
		std::cout << "Rows:" << r << " Cols:" << c << std::endl;
		exit(0);
	}
	return data[(c*rows) + r];
}

double& matrix::at(const int& i) 
{
	if(i >= rows*cols || i < 0) exit(0);
	return data[i];
}

const double& matrix::at(const int& i) const
{
	if(i >= rows*cols || i < 0) exit(0);
	return data[i];
}

int matrix::r() const
{
	return rows;
}

int matrix::c() const
{
	return cols;
}

matrix matrix::T()
{
	matrix tMat = matrix();
	tMat.rows = cols;
	tMat.cols = rows;
	tMat.data = new double[rows*cols];
	for(int r = 0; r < tMat.rows; r++)
		for(int c = 0; c < tMat.cols; c++)
			tMat.at(r,c) = this->at(c,r);
	return tMat;
}