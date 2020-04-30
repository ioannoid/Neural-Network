#pragma once

#include <vector>
#include <iostream>
#include <initializer_list>
#include <math.h>
#include <utility>
#include <fstream>
#include <time.h>
#include <thread>
#include <functional>
#include <future>
#include <mutex>

#include "matrix.h"

class network 
{
public:
    network();
    network(std::initializer_list<int> map);
    network(const char* fpath);

    std::pair<double,matrix> predict(const matrix& in, const matrix& actual) const;
    double train(int iterations, int batch_size, double training_rate, const std::vector<std::pair<matrix,matrix>>& training_data);

    void save(const char* fname);

    static std::pair<std::vector<std::pair<matrix,matrix>>,std::vector<std::pair<matrix,matrix>>> spiltData(const std::vector<std::vector<double>>& data, const std::vector<std::vector<double>>& labels, double training_fraction);

private:
    std::vector<matrix> weights;
    std::vector<matrix> biases;

    std::vector<matrix> delta_weights;
    std::vector<matrix> delta_biases;

    std::mutex mutex;

    void propagate(const matrix& in, const matrix& actual);
    void threadProp(const matrix& in, const matrix& actual);

    double cost(const matrix& out, const matrix& actual) const;
    matrix dcost(const matrix& out, const matrix& actual) const;
    matrix sigmoid(const matrix& mat) const;
    matrix dsigmoid(const matrix& mat) const;
};

network::network()
{
    
}

network::network(std::initializer_list<int> map)
{
    //Weights
    weights = std::vector<matrix>(map.size()-1);
    delta_weights = std::vector<matrix>(map.size()-1);
    for(size_t i = 0; i < map.size()-1; i++)
    {
        weights.at(i) = matrix::rand(*(map.begin()+1+i), *(map.begin()+i));
        delta_weights.at(i) = matrix(*(map.begin()+1+i), *(map.begin()+i));
    }

    //Biases
    biases = std::vector<matrix>(map.size()-1);
    delta_biases = std::vector<matrix>(map.size()-1);
    for(size_t i = 0; i < map.size()-1; i++)
    {
        biases.at(i) = matrix::rand(*(map.begin()+1+i), 1);
        delta_biases.at(i) = matrix(*(map.begin()+1+i), 1);
    }
}

network::network(const char* fpath)
{
    std::ifstream nnmap;
	nnmap.open(fpath);

    std::string type;
	std::string data;

    while (std::getline(nnmap, type)) {
		if (type[0] == 'w') {
			int row = stoi(type.substr(1, type.find('|') - 1));
			int length = stoi(type.substr(type.find('|') + 1));

			std::vector<std::vector<double>> vmatrix;
			std::vector<double> r;

			while (length != 0 && std::getline(nnmap, data)) {
				r.push_back(stod(data));
				row--;
				if (row == 0) {
					vmatrix.push_back(r);
					r.clear();
					row = stoi(type.substr(1, type.find('|')));
				}
				length--;
			}

			weights.push_back(matrix(vmatrix));
            delta_weights.push_back(matrix(vmatrix.size(), vmatrix.at(0).size()));
		}
		else if (type[0] == 'b') {
			int length = stoi(type.substr(1));
			std::vector<std::vector<double>> vmatrix;
			while (length != 0 && std::getline(nnmap, data)) {
				vmatrix.push_back(std::vector<double>{stod(data)});
				length--;
			}

			biases.push_back(matrix(vmatrix));
            delta_biases.push_back(matrix(vmatrix.size(), 1));
		}
    }

	nnmap.close();
}

std::pair<double, matrix> network::predict(const matrix& in, const matrix& actual) const
{
    if(in.r() != weights.at(0).c() || in.c() != 1) exit(0);

    matrix out = in;
    for(size_t i = 0; i < weights.size(); i++)
        out = sigmoid((weights.at(i) * out) + biases.at(i));

    return std::make_pair(cost(out, actual), out);
}

void network::propagate(const matrix& in, const matrix& actual)
{
    actual.print();
    printf("\n");
    std::vector<matrix> z(weights.size());
    std::vector<matrix> a(weights.size()+1);

    a.at(0) = in;
    for(size_t i = 0; i < z.size(); i++)
    {
        z.at(i) = (weights.at(i)*a.at(i)) + biases.at(i);
        a.at(i+1) = sigmoid(z.at(i));
    }

    matrix dc_daMat = actual, dz_dw, da_dz, dc_da, dc_dw, dc_db, dz_da;

    for(int i = z.size()-1; i >= 0; i--)
    {
        dz_dw = a.at(i);
        da_dz = dsigmoid(z.at(i));
        dc_da = dcost(a.at(i+1), dc_daMat);

        dc_dw = ((da_dz.hproduct(dc_da))*dz_dw.T());
        delta_weights.at(i) = delta_weights.at(i) + dc_dw;

        dc_db = da_dz.hproduct(dc_da);
        delta_biases.at(i) = delta_biases.at(i) + dc_db;

        dz_da = weights.at(i);
        dc_daMat = (dz_da.T()*(da_dz.hproduct(dc_da)));
    }

    std::cout << "Lol what\n";
}

void network::threadProp(const matrix& in, const matrix& actual)
{
    std::vector<matrix> z(weights.size());
    std::vector<matrix> a(weights.size()+1);

    a.at(0) = in;
    for(size_t i = 0; i < z.size(); i++)
    {
        z.at(i) = (weights.at(i)*a.at(i)) + biases.at(i);
        a.at(i+1) = sigmoid(z.at(i));
    }

    matrix dc_daMat = actual, dz_dw, da_dz, dc_da, dc_dw, dc_db, dz_da;

    for(int i = z.size()-1; i >= 0; i--)
    {
        dz_dw = a.at(i);
        da_dz = dsigmoid(z.at(i));
        dc_da = dcost(a.at(i+1), dc_daMat);

        dc_dw = ((da_dz.hproduct(dc_da))*dz_dw.T());
        mutex.lock();
        delta_weights.at(i) = delta_weights.at(i) + dc_dw;
        mutex.unlock();

        dc_db = da_dz.hproduct(dc_da);
        mutex.lock();
        delta_biases.at(i) = delta_biases.at(i) + dc_db;
        mutex.unlock();

        dz_da = weights.at(i);
        dc_daMat = (dz_da.T()*(da_dz.hproduct(dc_da)));
    }
}

double network::train(int iterations, int batch_size, double training_rate, const std::vector<std::pair<matrix,matrix>>& training_data)
{
    if(batch_size < 1) exit(0);

    for(int i = 0; i < iterations; i++)
    {
        int backPropsLeft = batch_size;
        std::vector<std::thread> threads;

        while(backPropsLeft != 0)
        {
            if(backPropsLeft>=10)
            {
                backPropsLeft-=10;
                for(int j = 0; j < 10; j++) 
                {
                    threads.push_back(std::thread([&training_data, this]{
                            srand (time(NULL));
                            int ri = std::rand() % training_data.size();
                            threadProp(training_data.at(ri).first, training_data.at(ri).second);
                        }));
                }
                for(std::thread& t : threads) t.join();
            }
            else
            {
                for(int j = 0; j < backPropsLeft; j++) 
                {
                    threads.push_back(std::thread([&training_data, this]{
                            srand (time(NULL));
                            int ri = std::rand() % training_data.size();
                            threadProp(training_data.at(ri).first, training_data.at(ri).second);
                        }));
                }
                for(std::thread& t : threads) t.join();

                backPropsLeft = 0;
            }
            threads.clear();
        }

        if(i%10 == 0)
        {
            int ri = std::rand() % training_data.size();
            double cost = predict(training_data.at(ri).first, training_data.at(ri).second).first;
            printf("Cost: %.7f\n", cost);
        }

        for(size_t w = 0; w < weights.size(); w++) 
        {
            weights.at(w) = weights.at(w) - (training_rate*(delta_weights.at(w)/batch_size));
            delta_weights.at(w) = matrix(delta_weights.at(w).r(), delta_weights.at(w).c());
        }
        for(size_t b = 0; b < biases.size(); b++) 
        {
            biases.at(b) = biases.at(b) - (training_rate*(delta_biases.at(b)/batch_size));
            delta_biases.at(b) = matrix(delta_biases.at(b).r(), 1);
        }
    }

    return 0;
}

double network::cost(const matrix& out, const matrix& actual) const
{
    if(out.r() != actual.r() || out.c() != 1 || actual.c() != 1)
    {
        std::cout << "Cost Function Error" << std::endl;
        exit(0);
    }

    double cost = 0;
    for(int i = 0; i < out.r(); i++)
        cost += std::pow(out.at(i, 0) - actual.at(i, 0), 2);
    
    return cost;
}

matrix network::dcost(const matrix& out, const matrix& actual) const
{
    if(out.r() != actual.r() || out.c() != 1 || actual.c() != 1)
    {
        std::cout << "dCost Function Error" << std::endl;
        exit(0);
    }

    return out.elementOp([&actual](auto x, auto i){ return 2*(x-actual.at(i));});
}

matrix network::sigmoid(const matrix& mat) const
{
    return mat.elementOp([](auto x, auto i){ return 1/(1+std::exp(-x)); });
}

matrix network::dsigmoid(const matrix& mat) const
{
    return mat.elementOp([](auto x, auto i){ return std::exp(x)/std::pow(1+std::exp(x),2); });
}

void network::save(const char* fname)
{
    std::ofstream nnmap;
	nnmap.open(fname);
    nnmap.precision(16);

	for (size_t m = 0; m < weights.size(); m++) {
		nnmap << "w" << weights.at(m).c() << "|" << weights.at(m).r() * weights.at(m).c() << "\n";
        weights.at(m).foreach([&nnmap](const double& x){ nnmap << x << "\n"; });
	}

	for (size_t m = 0; m < biases.size(); m++) {
		nnmap << "b" << biases.at(m).r() << "\n";
        biases.at(m).foreach([&nnmap](const double& x){ nnmap << x << "\n"; });
	}
	
	nnmap.close();
}

std::pair<std::vector<std::pair<matrix,matrix>>,std::vector<std::pair<matrix,matrix>>> network::spiltData(const std::vector<std::vector<double>>& data, const std::vector<std::vector<double>>& labels, double training_fraction) 
{
    if(data.size() != labels.size() || training_fraction == 0) exit(0);

    std::vector<std::pair<matrix,matrix>> training_data(data.size()*training_fraction);
    std::vector<std::pair<matrix,matrix>> testing_data(data.size()-training_data.size());

    for(size_t i = 0; i < training_data.size(); i++)
        training_data.at(i) = std::make_pair(matrix(data.at(i)), matrix(labels.at(i)));
    
    for(size_t i = 0; i < testing_data.size(); i++)
        testing_data.at(i) = std::make_pair(matrix(data.at(i)), matrix(labels.at(i)));


    return std::make_pair(training_data, testing_data);
}