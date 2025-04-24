// stdafx.cpp : source file that includes just the standard includes
// ddalpha.pch will be the pre-compiled header
// stdafx.obj will contain the pre-compiled type information
#include <random>
#include "stdafx.h"


int random(int x);


int random(int x){
	random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_int_distribution<> distrib(0,x-1);
 
	return distrib(gen);
}
