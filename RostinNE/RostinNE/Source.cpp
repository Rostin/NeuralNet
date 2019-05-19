#include "stdafx.h"
#include "Net.h"
auto main() -> int
{

	std::vector<unsigned> topology{2,2,1};
	
	Core::Net myNet(topology);
	
	
	std::vector<double> inputVals{00};
	myNet.feedForward(inputVals);
	std::vector<double> targetVals{0};
	myNet.backProp(targetVals);
	std::vector<double> resultVals{0};
	myNet.getResults(resultVals);

	for (auto a : resultVals)
	{
		std::cout << a << "\n";
	}
	return 0;
}
