#include "stdafx.h"
#include "Net.h"
#include "TrainingData.h"
#define DEBUG true




void showVectorVals(std::string label, std::vector<double>& v)
{
	std::cout << label << " ";
	for (unsigned i = 0; i < v.size(); ++i)
	{
		std::cout << v[i] << " ";
	}
	std::cout << "\n";
}


auto main() -> int
{
#ifdef DEBUG
	auto start = std::chrono::steady_clock::now();
#endif


	Core::TrainingData trainData("trainingData.txt");


	const auto topology = trainData.getTopology();
	Core::Net myNet(topology);

	std::vector<double> inputVals, targetVals, resultVals;
	int trainingPass = 0;
	while (!trainData.isEof())
	{
		++trainingPass;
		std::cout << "\n" << "Pass" << trainingPass;

		// Get new input data and feed it forward:
		if (inputVals = trainData.getNextInputs(); inputVals.size() != topology[0])
			break;
		showVectorVals(": Inputs :", inputVals);
		myNet.feedForward(inputVals);

		// Collect the net's actual results:
		resultVals = myNet.getResults();
		showVectorVals("Outputs:", resultVals);

		// Train the net what the outputs should have been:
		targetVals = trainData.getTargetOutputs();
		showVectorVals("Targets:", targetVals);
		assert(targetVals.size() == topology.back());

		myNet.backProp(targetVals);

		// Report how well the training is working, average over recnet
		std::cout << "Net recent average error: "
			<< myNet.getRecentAverageError() << "\n";
	}

#ifdef DEBUG
	auto end = std::chrono::steady_clock::now();
	std::cout << "\n" << "Done: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() <<"\n";
#else 
	std::cout << "\n" << "Done" << "\n";
#endif
	
	return 0;
}
