#include "stdafx.h"
#include "Net.h"
#include "TrainingData.h"


void showVectorVals(std::string label, std::vector<double>& v)
{
	std::cout << label << " ";
	for (unsigned i = 0; i < v.size(); ++i)
	{
		std::cout << v[i] << " ";
	}
	std::cout << std::endl;
}


auto main() -> int

{


	Core::TrainingData trainData("trainingData.txt");
	//e.g., {3, 2, 1 }
	//std::vector<unsigned> topology;
	//topology.push_back(3);
	//topology.push_back(2);
	//topology.push_back(1);




	std::vector<unsigned> topology; topology;
	trainData.getTopology(topology);
	Core::Net myNet(topology);

	std::vector<double> inputVals, targetVals, resultVals;
	int trainingPass = 0;
	while (!trainData.isEof())
	{
		++trainingPass;
		std::cout << std::endl << "Pass" << trainingPass;

		// Get new input data and feed it forward:
		if (trainData.getNextInputs(inputVals) != topology[0])
			break;
		showVectorVals(": Inputs :", inputVals);
		myNet.feedForward(inputVals);

		// Collect the net's actual results:
		myNet.getResults(resultVals);
		showVectorVals("Outputs:", resultVals);

		// Train the net what the outputs should have been:
		trainData.getTargetOutputs(targetVals);
		showVectorVals("Targets:", targetVals);
		assert(targetVals.size() == topology.back());

		myNet.backProp(targetVals);

		// Report how well the training is working, average over recnet
		std::cout << "Net recent average error: "
			<< myNet.getRecentAverageError() << std::endl;
	}

	std::cout << std::endl << "Done" << std::endl;
	return 0;
}
