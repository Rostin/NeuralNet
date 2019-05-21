#include "stdafx.h"
#include "Defines.h"
#include "Net.h"
#include "TrainingData.h"

void showVectorVals(std::string label, std::vector<double>& v)
{
	std::cout << label << " ";
	for (auto val : v)
	{
		std::cout << val << " ";
	}
	std::cout << "\n";
}


auto main() -> int
{
#if TIMINGS
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
#if FORCE_DEBUG
		std::cout << "\n" << "Pass" << trainingPass;
#endif

		// Get new input data and feed it forward:
		if (inputVals = trainData.getNextInputs(); inputVals.size() != topology[0])
			break;
#if FORCE_DEBUG
		showVectorVals(": Inputs :", inputVals);
#endif
		myNet.feedForward(inputVals);

		// Collect the net's actual results:
		resultVals = myNet.getResults();
#if FORCE_DEBUG
		showVectorVals("Outputs:", resultVals);
#endif

		// Train the net what the outputs should have been:
		targetVals = trainData.getTargetOutputs();
#if FORCE_DEBUG
		showVectorVals("Targets:", targetVals);
#endif
		assert(targetVals.size() == topology.back());

		myNet.backProp(targetVals);

		// Report how well the training is working, average over recnet
#if FORCE_DEBUG
		std::cout << "Net recent average error: "
			<< myNet.getRecentAverageError() << "\n";
#endif
	}

	std::cout << "Net recent average error: "
		<< myNet.getRecentAverageError() << "\n";

#if TIMINGS
	auto end = std::chrono::steady_clock::now();
	std::cout << "\n" << "Done: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() <<"\n";
#else 
	std::cout << "\n" << "Done" << "\n";
#endif
	
	return 0;
}
