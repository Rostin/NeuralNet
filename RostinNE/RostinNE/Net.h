#pragma once
#include "Neuron.h"
namespace Core {


	class Net
	{
	public:
		Net(const std::vector<unsigned> &topology);
		void feedForward(std::vector<double> &inputVals);
		void backProp(std::vector<double>& targetVals);
		void getResults(std::vector<double>& resultVals);
		double getRecentAverageError() const;
	private:
		std::vector<std::vector<Neuron>> m_layers;
		double m_error;
		double m_recentAverageError;
		static double m_recentAverageSmoothingFactor;

	};
}
