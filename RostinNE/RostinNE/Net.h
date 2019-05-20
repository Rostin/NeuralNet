#pragma once
#include "Neuron.h"
namespace Core {


	class Net
	{
	public:
		Net(const std::vector<unsigned> &topology);
		void feedForward(const std::vector<double> &inputVals);
		void backProp(std::vector<double>& targetVals) noexcept;
		[[nodiscard]] auto getResults() const noexcept->std::vector<double>;
		[[nodiscard]] auto getRecentAverageError() const noexcept ->double;
	private:
		std::vector<std::vector<Neuron>> m_layers;
		double m_error = -1;
		double m_recentAverageError = -1;
		double m_recentAverageSmoothingFactor = 100.0; // Number of training samples to average over

	};
}
