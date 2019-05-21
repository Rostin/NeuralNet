#pragma once
#include "INet.h"
namespace Core {


	class Net : public INet
	{
	public:
		Net(const std::vector<unsigned> &topology);
		virtual void feedForward(const std::vector<double> &inputVals) override;
		virtual void backProp(std::vector<double>& targetVals) noexcept override;
		[[nodiscard]] virtual auto getResults() const noexcept ->std::vector<double> override;
		[[nodiscard]] virtual auto getRecentAverageError() const noexcept ->double override;
	protected:
		std::vector<Neuron::Layer> m_layers{};
		double m_error{ -1 };
		double m_recentAverageError{ -1 };
		double m_recentAverageSmoothingFactor{ 100.0 }; // Number of training samples to average over
	};
}
