#include "Connection.h"
#pragma once
namespace Core
{
	class Neuron {
	public:
		Neuron(unsigned numOfOutputs, unsigned myIndex);
		[[nodiscard]] constexpr auto getOutputVal() const noexcept ->double;
		void setOutputVal(double val) noexcept;
		void feedForward(const std::vector<Neuron>& prevLayer) noexcept;
		void calcOutputGradients(const double targetVal) noexcept;
		void calcHiddenGradients(const std::vector<Neuron>& nextLayer) noexcept;
		void updateInputWeights(std::vector<Neuron>& prevLayer) noexcept;

	private:
		[[nodiscard]] auto sumDOW(const std::vector<Neuron>& nextLayer) const noexcept ->double;

		[[nodiscard]] static auto getRandomWeight() noexcept ->double;
		[[nodiscard]] static auto transferFunction(const double x) noexcept ->double;
		[[nodiscard]] static auto transferFunctionDerivative(const double x) noexcept ->double;

	private:

		unsigned m_myIndex = 0;
		double m_outputVal = -1.0;
		double m_gradient = -1.0;

		double eta = 0.15; //[0.0...1.0] training rate
		double alpha = 0.5; //[0.0....n] momentum

		std::vector<Connection> m_outputWeights;
	};
}
