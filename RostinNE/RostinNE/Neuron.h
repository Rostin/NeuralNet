#include "Connection.h"
#pragma once
namespace Core
{
	//Maybe find a better way ?
	class Neuron;
	using Layer = std::vector<Neuron>;

	class Neuron {
	public:
		Neuron(unsigned numOfOutputs, unsigned myIndex);
		[[nodiscard]] constexpr auto getOutputVal() const noexcept ->double;
		void setOutputVal(double val) noexcept;
		void feedForward(const Layer& prevLayer) noexcept;
		void calcOutputGradients(const double targetVal) noexcept;
		void calcHiddenGradients(const Layer& nextLayer) noexcept;
		void updateInputWeights(Layer& prevLayer) noexcept;

	private:
		[[nodiscard]] auto sumDOW(const Layer& nextLayer) const noexcept ->double;

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
