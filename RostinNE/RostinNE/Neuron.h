#include "Connection.h"
#pragma once
namespace Core
{

	class Neuron
	{
	public:
		using Layer = std::vector<Neuron>;
		Neuron(unsigned numOfOutputs, unsigned myIndex);
		[[nodiscard]] auto getOutputVal() const noexcept ->double;
		void setOutputVal(double val) noexcept;
		void feedForward(const Layer& prevLayer) noexcept;
		void calcOutputGradients(const double targetVal) noexcept;
		void calcHiddenGradients(const Layer& nextLayer) noexcept;
		void updateInputWeights(Layer& prevLayer) noexcept;

	private:
		[[nodiscard]] auto sumDOW(const Layer& nextLayer) const noexcept ->double;

		[[nodiscard]] auto getRandomWeight() noexcept ->double;
		[[nodiscard]] auto transferFunction(const double x) noexcept ->double;
		[[nodiscard]] auto transferFunctionDerivative(const double x) noexcept ->double;

	private:

		unsigned m_myIndex{ 0 };
		double m_outputVal {-1.0 };
		double m_gradient{ -1.0 };

		double eta = { 0.15 }; //[0.0...1.0] training rate
		double alpha = { 0.5 }; //[0.0....n] momentum

		std::vector<Connection> m_outputWeights{};
	};
}
