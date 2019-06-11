#pragma once
#include "INeuron.h"

namespace Core
{

	class Neuron : public INeuron
	{
	public:
		using Neurons = std::vector<Neuron>;
		Neuron(unsigned numOfOutputs, unsigned myIndex);
		[[nodiscard]] auto getOutputVal() const noexcept ->double override;
		void setOutputVal(double val) noexcept override;
		void feedForward(const Layer& prevLayer) noexcept override;
		void calcOutputGradients(const double targetVal) noexcept override;
		void calcHiddenGradients(const Layer& nextLayer) noexcept override;
		void updateInputWeights(Layer& prevLayer) noexcept override;

	private:
		[[nodiscard]] auto sumDOW(const Layer& nextLayer) const noexcept ->double override;
		[[nodiscard]] auto getRandomWeight() noexcept ->double override;
		[[nodiscard]] auto transferFunction(const double x) noexcept ->double override;
		[[nodiscard]] auto transferFunctionDerivative(const double x) noexcept ->double override;
	private:
		unsigned m_myIndex{ 0 };
		double m_outputVal{ -1.0 };
		double m_gradient{ -1.0 };

		double eta = { 0.15 }; //[0.0...1.0] training rate
		double alpha = { 0.5 }; //[0.0....n] momentum

		std::vector<Connection> m_outputWeights{};
	};
}
