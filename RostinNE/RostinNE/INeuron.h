#pragma once
#include <vector>
#include "Connection.h"

namespace Core
{
	class INeuron
	{
	public:
		using Layer = std::vector<INeuron>;
		//using Layer = std::vector<Neuron>;
		[[nodiscard]] virtual auto getOutputVal() const noexcept ->double = 0;
		virtual void setOutputVal(double val) noexcept = 0;
		virtual void feedForward(const Layer& prevLayer) noexcept = 0;
		virtual void calcOutputGradients(const double targetVal) noexcept = 0;
		virtual void calcHiddenGradients(const Layer& nextLayer) noexcept = 0;
		virtual void updateInputWeights(Layer& prevLayer) noexcept = 0;
	protected:
		[[nodiscard]] virtual auto sumDOW(const Layer& nextLayer) const noexcept ->double = 0 ;
		[[nodiscard]] virtual auto getRandomWeight() noexcept ->double = 0;
		[[nodiscard]] virtual auto transferFunction(const double x) noexcept ->double = 0;
		[[nodiscard]] virtual auto transferFunctionDerivative(const double x) noexcept ->double = 0;
	protected:
		unsigned m_myIndex{ 0 };
		double m_outputVal{ -1.0 };
		double m_gradient{ -1.0 };

		double eta = { 0.15 }; //[0.0...1.0] training rate
		double alpha = { 0.5 }; //[0.0....n] momentum

		std::vector<Connection> m_outputWeights{};
	};
}