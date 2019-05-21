#include "stdafx.h"
#include "Neuron.h"

namespace Core {

	Neuron::Neuron(unsigned numOfOutputs, unsigned myIndex):
		
		m_myIndex {std::move(myIndex)}
	{
		m_outputWeights.reserve(numOfOutputs);

		for (auto c{ 0u }; c < numOfOutputs; ++c)
		{
			auto& w{ m_outputWeights.emplace_back() };
			w.weight = getRandomWeight();
		}
	}


	[[nodiscard]] auto Neuron::getOutputVal() const noexcept ->double
	{
		return m_outputVal;
	}


	void Neuron::setOutputVal(double val) noexcept
	{
		m_outputVal = std::move(val);
	}


	void Neuron::feedForward(const Layer& prevLayer) noexcept
	{
		const auto sum {
			std::accumulate(prevLayer.cbegin(), prevLayer.cend(), .0,
			[&](auto sum, auto & pLayer)
			{
				return sum + pLayer.getOutputVal() * pLayer.m_outputWeights[m_myIndex].weight;
			})
		};

		m_outputVal = transferFunction(sum);
	}
	

	void Neuron::calcOutputGradients(const double targetVal) noexcept
	{
		const auto delta{ targetVal - m_outputVal };
		m_gradient = delta * transferFunctionDerivative(m_outputVal);
	}


	void Neuron::calcHiddenGradients(const Layer& nextLayer) noexcept
	{
		const auto dow{ sumDOW(nextLayer) };
		m_gradient = dow * transferFunctionDerivative(m_outputVal);
	}


	void Neuron::updateInputWeights(Layer& prevLayer) noexcept
	{

		for(auto& neuron : prevLayer)
		{
			const auto oldDeltaWeight{ neuron.m_outputWeights[m_myIndex].deltaWeight };

			const auto newDeltaWeight =
				eta
				* neuron.getOutputVal()
				* m_gradient
				+ alpha
				* oldDeltaWeight;
			neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
			neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
		}
	}

	[[nodiscard]] auto Neuron::sumDOW(const Layer& nextLayer) const noexcept ->double
	{
		auto idx{ size_t{0} };
		return std::accumulate(std::cbegin(m_outputWeights), std::cend(m_outputWeights), .0,
			[&](auto sum, auto & weights)
			{
				return sum + (weights.weight * nextLayer[idx++].m_gradient);
			});
	}

	[[nodiscard]] auto Neuron::getRandomWeight() noexcept ->double
	{
		const auto seed{ std::chrono::high_resolution_clock::now().time_since_epoch().count() };
		const auto dis{ std::uniform_real_distribution<double>(0, 1) };
		auto gen{ std::mt19937_64(seed) };

		return dis(gen);
	}
	[[nodiscard]] auto Neuron::transferFunction(const double x) noexcept ->double
	{
		return tanh(x);
	}
	[[nodiscard]] auto Neuron::transferFunctionDerivative(const double x) noexcept ->double
	{
		return 1.0 - std::pow(x, 2);
	}
}
