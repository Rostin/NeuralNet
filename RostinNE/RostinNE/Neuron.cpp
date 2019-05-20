#include "stdafx.h"
#include "Neuron.h"

namespace Core {

	Neuron::Neuron(unsigned numOfOutputs, unsigned myIndex):
		
		m_myIndex {std::move(myIndex)}
	{
		m_outputWeights.reserve(numOfOutputs);

		for (unsigned c = 0; c < numOfOutputs; ++c)
		{
			auto& w = m_outputWeights.emplace_back();
			w.weight = getRandomWeight();
		}
	}


	[[nodiscard]] constexpr auto Neuron::getOutputVal() const noexcept ->double
	{
		return m_outputVal;
	}


	void Neuron::setOutputVal(double val) noexcept
	{
		m_outputVal = std::move(val);
	}


	void Neuron::feedForward(const std::vector<Neuron>& prevLayer) noexcept
	{
		auto sum = 0.0;

		for(const auto& neuron : prevLayer)
		{
			sum += neuron.getOutputVal() * neuron.m_outputWeights[m_myIndex].weight;
		}
		m_outputVal = Neuron::transferFunction(sum);
	}
	

	void Neuron::calcOutputGradients(const double targetVal) noexcept
	{
		const auto delta = targetVal - m_outputVal;
		m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
	}


	void Neuron::calcHiddenGradients(const std::vector<Neuron>& nextLayer) noexcept
	{
		const auto dow = sumDOW(nextLayer);
		m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
	}


	void Neuron::updateInputWeights(std::vector<Neuron>& prevLayer) noexcept
	{
		for(auto& neuron : prevLayer)
		{
			const auto oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

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

	[[nodiscard]] auto Neuron::sumDOW(const std::vector<Neuron>& nextLayer) const noexcept ->double
	{
		auto sum = 0.0;
		for (size_t n = 0; n < nextLayer.size() - 1; ++n)
		{
			sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
		}
		return sum;
	}

	[[nodiscard]] auto Neuron::getRandomWeight() noexcept ->double
	{
		const auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
		const auto dis = std::uniform_real_distribution<double>(0, 1);
		auto gen = std::mt19937_64(seed);

		return dis(gen);
	}
	[[nodiscard]] auto Neuron::transferFunction(const double x) noexcept ->double
	{
		return tanh(x);
	}
	[[nodiscard]] auto Neuron::transferFunctionDerivative(const double x) noexcept ->double
	{
		return 1.0 - x*x;
	}
}
