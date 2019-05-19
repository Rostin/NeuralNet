#include "stdafx.h"
#include "Neuron.h"

namespace Core {

	Neuron::Neuron(unsigned numOfOutputs, unsigned myIndex)
	{
		for (unsigned c = 0; c < numOfOutputs; ++c)
		{
			m_outputWeights.push_back(Connection());
			//Create connection a class to initialize weight with a random number
			m_outputWeights.back().weight = getRandomWeight();
		}
		m_myIndex = myIndex;
	}

	double Neuron::eta = 0.15;
	double Neuron::alpha = 0.5;

	void Neuron::feedForward(std::vector<Neuron>& prevLayer)
	{
		double sum = 0.0;

		for (unsigned n = 0; n < prevLayer.size(); ++n)
		{
			sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
		}
		m_outputVal = Neuron::transferFunction(sum);
	}
	void Neuron::setOutputVal(double val)
	{
		m_outputVal = val;
	}
	double Neuron::getOutputVal()
	{
		return m_outputVal;
	}
	void Neuron::calcOutputGradients(double targetVal)
	{
		double delta = targetVal - m_outputVal;
		m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
	}
	void Neuron::calcHiddenGradients(const std::vector<Neuron>& nextLayer)
	{
		double dow = sumDOW(nextLayer);
		m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
	}
	double Neuron::sumDOW(const std::vector<Neuron>& nextLayer) const
	{
		double sum = 0.0;
		for (unsigned n = 0; n < nextLayer.size() - 1; ++n)
		{
			sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
		}
		return sum;
	}
	void Neuron::updateInputWeights(std::vector<Neuron>& prevLayer)
	{
		for (unsigned n = 0; n < prevLayer.size(); ++n)
		{
			Neuron& neuron = prevLayer[n];
			double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

			double newDeltaWeight =
				eta
				* neuron.getOutputVal()
				* m_gradient
				+ alpha
				* oldDeltaWeight;
			neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
			neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
		}
	}
	double Neuron::transferFunction(double x)
	{
		return tanh(x);
	}
	double Neuron::transferFunctionDerivative(double x)
	{
		return 1.0 - x*x;
	}
}
