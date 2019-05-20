#include "stdafx.h"
#include "Net.h"

namespace Core {

	double Net::m_recentAverageSmoothingFactor = 100.0; // Number of training samples to average over

	Net::Net(const std::vector<unsigned>& topology)
	{
		unsigned numLayers = topology.size();
		for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
			// numOutputs of layer[i] is the numInputs of layer[i+1]
			// numOutputs of last layer is 0
			unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

			auto& blah = m_layers.emplace_back();

			blah.reserve(topology[layerNum]);

			// We have made a new Layer, now fill it ith neurons, and
			// add a bias neuron to the layer:
			for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
				blah.emplace_back(numOutputs, neuronNum);
				std::cout << "Made a Neuron!" << std::endl;
			}

			// Force the bias node's output value to 1.0. It's the last neuron created above
			m_layers.back().back().setOutputVal(1.0);
		}

	}


	void Net::feedForward(std::vector<double>& inputVals)
	{
		// Check the num of inputVals euqal to neuronnum expect bias
		assert(inputVals.size() == m_layers[0].size() - 1);

		// Assign {latch} the input values into the input neurons
		for (unsigned i = 0; i < inputVals.size(); ++i) {
			m_layers[0][i].setOutputVal(inputVals[i]);
		}

		// Forward propagate
		for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
			std::vector<Neuron>& prevLayer = m_layers[layerNum - 1];
			for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
				m_layers[layerNum][n].feedForward(prevLayer);
			}
		}


	}
	void Net::backProp(std::vector<double>& targetVals)
	{
		std::vector<Neuron>& outputLayer = m_layers.back();

		m_error = 0.0;

		for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
		{
			double delta = targetVals[n] - outputLayer[n].getOutputVal();
			m_error += delta * delta;
		}
		m_error /= outputLayer.size() - 1;
		m_error = sqrt(m_error);

		m_recentAverageError = 
			(m_recentAverageError * m_recentAverageSmoothingFactor + m_error) 
			/ (m_recentAverageSmoothingFactor + 1.0);

		for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
		{
			outputLayer[n].calcOutputGradients(targetVals[n]);

		}
		for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum)
		{
			std::vector<Neuron>& hiddenLayer = m_layers[layerNum];
			std::vector<Neuron>& nextLayer = m_layers[layerNum + 1];
			for (unsigned n = 0; n < hiddenLayer.size(); ++n)
			{
				hiddenLayer[n].calcHiddenGradients(nextLayer);
			}

		}

		for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum)
		{
			std::vector<Neuron>& layer = m_layers[layerNum];
			std::vector<Neuron>& prevLayer = m_layers[layerNum - 1];

			for (unsigned n = 0; n < layer.size() - 1; ++n)
			{
				layer[n].updateInputWeights(prevLayer);
			}
		}

	}
	void Net::getResults(std::vector<double>& resultVals) 
	{
		resultVals.clear();

		for (unsigned n = 0; n < m_layers.back().size() - 1; ++n)
		{
			resultVals.push_back(m_layers.back()[n].getOutputVal());
		}
	}

	double Net::getRecentAverageError() const
	{
		 return m_recentAverageError; 
	}
}
