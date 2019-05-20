#include "stdafx.h"
#include "Net.h"

namespace Core {


	Net::Net(const std::vector<unsigned>& topology)
	{
		const auto numLayers = topology.size();

		for (auto layerNum = 0u; layerNum < numLayers; ++layerNum) {
			// numOutputs of layer[i] is the numInputs of layer[i+1]
			// numOutputs of last layer is 0
			const auto numOutputs = layerNum == topology.size() - 1u ? 0 : topology[static_cast<uint64_t>(layerNum) + 1];

			auto& neuron = m_layers.emplace_back();
			auto connection = static_cast<Neuron*>(nullptr);
			neuron.reserve(topology[layerNum]);

			// We have made a new Layer, now fill it ith neurons, and
			// add a bias neuron to the layer:
			for (auto neuronNum = 0u; neuronNum <= topology[layerNum]; ++neuronNum) {
				connection = &neuron.emplace_back(numOutputs, neuronNum);
				std::cout << "Made a Neuron!" << "\n";
			}

			// Force the bias node's output value to 1.0. It's the last neuron created above
			connection == nullptr ? throw : connection->setOutputVal(1.0);

		}

	}


	void Net::feedForward(const std::vector<double>& inputVals)
	{
		// Check the num of inputVals euqal to neuronnum expect bias
		assert(inputVals.size() == m_layers[0].size() - 1);

		// Assign {latch} the input values into the input neurons
		for (size_t i = 0; i < inputVals.size(); ++i) {
			m_layers[0][i].setOutputVal(inputVals[i]);
		}

		// Forward propagate
		for (size_t layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
			std::vector<Neuron>& prevLayer = m_layers[layerNum - 1];
			for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
				m_layers[layerNum][n].feedForward(prevLayer);
			}
		}
	}


	void Net::backProp(std::vector<double>& targetVals) noexcept
	{
		auto& outputLayer = m_layers.back();
		const auto outputNeurons = outputLayer.size();
		m_error = 0.0;

		for (size_t n = 0; n < outputNeurons - 1; ++n)
		{
			const auto delta = targetVals[n] - outputLayer[n].getOutputVal();
			m_error += delta * delta;
		}
		m_error /= outputNeurons - 1;
		m_error = sqrt(m_error);

		m_recentAverageError = 
			(m_recentAverageError * m_recentAverageSmoothingFactor + m_error) 
			/ (m_recentAverageSmoothingFactor + 1.0);

		for (size_t n = 0; n < outputNeurons - 1; ++n)
		{
			outputLayer[n].calcOutputGradients(targetVals[n]);

		}
		for (size_t layerNum = m_layers.size() - 2; layerNum > 0; --layerNum)
		{
			std::vector<Neuron>& hiddenLayer = m_layers[layerNum];
			std::vector<Neuron>& nextLayer = m_layers[static_cast<uint64_t>(layerNum) + 1];
			for(auto& h : hiddenLayer)
			{
				h.calcHiddenGradients(nextLayer);
			}
		}

		for (size_t layerNum = m_layers.size() - 1; layerNum > 0; --layerNum)
		{
			std::vector<Neuron>& layer = m_layers[layerNum];
			std::vector<Neuron>& prevLayer = m_layers[layerNum - 1];

			for (size_t n = 0; n < layer.size() - 1; ++n)
			{
				layer[n].updateInputWeights(prevLayer);
			}
		}
	}
	[[nodiscard]] auto Net::getResults() const noexcept ->std::vector<double>
	{
		const auto& outputLayer = m_layers.back();
		const auto outputNeurons = outputLayer.size();
		auto resultVals{ std::vector<double> {} };	
		resultVals.reserve(outputNeurons);

		for (size_t n = 0; n < outputNeurons - 1; ++n)
		{
			resultVals.push_back(outputLayer[n].getOutputVal());
		}

		return resultVals;
	}

	[[nodiscard]] auto Net::getRecentAverageError() const noexcept ->double
	{
		 return m_recentAverageError; 
	}
}
