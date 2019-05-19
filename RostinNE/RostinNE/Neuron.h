
#pragma once
namespace Core
{
	struct Connection{
		double weight;
		double deltaWeight;
	};

	class Neuron {
	public:
		Neuron(unsigned numOfOutputs, unsigned myIndex);
		void feedForward(std::vector<Neuron>& prevLayer);
		void setOutputVal(double val);
		double getOutputVal();
		void calcOutputGradients(double targetVal);
		void calcHiddenGradients(const std::vector<Neuron>& nextLayer);
		double sumDOW(const std::vector<Neuron>& nextLayer) const;
		void updateInputWeights(std::vector<Neuron>& prevLayer);
		// replace with mt
		static double getRandomWeight() { return rand() / double(RAND_MAX); }
		static double transferFunction(double x);
		static double transferFunctionDerivative(double x);
	private:
		unsigned m_myIndex;
		double m_outputVal;
		double m_gradient;

		static double eta; //[0.0...1.0] training rate
		static double alpha; //[0.0....n] momentum

		std::vector<Connection> m_outputWeights;
	};
}
