#include "stdafx.h"
#include "TrainingData.h"

namespace Core {


	TrainingData::TrainingData(const std::string filename)
	{
		m_trainingDataFile.open(filename.c_str());
	}


	[[nodiscard]] auto TrainingData::isEof() const noexcept ->bool
	{
		return m_trainingDataFile.eof();
	}


	[[nodiscard]] auto TrainingData::getTargetOutputs() noexcept ->std::vector<double>
	{
		auto targetOutputVals{ std::vector<double>{} };
		auto line{ std::string("") };
		auto label{ std::string("") };

		std::getline(m_trainingDataFile, line);

		auto ss{ std::stringstream(line) };
		ss >> label;

		if (label.compare("out") != 0)
		{
			auto oneValue = 0.0;
			while (ss >> oneValue)
			{
				targetOutputVals.push_back(oneValue);
			}
		}
		return targetOutputVals;
	}


	[[nodiscard]] auto TrainingData::getTopology() noexcept ->std::vector<unsigned>
	{
		auto topology{ std::vector<unsigned>{} };
		auto line{ std::string("") };
		auto label{ std::string("") };

		std::getline(m_trainingDataFile, line);

		auto ss{ std::stringstream(line) };
		ss >> label;

		if (this->isEof() || label.compare("topology:") != 0)
		{
			abort();
		}

		while (!ss.eof())
		{
			auto n = 0u;
			ss >> n;
			topology.push_back(n);
		}
		return topology;
	}


	[[nodiscard]] auto TrainingData::getNextInputs() noexcept ->std::vector<double>
	{
		auto inputVals{ std::vector<double> {} };
		auto line{ std::string("") };
		auto label{ std::string("") };

		std::getline(m_trainingDataFile, line);
		auto ss{ std::stringstream(line) };

		ss >> label;

		if (label.compare("in:") == 0)
		{
			auto oneValue = 0.0;
			while (ss >> oneValue)
			{
				inputVals.push_back(oneValue);
			}
		}
		return inputVals;
	}
}
