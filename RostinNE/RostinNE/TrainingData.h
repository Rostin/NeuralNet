#pragma once
namespace Core
{
	class TrainingData
	{
	public:
		TrainingData(const std::string filename);
		[[nodiscard]] auto isEof() const noexcept ->bool;
		[[nodiscard]] auto getTargetOutputs() noexcept->std::vector<double>;
		[[nodiscard]] auto getTopology() noexcept->std::vector<unsigned>;
		[[nodiscard]] auto getNextInputs() noexcept->std::vector<double>;

	private:
		std::ifstream m_trainingDataFile;
	};
}
