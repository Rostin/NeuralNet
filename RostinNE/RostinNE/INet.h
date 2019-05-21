#pragma once
#include "Neuron.h"
#include <vector>

namespace Core
{
	class INet 
	{
	public:
		virtual void feedForward(const std::vector<double>& inputVals) = 0;
		virtual void backProp(std::vector<double>& targetVals) noexcept = 0;
		[[nodiscard]] virtual auto getResults() const noexcept->std::vector<double> = 0;
		[[nodiscard]] virtual auto getRecentAverageError() const noexcept ->double = 0;
	};
}