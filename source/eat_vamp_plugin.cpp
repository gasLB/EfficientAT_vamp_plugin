#include "eat_vamp_plugin.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numbers>
#include <numeric>
#include <regex>
#include <vamp-sdk/PluginAdapter.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace
{
    constexpr size_t kBatchSize = 8;
    constexpr double kHopDuration = 2.5;

    std::vector<std::filesystem::path> const& getModelPaths()
    {
        static auto const paths = []
        {
            std::filesystem::path library;
#if defined(_WIN32)
            HMODULE module = nullptr;
            wchar_t path[MAX_PATH];
            if(GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCWSTR>(&getModelPaths), &module) && GetModuleFileNameW(module, path, MAX_PATH) > 0)
            {
                library = path;
            }
#else
            Dl_info info;
            if(dladdr(reinterpret_cast<void const*>(&getModelPaths), &info) != 0 && info.dli_fname != nullptr)
            {
                library = info.dli_fname;
            }
#endif
            std::error_code ec;
            auto const directory = std::filesystem::absolute(library, ec).parent_path().parent_path() / "Resources" / "models";
            std::vector<std::filesystem::path> models;
            for(auto const& entry : std::filesystem::directory_iterator(directory, ec))
            {
                if(entry.path().extension() == ".ts")
                {
                    models.push_back(entry.path());
                }
            }
            std::sort(models.begin(), models.end());
            return models;
        }();
        return paths;
    }

    // Blackman windowed-sinc low-pass filter that removes the frequencies above the Nyquist frequency
    std::vector<float> makeLowPassFilter(double ratio)
    {
        if(ratio <= 1.0)
        {
            return {1.0f};
        }
        auto const cutoff = 0.5 / ratio;
        auto const size = static_cast<size_t>(std::ceil(5.5 * ratio / 0.0625)) | 1;
        std::vector<double> filter(size);
        for(size_t n = 0; n < size; ++n)
        {
            auto const x = static_cast<double>(n) - static_cast<double>(size / 2);
            auto const sinc = x == 0.0 ? 2.0 * cutoff : std::sin(2.0 * std::numbers::pi * cutoff * x) / (std::numbers::pi * x);
            auto const phase = 2.0 * std::numbers::pi * static_cast<double>(n) / static_cast<double>(size - 1);
            filter[n] = sinc * (0.42 - 0.5 * std::cos(phase) + 0.08 * std::cos(2.0 * phase));
        }
        auto const sum = std::accumulate(filter.cbegin(), filter.cend(), 0.0);
        std::vector<float> result;
        for(auto const value : filter)
        {
            result.push_back(static_cast<float>(value / sum));
        }
        return result;
    }

    // Removes the samples of the buffer before the position (counted from the start of the stream)
    void dropSamples(std::vector<float>& buffer, size_t& offset, size_t position)
    {
        if(position > offset)
        {
            auto const count = std::min(position - offset, buffer.size());
            buffer.erase(buffer.begin(), std::next(buffer.begin(), static_cast<std::ptrdiff_t>(count)));
            offset += count;
        }
    }
} // namespace

EfficientAT::Plugin::Plugin(float inputSampleRate)
: Vamp::Plugin(inputSampleRate)
{
}

bool EfficientAT::Plugin::loadModel()
{
    if(mLoadedModelIndex == mModelIndex)
    {
        return true;
    }
    mLoadedModelIndex.reset();
    mClassNames.clear();

    auto const& models = getModelPaths();
    if(mModelIndex >= models.size())
    {
        std::cerr << "[EfficientAT] No model found\n";
        return false;
    }
    try
    {
        mModel = torch::jit::load(models[mModelIndex].string());
        mModel.eval();
        mModelSampleRate = static_cast<double>(mModel.run_method("sample_rate").toInt());
        mSegmentLength = static_cast<size_t>(mModel.run_method("segment_length").toInt());
        for(auto const& name : mModel.run_method("class_names").toList())
        {
            mClassNames.push_back(name.get().to<std::string>());
        }

        mDevice = torch::kCPU;
#ifdef __APPLE__
        if(torch::mps::is_available())
        {
            try
            {
                torch::NoGradGuard noGrad;
                mModel.to(torch::kMPS);
                mModel.forward({torch::zeros({1, 1, static_cast<long long>(mSegmentLength)}).to(torch::kMPS)});
                mDevice = torch::kMPS;
            }
            catch(std::exception const&)
            {
                mModel.to(torch::kCPU);
            }
        }
#endif
    }
    catch(std::exception const& e)
    {
        std::cerr << "[EfficientAT] Failed to load the model " << models[mModelIndex] << ": " << e.what() << "\n";
        mClassNames.clear();
        return false;
    }

    if(mModelSampleRate <= 0.0 || mSegmentLength == 0 || mClassNames.empty())
    {
        std::cerr << "[EfficientAT] The model " << models[mModelIndex] << " returned invalid metadata\n";
        mClassNames.clear();
        return false;
    }
    mLoadedModelIndex = mModelIndex;
    return true;
}

bool EfficientAT::Plugin::initialise(size_t channels, size_t stepSize, size_t blockSize)
{
    if(channels != 1 || stepSize != blockSize || !loadModel())
    {
        return false;
    }
    mBlockSize = blockSize;
    mHopSize = std::min(static_cast<size_t>(std::round(mModelSampleRate * kHopDuration)), mSegmentLength);
    mRatio = static_cast<double>(getInputSampleRate()) / mModelSampleRate;
    mFilter = makeLowPassFilter(mRatio);
    reset();
    return true;
}

void EfficientAT::Plugin::reset()
{
    // the first window is centered on the first hop.
    mInput.assign(mFilter.size() / 2 + 1, 0.0f);
    mInputOffset = 0;
    mNumInputSamples = 0;
    mSamples.assign((mSegmentLength - mHopSize) / 2, 0.0f);
    mSamplesOffset = 0;
    mNumSamples = 0;
    mNextFrame = 0;
    mBatch.clear();
    mBatchFrames.clear();
    mMarker.reset();
}

EfficientAT::Plugin::OutputList EfficientAT::Plugin::getOutputDescriptors() const
{
    OutputDescriptor marker;
    marker.identifier = "marker";
    marker.name = "AudioSet Tag";
    marker.description = "The highest-scoring category of the audio content";
    marker.hasFixedBinCount = true;
    marker.binCount = 0;
    marker.sampleType = OutputDescriptor::VariableSampleRate;
    marker.hasDuration = true;

    OutputDescriptor matrix;
    matrix.identifier = "matrix";
    matrix.name = "AudioSet Matrix";
    matrix.description = "Probability of each AudioSet class";
    matrix.hasFixedBinCount = true;
    matrix.binCount = mClassNames.empty() ? 527 : mClassNames.size();
    matrix.binNames = mClassNames;
    matrix.hasKnownExtents = true;
    matrix.minValue = 0.0f;
    matrix.maxValue = 1.0f;
    matrix.sampleType = OutputDescriptor::VariableSampleRate;
    matrix.hasDuration = true;

    return {marker, matrix};
}

EfficientAT::Plugin::OutputExtraList EfficientAT::Plugin::getOutputExtraDescriptors(size_t outputDescriptorIndex) const
{
    if(outputDescriptorIndex != 0)
    {
        return {};
    }
    OutputExtraDescriptor score;
    score.identifier = "score";
    score.name = "Score";
    score.description = "The highest score of the detected category over its consecutive occurrences";
    score.hasKnownExtents = true;
    score.minValue = 0.0f;
    score.maxValue = 1.0f;

    OutputExtraDescriptor occurrences;
    occurrences.identifier = "occurrences";
    occurrences.name = "Occurrences";
    occurrences.description = "The number of consecutive occurrences of the category";
    occurrences.minValue = 1.0f;
    occurrences.maxValue = 100.0f;
    occurrences.isQuantized = true;
    occurrences.quantizeStep = 1.0f;

    return {score, occurrences};
}

EfficientAT::Plugin::ParameterList EfficientAT::Plugin::getParameterDescriptors() const
{
    auto const& models = getModelPaths();
    if(models.empty())
    {
        return {};
    }
    ParameterDescriptor model;
    model.identifier = "model";
    model.name = "Model";
    model.description = "The EfficientAT model used for the analysis";
    for(auto const& path : models)
    {
        // "01_mn10_as.ts" is displayed as "mn10_as"
        model.valueNames.push_back(std::regex_replace(path.stem().string(), std::regex("^[0-9]{2}_"), ""));
    }
    model.minValue = 0.0f;
    model.maxValue = static_cast<float>(models.size() - 1);
    model.defaultValue = 0.0f;
    model.isQuantized = true;
    model.quantizeStep = 1.0f;
    return {model};
}

void EfficientAT::Plugin::setParameter(std::string paramid, float newval)
{
    auto const& models = getModelPaths();
    if(paramid == "model" && !models.empty())
    {
        mModelIndex = std::min(static_cast<size_t>(std::max(std::round(newval), 0.0f)), models.size() - 1);
    }
}

float EfficientAT::Plugin::getParameter(std::string paramid) const
{
    return paramid == "model" ? static_cast<float>(mModelIndex) : 0.0f;
}

EfficientAT::Plugin::FeatureSet EfficientAT::Plugin::process(float const* const* inputBuffers, [[maybe_unused]] Vamp::RealTime timestamp)
{
    mInput.insert(mInput.end(), inputBuffers[0], inputBuffers[0] + mBlockSize);
    mNumInputSamples += mBlockSize;

    FeatureSet fs;
    resample(false);
    analyse(fs, false);
    return fs;
}

EfficientAT::Plugin::FeatureSet EfficientAT::Plugin::getRemainingFeatures()
{
    FeatureSet fs;
    resample(true);
    analyse(fs, true);
    runModel(fs);
    if(mMarker.has_value())
    {
        fs[0].push_back(*mMarker);
        mMarker.reset();
    }
    return fs;
}

// Converts the input to the sample rate of the model: low-pass filter and 4-point Lagrange interpolation.
// The output sample n is at the position n * ratio of the input, mInput[i] contains the input sample
// i - (filter size / 2 + 1) because of the zeros added at the start.
void EfficientAT::Plugin::resample(bool isEnd)
{
    auto const filterSize = mFilter.size();
    auto numSamples = std::numeric_limits<size_t>::max();
    if(isEnd)
    {
        mInput.insert(mInput.end(), filterSize + 2, 0.0f);
        numSamples = static_cast<size_t>(std::ceil(static_cast<double>(mNumInputSamples) / mRatio));
    }

    while(mNumSamples < numSamples)
    {
        auto const time = static_cast<double>(mNumSamples) * mRatio;
        auto const position = static_cast<size_t>(time);
        if(position + filterSize + 2 >= mInputOffset + mInput.size())
        {
            break;
        }
        // The filtered input samples at position - 1, position, position + 1 and position + 2
        auto const input = std::next(mInput.cbegin(), static_cast<std::ptrdiff_t>(position - mInputOffset));
        std::array<float, 4> y;
        for(size_t i = 0; i < y.size(); ++i)
        {
            y[i] = std::inner_product(mFilter.cbegin(), mFilter.cend(), std::next(input, static_cast<std::ptrdiff_t>(i)), 0.0f);
        }
        auto const f = static_cast<float>(time - static_cast<double>(position));
        mSamples.push_back(-f * (f - 1.0f) * (f - 2.0f) / 6.0f * y[0] + (f + 1.0f) * (f - 1.0f) * (f - 2.0f) / 2.0f * y[1] - (f + 1.0f) * f * (f - 2.0f) / 2.0f * y[2] + (f + 1.0f) * f * (f - 1.0f) / 6.0f * y[3]);
        ++mNumSamples;
    }
    dropSamples(mInput, mInputOffset, static_cast<size_t>(static_cast<double>(mNumSamples) * mRatio));
}

void EfficientAT::Plugin::analyse(FeatureSet& fs, bool isEnd)
{
    if(isEnd)
    {
        mSamples.insert(mSamples.end(), mSegmentLength, 0.0f);
    }

    while(mNextFrame * mHopSize < mNumSamples && mNextFrame * mHopSize + mSegmentLength <= mSamplesOffset + mSamples.size())
    {
        auto const window = std::next(mSamples.cbegin(), static_cast<std::ptrdiff_t>(mNextFrame * mHopSize - mSamplesOffset));
        mBatch.insert(mBatch.end(), window, std::next(window, static_cast<std::ptrdiff_t>(mSegmentLength)));
        mBatchFrames.push_back(mNextFrame++);
        if(mBatchFrames.size() == kBatchSize)
        {
            runModel(fs);
        }
    }
    dropSamples(mSamples, mSamplesOffset, mNextFrame * mHopSize);
}

void EfficientAT::Plugin::runModel(FeatureSet& fs)
{
    if(mBatchFrames.empty())
    {
        return;
    }
    try
    {
        torch::NoGradGuard noGrad;
        auto const input = torch::from_blob(mBatch.data(), {static_cast<long long>(mBatchFrames.size()), 1, static_cast<long long>(mSegmentLength)}).to(mDevice);
        auto const output = mModel.forward({input}).toTuple()->elements()[0].toTensor();
        auto const scores = torch::sigmoid(output).to(torch::kCPU).contiguous();
        auto const numClasses = mClassNames.size();
        if(scores.size(-1) != static_cast<long long>(numClasses))
        {
            throw std::runtime_error("the model returned " + std::to_string(scores.size(-1)) + " classes instead of " + std::to_string(numClasses));
        }

        auto const sampleRate = static_cast<unsigned int>(mModelSampleRate);
        for(size_t i = 0; i < mBatchFrames.size(); ++i)
        {
            auto const* frameScores = scores.data_ptr<float>() + i * numClasses;
            auto const best = std::max_element(frameScores, frameScores + numClasses);
            auto const& label = mClassNames[static_cast<size_t>(best - frameScores)];
            // The last frame is clipped to the end of the audio
            auto const start = Vamp::RealTime::frame2RealTime(static_cast<long>(mBatchFrames[i] * mHopSize), sampleRate);
            auto const end = Vamp::RealTime::frame2RealTime(static_cast<long>(std::min((mBatchFrames[i] + 1) * mHopSize, mNumSamples)), sampleRate);

            Feature column;
            column.hasTimestamp = true;
            column.timestamp = start;
            column.hasDuration = true;
            column.duration = end - start;
            column.values.assign(frameScores, frameScores + numClasses);
            fs[1].push_back(column);

            // Consecutive frames with the same label are merged into a single marker
            if(mMarker.has_value() && mMarker->label == label)
            {
                mMarker->duration = end - mMarker->timestamp;
                mMarker->values[0] = std::max(mMarker->values[0], *best);
                mMarker->values[1] += 1.0f;
            }
            else
            {
                if(mMarker.has_value())
                {
                    fs[0].push_back(*mMarker);
                }
                column.label = label;
                column.values = {*best, 1.0f};
                mMarker = column;
            }
        }
    }
    catch(std::exception const& e)
    {
        std::cerr << "[EfficientAT] Failed to run the model: " << e.what() << "\n";
    }
    mBatch.clear();
    mBatchFrames.clear();
}

extern "C"
{
    VampPluginDescriptor const* vampGetPluginDescriptor(unsigned int version, unsigned int index)
    {
        static Vamp::PluginAdapter<EfficientAT::Plugin> adapter;
        return version >= 1 && index == 0 ? adapter.getDescriptor() : nullptr;
    }

    IVE_EXTERN IvePluginDescriptor const* iveGetPluginDescriptor(unsigned int version, unsigned int index)
    {
        return version >= 1 && index == 0 ? Ive::PluginAdapter::getDescriptor<EfficientAT::Plugin>() : nullptr;
    }
}
