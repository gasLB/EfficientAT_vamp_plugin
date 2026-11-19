#include "eat_vamp_plugin.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <numbers>
#include <numeric>
#include <regex>
#include <vamp-sdk/PluginAdapter.h>

#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

#if defined(_MSC_VER)
#define forcedinline __forceinline
#else
#define forcedinline inline __attribute__((always_inline))
#endif

// Errors are always reported so users can see why the plugin failed to
// initialise, debug messages are only printed in debug builds.
#define EfficientATErr(message) std::cerr << "[EfficientAT] " << message << "\n"
#ifndef NDEBUG
#define EfficientATDbg(message) std::cout << "[EfficientAT] " << message << "\n"
#else
#define EfficientATDbg(message)
#endif

namespace EfficientAT
{
    static torch::Device getOptimalDevice()
    {
#ifdef __APPLE__
        if(torch::mps::is_available())
        {
            return torch::Device(torch::kMPS);
        }
#endif
        return torch::Device(torch::kCPU);
    }

#if defined(__unix__) || defined(__APPLE__)
    static std::filesystem::path getContentsDir()
    {
        static auto const dir = []() -> std::filesystem::path
        {
            try
            {
                Dl_info info;
                if(dladdr(reinterpret_cast<void const*>(&getContentsDir), &info) != 0 && info.dli_fname != nullptr)
                {
                    return std::filesystem::absolute(std::filesystem::path(info.dli_fname)).parent_path().parent_path();
                }
            }
            catch(...)
            {
            }
            return std::filesystem::current_path() / "Contents";
        }();
        return dir;
    }
#elif defined(_WIN32)
    static std::filesystem::path getContentsDir()
    {
        static auto const dir = []() -> std::filesystem::path
        {
            try
            {
                HMODULE hm = NULL;
                if(GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                          GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                      (LPCWSTR)&getContentsDir, &hm))
                {
                    wchar_t path[MAX_PATH];
                    GetModuleFileNameW(hm, path, MAX_PATH);
                    return std::filesystem::absolute(std::filesystem::path(path)).parent_path().parent_path();
                }
            }
            catch(...)
            {
            }
            return std::filesystem::current_path() / "Contents";
        }();
        return dir;
    }
#else
    static std::filesystem::path const executableDir = std::filesystem::current_path();
#endif

    static std::vector<std::filesystem::path> getModelPaths()
    {
#if defined(__unix__) || defined(__APPLE__) || defined(_WIN32)
        auto const resourcesDir = getContentsDir() / "Resources" / "models";
#else
        static auto const resourcesDir = executableDir / "Contents" / "Resources" / "models";
#endif
        std::vector<std::filesystem::path> modelPaths;
        std::error_code ec;
        if(!std::filesystem::exists(resourcesDir, ec) || ec)
        {
            return modelPaths;
        }

        for(auto const& entry : std::filesystem::directory_iterator(resourcesDir, ec))
        {
            if(entry.is_regular_file() && entry.path().extension() == ".ts")
            {
                modelPaths.push_back(entry.path());
            }
        }
        std::sort(modelPaths.begin(), modelPaths.end());
        return modelPaths;
    }
} // namespace EfficientAT

namespace ResamplerUtils
{
    template <int k>
    struct LagrangeResampleHelper
    {
        static forcedinline void calc(float& a, float b) noexcept
        {
            a *= b * (1.0f / k);
        }
    };

    template <>
    struct LagrangeResampleHelper<0>
    {
        static forcedinline void calc(float&, float) noexcept {}
    };

    template <int k>
    static float calcCoefficient(float input, float offset) noexcept
    {
        LagrangeResampleHelper<0 - k>::calc(input, -2.0f - offset);
        LagrangeResampleHelper<1 - k>::calc(input, -1.0f - offset);
        LagrangeResampleHelper<2 - k>::calc(input, 0.0f - offset);
        LagrangeResampleHelper<3 - k>::calc(input, 1.0f - offset);
        LagrangeResampleHelper<4 - k>::calc(input, 2.0f - offset);
        return input;
    }

    static float valueAtOffset(const float* inputs, float offset, int index) noexcept
    {
        auto result = 0.0f;
        result += calcCoefficient<0>(inputs[index], offset);
        index = (++index % 5);
        result += calcCoefficient<1>(inputs[index], offset);
        index = (++index % 5);
        result += calcCoefficient<2>(inputs[index], offset);
        index = (++index % 5);
        result += calcCoefficient<3>(inputs[index], offset);
        index = (++index % 5);
        result += calcCoefficient<4>(inputs[index], offset);
        return result;
    }
} // namespace ResamplerUtils

EfficientAT::Plugin::Resampler::Resampler(double sourceSampleRate, double targetSampleRate)
: mSourceSampleRate(sourceSampleRate)
, mTargetSampleRate(targetSampleRate)
{
    reset();
}

void EfficientAT::Plugin::Resampler::updateFilter()
{
    mFilterCoefficients.clear();
    mFilterDelayLine.clear();
    mFilterPosition = 0;
    if(mTargetSampleRate <= 0.0 || mSourceSampleRate <= mTargetSampleRate)
    {
        return;
    }

    // Windowed-sinc (Blackman) low-pass FIR applied before the polynomial
    // interpolation when downsampling. The cutoff is the target Nyquist
    // frequency and the transition band is 1/16 of the target sample rate
    // (e.g. 15-17 kHz for a 32 kHz target), so everything that would fold
    // back below 15 kHz is attenuated by ~75 dB. The filter introduces a
    // delay of (numTaps - 1) / 2 source samples (~1.4 ms at 48 kHz), which is
    // negligible compared to the analysis hop size.
    auto const cutoff = 0.5 * mTargetSampleRate / mSourceSampleRate;
    auto const transition = 0.0625 * mTargetSampleRate / mSourceSampleRate;
    auto const numTaps = static_cast<size_t>(std::ceil(5.5 / transition)) | static_cast<size_t>(1);
    auto const center = static_cast<double>(numTaps - 1) / 2.0;

    std::vector<double> coefficients(numTaps);
    for(size_t n = 0; n < numTaps; ++n)
    {
        auto const x = static_cast<double>(n) - center;
        auto const sinc = x == 0.0 ? 2.0 * cutoff : std::sin(2.0 * std::numbers::pi * cutoff * x) / (std::numbers::pi * x);
        auto const phase = 2.0 * std::numbers::pi * static_cast<double>(n) / static_cast<double>(numTaps - 1);
        auto const window = 0.42 - 0.5 * std::cos(phase) + 0.08 * std::cos(2.0 * phase);
        coefficients[n] = sinc * window;
    }
    auto const sum = std::accumulate(coefficients.cbegin(), coefficients.cend(), 0.0);
    mFilterCoefficients.resize(numTaps);
    std::transform(coefficients.cbegin(), coefficients.cend(), mFilterCoefficients.begin(), [&](auto const value)
                   {
                       return static_cast<float>(value / sum);
                   });
    // The delay line is duplicated so the most recent numTaps samples are
    // always contiguous in memory.
    mFilterDelayLine.assign(numTaps * 2, 0.0f);
}

float EfficientAT::Plugin::Resampler::filterSample(float sample) noexcept
{
    auto const numTaps = mFilterCoefficients.size();
    if(numTaps == 0)
    {
        return sample;
    }
    mFilterPosition = mFilterPosition == 0 ? numTaps - 1 : mFilterPosition - 1;
    mFilterDelayLine[mFilterPosition] = sample;
    mFilterDelayLine[mFilterPosition + numTaps] = sample;
    auto const historyStart = std::next(mFilterDelayLine.cbegin(), static_cast<std::ptrdiff_t>(mFilterPosition));
    return std::inner_product(mFilterCoefficients.cbegin(), mFilterCoefficients.cend(), historyStart, 0.0f);
}

std::tuple<size_t, size_t> EfficientAT::Plugin::Resampler::process(size_t numInputSamples, float const* inputBuffer, size_t numOutputSamples, float* outputBuffer)
{
    double const speedRatio = getRatio();
    size_t numGeneratedSamples = 0;
    size_t numUsedSamples = 0;
    auto subSamplePos = mSubSamplePos;
    while(numUsedSamples < numInputSamples && numGeneratedSamples < numOutputSamples)
    {
        while(subSamplePos >= 1.0 && numUsedSamples < numInputSamples)
        {
            mLastInputSamples[mIndexBuffer] = filterSample(inputBuffer[numUsedSamples++]);
            if(++mIndexBuffer == mLastInputSamples.size())
            {
                mIndexBuffer = 0;
            }
            subSamplePos -= 1.0;
        }
        if(subSamplePos < 1.0)
        {
            outputBuffer[numGeneratedSamples++] = ResamplerUtils::valueAtOffset(
                mLastInputSamples.data(),
                static_cast<float>(subSamplePos),
                static_cast<int>(mIndexBuffer));
            subSamplePos += speedRatio;
        }
    }
    while(subSamplePos >= 1.0 && numUsedSamples < numInputSamples)
    {
        mLastInputSamples[mIndexBuffer] = filterSample(inputBuffer[numUsedSamples++]);
        if(++mIndexBuffer == mLastInputSamples.size())
        {
            mIndexBuffer = 0;
        }
        subSamplePos -= 1.0;
    }
    mSubSamplePos = subSamplePos;
    return std::make_tuple(numUsedSamples, numGeneratedSamples);
}

void EfficientAT::Plugin::Resampler::reset()
{
    mIndexBuffer = 0;
    mSubSamplePos = 1.0;
    mFilterPosition = 0;
    std::fill(mLastInputSamples.begin(), mLastInputSamples.end(), 0.0f);
    std::fill(mFilterDelayLine.begin(), mFilterDelayLine.end(), 0.0f);
}

void EfficientAT::Plugin::Resampler::setTargetSampleRate(double sampleRate)
{
    mTargetSampleRate = sampleRate;
    updateFilter();
    reset();
}

double EfficientAT::Plugin::Resampler::getRatio() const noexcept
{
    return mSourceSampleRate / mTargetSampleRate;
}

double EfficientAT::Plugin::Resampler::getLatency() const noexcept
{
    // The latency in source samples: the group delay of the linear phase FIR
    // filter plus the 2 samples of the 5-point Lagrange interpolation.
    auto const filterLatency = mFilterCoefficients.empty() ? 0.0 : static_cast<double>(mFilterCoefficients.size() - 1) / 2.0;
    return filterLatency + 2.0;
}

EfficientAT::Plugin::Plugin(float inputSampleRate)
: Vamp::Plugin(inputSampleRate)
, mResampler(inputSampleRate, inputSampleRate)
, mDevice(getOptimalDevice())
{
    loadModel();
}

void EfficientAT::Plugin::reset()
{
    mFlatInputs.clear();
    mFlatIndices.clear();
    mPendingMarker.reset();
    mResampler.reset();
    std::fill(mBuffer.begin(), mBuffer.end(), 0.0f);
    std::fill(mIntermediary.begin(), mIntermediary.end(), 0.0f);
    // The analysis windows are centered on the hops: the buffer starts with
    // zeros so the first window covers [-padding, segment - padding[ and its
    // central hop [0, hop[.
    // The resampler latency is compensated by starting the buffer earlier.
    mBufferWritePosition = std::min(mPaddingSize - std::min(mResamplerLatency, mPaddingSize), mBuffer.size());
    mBufferReadPosition = 0;
    mNumFrameProcessed = 0;
    mAudioEnd = std::numeric_limits<size_t>::max();
}

bool EfficientAT::Plugin::loadModel()
{
    if(mLoadedModelIndex.has_value() && *mLoadedModelIndex == mModelIndex)
    {
        return true;
    }
    mLoadedModelIndex.reset();

    auto const models = getModelPaths();
    if(mModelIndex >= models.size() || models.at(mModelIndex).empty())
    {
        EfficientATErr("Failed to retrieve the model at index " << mModelIndex << " (" << models.size() << " models found)");
        return false;
    }

    try
    {
        mDevice = getOptimalDevice();
        mModel = torch::jit::load(models.at(mModelIndex).string());
        mModel.eval();
        mModel.to(mDevice);

        mModelSampleRate = static_cast<double>(mModel.get_method(kSampleRateMethod)(std::vector<c10::IValue>()).to<int>());
        mModelSegmentLength = static_cast<size_t>(mModel.get_method(kSegmentLengthMethod)(std::vector<c10::IValue>()).to<int>());

        mModelClassNames.clear();
        for(const auto& classname : mModel.get_method(kClassNamesMethod)(std::vector<c10::IValue>()).toList())
        {
            mModelClassNames.emplace_back(classname.get().to<std::string>());
        }
    }
    catch(const std::exception& e)
    {
        EfficientATErr("Failed to load the model " << models.at(mModelIndex).string() << ": " << e.what());
        return false;
    }

    if(mModelSampleRate <= 0.0 || mModelSegmentLength == 0 || mModelClassNames.empty())
    {
        EfficientATErr("The model " << models.at(mModelIndex).string() << " returned invalid metadata");
        return false;
    }

    // Some exported models can't run on an accelerator (e.g. the TorchScript trace hardcodes the CPU
    // device of the mel filter bank, which fails on MPS), so a dry run validates the device and falls
    // back to the CPU if necessary.
    if(mDevice != torch::Device(torch::kCPU))
    {
        try
        {
            torch::NoGradGuard no_grad;
            auto const dummy = torch::zeros({1, 1, static_cast<long long>(mModelSegmentLength)}, torch::kFloat32).to(mDevice);
            mModel.get_method(kForwardMethod)(std::vector<torch::jit::IValue>{dummy});
        }
        catch(const std::exception& e)
        {
            EfficientATErr("The model can't run on " << mDevice << ", falling back to CPU");
            EfficientATDbg(e.what());
            try
            {
                mDevice = torch::Device(torch::kCPU);
                mModel.to(mDevice);
            }
            catch(const std::exception& e2)
            {
                EfficientATErr("Failed to move the model to CPU: " << e2.what());
                return false;
            }
        }
    }
    mLoadedModelIndex = mModelIndex;
    return true;
}

bool EfficientAT::Plugin::initialise(size_t channels, size_t stepSize, size_t blockSize)
{
    if(channels != static_cast<size_t>(1) || stepSize != blockSize)
    {
        return false;
    }
    if(!loadModel())
    {
        return false;
    }

    mBlockSize = blockSize;

    // The hop can't be larger than the segment, otherwise samples would be skipped.
    auto const hopSize = static_cast<size_t>(std::round(mModelSampleRate * kSegmentHopSizeMs / 1000.0));
    mHopSize = std::clamp(hopSize, static_cast<size_t>(1), mModelSegmentLength);
    mPaddingSize = (mModelSegmentLength - mHopSize) / 2;

    mBuffer.assign(mModelSegmentLength * 2, 0.0f);
    mIntermediary.assign(mModelSegmentLength, 0.0f);
    mFlatInputs.reserve(mModelSegmentLength * kBatchSize);
    mFlatIndices.reserve(kBatchSize);
    mResampler.setTargetSampleRate(mModelSampleRate);
    mIsResampling = std::abs(mModelSampleRate / static_cast<double>(getInputSampleRate()) - 1.0) >= 0.0001;
    mResamplerLatency = mIsResampling ? static_cast<size_t>(std::round(mResampler.getLatency() / mResampler.getRatio())) : 0;

    reset();
    return true;
}

std::string EfficientAT::Plugin::getIdentifier() const
{
    return "efficientat";
}

std::string EfficientAT::Plugin::getName() const
{
    return "EfficientAT";
}

std::string EfficientAT::Plugin::getDescription() const
{
    return "Audio tagging and classification of AudioSet sound event classes using EfficientAT deep learning models.";
}

std::string EfficientAT::Plugin::getMaker() const
{
    return "Ircam";
}

int EfficientAT::Plugin::getPluginVersion() const
{
    return EFFICIENTAT_VAMP_PLUGIN_PLUGIN_VERSION;
}

std::string EfficientAT::Plugin::getCopyright() const
{
    return "EfficientAT. Vamp plug-in by Pierre Guillot, Gaston Fontagneu at Ircam.";
}

EfficientAT::Plugin::InputDomain EfficientAT::Plugin::getInputDomain() const
{
    return TimeDomain;
}

size_t EfficientAT::Plugin::getPreferredBlockSize() const
{
    return static_cast<size_t>(256);
}

size_t EfficientAT::Plugin::getPreferredStepSize() const
{
    return static_cast<size_t>(0);
}

EfficientAT::Plugin::OutputList EfficientAT::Plugin::getOutputDescriptors() const
{
    OutputList list;

    // Top-1 Label Marker
    {
        OutputDescriptor d;
        d.identifier = "marker";
        d.name = "AudioSet Tag";
        d.description = "The highest-scoring category of the audio content";
        d.unit = "";
        d.hasFixedBinCount = true;
        d.binCount = 0;
        d.hasKnownExtents = false;
        d.minValue = 0.0f;
        d.maxValue = 0.0f;
        d.isQuantized = false;
        d.sampleType = OutputDescriptor::SampleType::VariableSampleRate;
        d.hasDuration = true;
        list.push_back(std::move(d));
    }

    // Class Score Matrix
    {
        OutputDescriptor d;
        d.identifier = "matrix";
        d.name = "AudioSet Matrix";
        d.description = "Probability of each AudioSet class";
        d.unit = "";
        d.hasFixedBinCount = true;
        d.binCount = mModelClassNames.size();
        d.binNames = mModelClassNames;
        d.hasKnownExtents = true;
        d.minValue = 0.0f;
        d.maxValue = 1.0f;
        d.isQuantized = false;
        d.sampleType = OutputDescriptor::SampleType::VariableSampleRate;
        d.hasDuration = true;
        list.push_back(std::move(d));
    }
    return list;
}

EfficientAT::Plugin::ParameterList EfficientAT::Plugin::getParameterDescriptors() const
{
    ParameterList list;
    auto const models = getModelPaths();
    if(!models.empty())
    {
        ParameterDescriptor param;
        param.identifier = "model";
        param.name = "Model";
        param.description = "The EfficientAT model used for the analysis";
        param.unit = "";
        for(auto const& model : models)
        {
            auto modelName = model.filename().replace_extension().string();
            modelName = std::regex_replace(modelName, std::regex("^[0-9]{2}_"), "");
            param.valueNames.push_back(std::move(modelName));
        }
        param.minValue = 0.0f;
        param.maxValue = static_cast<float>(models.size() - 1);
        param.defaultValue = 0.0f;
        param.isQuantized = true;
        param.quantizeStep = 1.0f;
        list.push_back(std::move(param));
    }
    return list;
}

void EfficientAT::Plugin::setParameter(std::string paramid, float newval)
{
    if(paramid == "model")
    {
        auto const modelCount = getModelPaths().size();
        auto const max = modelCount == 0 ? 0.0f : static_cast<float>(modelCount - 1);
        mModelIndex = static_cast<size_t>(std::floor(std::clamp(newval, 0.0f, max)));
    }
    else
    {
        std::cerr << "Invalid parameter : " << paramid << "\n";
    }
}

float EfficientAT::Plugin::getParameter(std::string paramid) const
{
    if(paramid == "model")
    {
        return static_cast<float>(mModelIndex);
    }
    std::cerr << "Invalid parameter : " << paramid << "\n";
    return 0.0f;
}

EfficientAT::Plugin::OutputExtraList EfficientAT::Plugin::getOutputExtraDescriptors(size_t outputDescriptorIndex) const
{
    if(outputDescriptorIndex != 0)
    {
        return {};
    }
    OutputExtraList list;
    {
        OutputExtraDescriptor d;
        d.identifier = "score";
        d.name = "Score";
        d.description = "The highest score of the detected category over its consecutive occurrences";
        d.unit = "";
        d.hasKnownExtents = true;
        d.minValue = 0.0f;
        d.maxValue = 1.0f;
        d.isQuantized = false;
        d.quantizeStep = 0.0f;
        list.push_back(std::move(d));
    }
    {
        OutputExtraDescriptor d;
        d.identifier = "occurrences";
        d.name = "Occurrences";
        d.description = "The number of consecutive occurrences of the category";
        d.unit = "";
        d.hasKnownExtents = false;
        d.minValue = 1.0f;
        d.maxValue = 100.0f;
        d.isQuantized = true;
        d.quantizeStep = 1.0f;
        list.push_back(std::move(d));
    }
    return list;
}

namespace
{
    // Returns true if the RMS level of the samples is above the silence threshold.
    bool isAboveSilenceThreshold(std::vector<float>::const_iterator begin, std::vector<float>::const_iterator end)
    {
        static auto const linearThreshold = std::pow(10.0, static_cast<double>(kMinimumEnergyThreshold) / 20.0);
        auto const numSamples = std::distance(begin, end);
        if(numSamples <= 0)
        {
            return false;
        }
        auto const sumSquares = std::accumulate(begin, end, 0.0, [](auto const sum, auto const amp)
                                                {
                                                    return sum + static_cast<double>(amp) * static_cast<double>(amp);
                                                });
        return std::sqrt(sumSquares / static_cast<double>(numSamples)) >= linearThreshold;
    }
} // namespace

void EfficientAT::Plugin::flushPendingMarker(FeatureSet& fs)
{
    if(mPendingMarker.has_value())
    {
        fs[0].push_back(std::move(*mPendingMarker));
        mPendingMarker.reset();
    }
}

void EfficientAT::Plugin::runModel(FeatureSet& fs)
{
    if(mFlatIndices.empty())
    {
        return;
    }
    if(mFlatIndices.size() * mModelSegmentLength != mFlatInputs.size())
    {
        assert(false && "Invalid input size");
        mFlatInputs.clear();
        mFlatIndices.clear();
        return;
    }

    try
    {
        torch::NoGradGuard no_grad;

        auto tensorIn = torch::from_blob(
                            mFlatInputs.data(),
                            {static_cast<long long>(mFlatIndices.size()), 1, static_cast<long long>(mModelSegmentLength)},
                            torch::kFloat32)
                            .to(mDevice);

        auto output = mModel.get_method(kForwardMethod)(std::vector<torch::jit::IValue>{tensorIn});
        auto tensorOut = output.toTuple()->elements()[0].toTensor();
        tensorOut = torch::sigmoid(tensorOut).to(torch::kCPU).contiguous();

        auto const numClasses = static_cast<size_t>(tensorOut.size(-1));
        if(numClasses != mModelClassNames.size() || numClasses == 0)
        {
            EfficientATErr("The model returned " << numClasses << " classes instead of " << mModelClassNames.size());
            mFlatInputs.clear();
            mFlatIndices.clear();
            return;
        }

        auto const sampleRate = static_cast<unsigned int>(mModelSampleRate);
        auto const* frameStart = tensorOut.data_ptr<float>();
        auto& matrixList = fs[1];

        for(size_t index = 0; index < mFlatIndices.size(); ++index)
        {
            auto const* frameEnd = std::next(frameStart, static_cast<std::ptrdiff_t>(numClasses));
            auto const framePosition = mFlatIndices.at(index);

            // The window is centered on the hop so the frame covers [position * hop, (position + 1) * hop[
            // (the last frame is clipped to the end of the audio)
            auto const frameStartSample = framePosition * mHopSize;
            auto const frameEndSample = std::min(frameStartSample + mHopSize, std::max(mAudioEnd, frameStartSample));
            auto const timestamp = Vamp::RealTime::frame2RealTime(static_cast<long>(frameStartSample), sampleRate);
            auto const duration = Vamp::RealTime::frame2RealTime(static_cast<long>(frameEndSample - frameStartSample), sampleRate);

            auto const maxIt = std::max_element(frameStart, frameEnd);
            auto const bestClassIndex = static_cast<size_t>(std::distance(frameStart, maxIt));
            auto const bestScore = *maxIt;
            auto const& label = mModelClassNames.at(bestClassIndex);

            // Consecutive frames with the same top-1 category are merged into a single marker
            if(mPendingMarker.has_value() && mPendingMarker->label == label && mPendingMarkerLastFrame + 1 == framePosition)
            {
                mPendingMarkerLastFrame = framePosition;
                auto const numFrames = mPendingMarkerLastFrame - mPendingMarkerFirstFrame + 1;
                mPendingMarker->duration = Vamp::RealTime::frame2RealTime(static_cast<long>(frameEndSample - mPendingMarkerFirstFrame * mHopSize), sampleRate);
                mPendingMarker->values[0] = std::max(mPendingMarker->values[0], bestScore);
                mPendingMarker->values[1] = static_cast<float>(numFrames);
            }
            else
            {
                flushPendingMarker(fs);
                Feature markerFeature;
                markerFeature.hasTimestamp = true;
                markerFeature.timestamp = timestamp;
                markerFeature.hasDuration = true;
                markerFeature.duration = duration;
                markerFeature.label = label;
                markerFeature.values = {bestScore, 1.0f};
                mPendingMarker = std::move(markerFeature);
                mPendingMarkerFirstFrame = framePosition;
                mPendingMarkerLastFrame = framePosition;
            }

            Feature matrixFeature;
            matrixFeature.hasTimestamp = true;
            matrixFeature.timestamp = timestamp;
            matrixFeature.hasDuration = true;
            matrixFeature.duration = duration;
            matrixFeature.values.assign(frameStart, frameEnd);
            matrixList.push_back(std::move(matrixFeature));

            frameStart = frameEnd;
        }
    }
    catch(const std::exception& e)
    {
        EfficientATErr("Exception during inference: " << e.what());
    }

    mFlatInputs.clear();
    mFlatIndices.clear();
}

void EfficientAT::Plugin::pushInput(float const* inputBuffer, size_t numSamples)
{
    if(!mIsResampling)
    {
        if((mBuffer.size() - mBufferWritePosition) < numSamples)
        {
            mBuffer.resize(mBufferWritePosition + numSamples, 0.0f);
        }
        std::copy(inputBuffer, inputBuffer + numSamples, std::next(mBuffer.begin(), static_cast<std::ptrdiff_t>(mBufferWritePosition)));
        mBufferWritePosition += numSamples;
    }
    else
    {
        auto const expectedSize = static_cast<size_t>(std::ceil(static_cast<double>(numSamples) / mResampler.getRatio())) + 4;
        if((mBuffer.size() - mBufferWritePosition) < expectedSize)
        {
            mBuffer.resize(mBufferWritePosition + expectedSize, 0.0f);
        }
        auto const result = mResampler.process(numSamples, inputBuffer, expectedSize, mBuffer.data() + mBufferWritePosition);
        if(std::get<0>(result) != numSamples)
        {
            EfficientATErr("Resampler failed to consume all input samples! Consumed " << std::get<0>(result) << " out of " << numSamples);
        }
        mBufferWritePosition += std::get<1>(result);
    }
}

EfficientAT::Plugin::FeatureSet EfficientAT::Plugin::process(float const* const* inputBuffers, [[maybe_unused]] Vamp::RealTime timestamp)
{
    pushInput(inputBuffers[0], mBlockSize);

    FeatureSet fs;
    // mHopSize <= mModelSegmentLength so the read position never exceeds the write position
    while(mBufferWritePosition - mBufferReadPosition >= mModelSegmentLength)
    {
        auto const windowStart = std::next(mBuffer.cbegin(), static_cast<std::ptrdiff_t>(mBufferReadPosition));
        // Only the central hop of the window is labelled, so only this part is used for the silence gating
        auto const hopStart = std::next(windowStart, static_cast<std::ptrdiff_t>(mPaddingSize));
        //if(isAboveSilenceThreshold(hopStart, std::next(hopStart, static_cast<std::ptrdiff_t>(mHopSize))))
        {
            mFlatInputs.insert(mFlatInputs.end(), windowStart, std::next(windowStart, static_cast<std::ptrdiff_t>(mModelSegmentLength)));
            mFlatIndices.push_back(mNumFrameProcessed);
            if(mFlatIndices.size() >= kBatchSize)
            {
                runModel(fs);
            }
        }

        mBufferReadPosition += mHopSize;
        ++mNumFrameProcessed;
    }

    auto const remaining = mBufferWritePosition - mBufferReadPosition;
    if(remaining > 0 && mBufferReadPosition > 0)
    {
        std::copy_n(std::next(mBuffer.cbegin(), static_cast<std::ptrdiff_t>(mBufferReadPosition)), remaining, mBuffer.begin());
    }
    mBufferWritePosition = remaining;
    mBufferReadPosition = 0;

    return fs;
}

EfficientAT::Plugin::FeatureSet EfficientAT::Plugin::getRemainingFeatures()
{
    FeatureSet fs;

    // Flush the samples retained by the resampler
    if(mIsResampling)
    {
        // The last mResamplerLatency samples of the audio are still in the
        // resampler, the extra samples generated by the zeros are discarded.
        auto const audioEnd = mBufferWritePosition + mResamplerLatency;
        std::vector<float> const zeros(static_cast<size_t>(std::ceil(mResampler.getLatency())) + 1, 0.0f);
        pushInput(zeros.data(), zeros.size());
        mBufferWritePosition = std::min(mBufferWritePosition, audioEnd);
    }

    // The position of the end of the audio relative to the start of the first hop
    mAudioEnd = mNumFrameProcessed * mHopSize + (mBufferWritePosition - mBufferReadPosition) - std::min(mPaddingSize, mBufferWritePosition - mBufferReadPosition);

    // Slide the window until its central hop starts after the end of the audio,
    // the end of the window is zero-padded.
    while(mBufferReadPosition + mPaddingSize < mBufferWritePosition)
    {
        auto const remainingSamples = mBufferWritePosition - mBufferReadPosition;
        auto const samplesToCopy = std::min(remainingSamples, mModelSegmentLength);

        std::fill(mIntermediary.begin(), mIntermediary.end(), 0.0f);
        std::copy_n(std::next(mBuffer.cbegin(), static_cast<std::ptrdiff_t>(mBufferReadPosition)), samplesToCopy, mIntermediary.begin());

        // Check the energy only on the valid (non-zero-padded) samples of the central hop
        auto const hopStart = std::next(mIntermediary.cbegin(), static_cast<std::ptrdiff_t>(mPaddingSize));
        auto const numValidSamples = std::min(mHopSize, remainingSamples - mPaddingSize);
        if(isAboveSilenceThreshold(hopStart, std::next(hopStart, static_cast<std::ptrdiff_t>(numValidSamples))))
        {
            mFlatInputs.insert(mFlatInputs.end(), mIntermediary.cbegin(), mIntermediary.cend());
            mFlatIndices.push_back(mNumFrameProcessed);
            if(mFlatIndices.size() >= kBatchSize)
            {
                runModel(fs);
            }
        }

        mBufferReadPosition += mHopSize;
        ++mNumFrameProcessed;
    }

    // Flush any remaining frames in the batch and the last marker
    runModel(fs);
    flushPendingMarker(fs);

    mBufferWritePosition = 0;
    mBufferReadPosition = 0;

    return fs;
}

#ifdef __cplusplus
extern "C"
{
#endif
    VampPluginDescriptor const* vampGetPluginDescriptor(unsigned int version, unsigned int index)
    {
        if(version < 1)
        {
            return nullptr;
        }
        switch(index)
        {
            case 0:
            {
                static Vamp::PluginAdapter<EfficientAT::Plugin> adapter;
                return adapter.getDescriptor();
            }
            default:
            {
                return nullptr;
            }
        }
    }

    IVE_EXTERN IvePluginDescriptor const* iveGetPluginDescriptor(unsigned int version, unsigned int index)
    {
        if(version < 1)
        {
            return nullptr;
        }
        switch(index)
        {
            case 0:
            {
                return Ive::PluginAdapter::getDescriptor<EfficientAT::Plugin>();
            }
            default:
            {
                return nullptr;
            }
        }
    }
#ifdef __cplusplus
}
#endif
