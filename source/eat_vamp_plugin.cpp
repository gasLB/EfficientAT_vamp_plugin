#include "eat_vamp_plugin.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
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

#ifndef NDEBUG
#define EfficientATDbg(message) std::cout << "[Class] " << message << "\n"
#define EfficientATErr(message) std::cerr << "[Class] " << message << "\n"
#else
#define EfficientATDbg(message)
#define EfficientATErr(message)
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

std::tuple<size_t, size_t> EfficientAT::Plugin::Resampler::process(size_t numInputSamples, float const* inputBuffer, size_t numOutputSamples, float* outputBuffer)
{
    // NOTE: Polynomial interpolation lacks a low-pass anti-aliasing filter.
    // If input sample rate > target sample rate (e.g., 44.1kHz -> 32kHz),
    // high frequencies may alias and degrade model classification accuracy.
    double const speedRatio = getRatio();
    size_t numGeneratedSamples = 0;
    size_t numUsedSamples = 0;
    auto subSamplePos = mSubSamplePos;
    while(numUsedSamples < numInputSamples && numGeneratedSamples < numOutputSamples)
    {
        while(subSamplePos >= 1.0 && numUsedSamples < numInputSamples)
        {
            mLastInputSamples[mIndexBuffer] = inputBuffer[numUsedSamples++];
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
        mLastInputSamples[mIndexBuffer] = inputBuffer[numUsedSamples++];
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
    std::fill(mLastInputSamples.begin(), mLastInputSamples.end(), 0.0f);
}

void EfficientAT::Plugin::Resampler::setTargetSampleRate(double sampleRate) noexcept
{
    mTargetSampleRate = sampleRate;
}

double EfficientAT::Plugin::Resampler::getRatio() const noexcept
{
    return mSourceSampleRate / mTargetSampleRate;
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
    mResampler.reset();
    mBufferWritePosition = 0;
    mBufferReadPosition = 0;
    mNumFrameProcessed = 0;
    std::fill(mBuffer.begin(), mBuffer.end(), 0.0f);
    std::fill(mIntermediary.begin(), mIntermediary.end(), 0.0f);
}

bool EfficientAT::Plugin::loadModel()
{
    auto const models = getModelPaths();
    if(mModelIndex >= models.size() || models.at(mModelIndex).empty())
    {
        EfficientATErr("Failed to retrieve the model!");
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
        EfficientATErr("Exception loading model: " << e.what());
        return false;
    }
    return (mModelSampleRate > 0.0 && mModelSegmentLength > 0 && !mModelClassNames.empty());
}

bool EfficientAT::Plugin::initialise(size_t channels, size_t stepSize, size_t blockSize)
{
    if(channels != static_cast<size_t>(1) || stepSize != blockSize)
    {
        return false;
    }
    reset();
    auto const models = getModelPaths();
    if(mModelIndex >= models.size())
    {
        EfficientATErr("Failed to retrieve the model: index out of range!");
        return false;
    }

    auto const model = models.at(mModelIndex);
    if(model.empty())
    {
        EfficientATErr("Failed to retrieve the model: empty path!");
        return false;
    }

    mBlockSize = blockSize;

    try
    {
        mDevice = getOptimalDevice();
        mModel = torch::jit::load(model.string());
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
    catch(const c10::Error& e)
    {
        EfficientATErr("TorchScript error loading model: " << e.what());
        return false;
    }
    catch(const std::exception& e)
    {
        EfficientATErr("Standard exception loading model: " << e.what());
        return false;
    }

    if(mModelSampleRate <= 0.0 || mModelSegmentLength == 0 || mModelClassNames.empty())
    {
        EfficientATErr("Model returned invalid metadata parameters!");
        return false;
    }

    mBuffer.assign(mModelSegmentLength * 2, 0.0f);
    mIntermediary.assign(mModelSegmentLength, 0.0f);
    mFlatInputs.reserve(mModelSegmentLength * kBatchSize);
    mFlatIndices.reserve(kBatchSize);
    mResampler.setTargetSampleRate(mModelSampleRate);

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
    return "Audio tagging and classification based on AudioSet (527 classes) using EfficientAT deep learning models.";
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
    return getPreferredBlockSize();
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

    // 527-Class Score Matrix
    {
        OutputDescriptor d;
        d.identifier = "matrix";
        d.name = "AudioSet Matrix";
        d.description = "Probability distribution across all 527 AudioSet classes";
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
        param.description = "The model used to generate the tokens";
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
        d.description = "The score of the detected category";
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

        auto const hopSizeSamples = static_cast<size_t>(
            static_cast<double>(mModelSampleRate) * kSegmentHopSizeMs / 1000.0);

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
            assert(false && "Model output class count mismatch");
            mFlatInputs.clear();
            mFlatIndices.clear();
            return;
        }

        auto const* frameStart = tensorOut.data_ptr<float>();
        auto& markerList = fs[0]; // Top-1 Marker
        auto& matrixList = fs[1]; // 527-class Matrix

        for(size_t index = 0; index < mFlatIndices.size(); ++index)
        {
            auto const* frameEnd = std::next(frameStart, static_cast<std::ptrdiff_t>(numClasses));
            auto const framePosition = mFlatIndices.at(index);

            auto const timestamp = Vamp::RealTime::frame2RealTime(
                framePosition * hopSizeSamples, static_cast<unsigned int>(mModelSampleRate));
            auto const duration = Vamp::RealTime::frame2RealTime(
                hopSizeSamples, static_cast<unsigned int>(mModelSampleRate));

            auto const maxIt = std::max_element(frameStart, frameEnd);
            auto const bestClassIndex = static_cast<size_t>(std::distance(frameStart, maxIt));
            auto const bestScore = *maxIt;

            Feature markerFeature;
            markerFeature.hasTimestamp = true;
            markerFeature.timestamp = timestamp;
            markerFeature.hasDuration = true;
            markerFeature.duration = duration;
            markerFeature.label = mModelClassNames.at(bestClassIndex);
            markerFeature.values = {bestScore, 1.0f};
            markerList.push_back(std::move(markerFeature));

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
    catch(const c10::Error& e)
    {
        EfficientATErr("TorchScript error during inference: " << e.what());
    }
    catch(const std::exception& e)
    {
        EfficientATErr("Exception during inference: " << e.what());
    }

    mFlatInputs.clear();
    mFlatIndices.clear();
}

EfficientAT::Plugin::FeatureSet EfficientAT::Plugin::process(float const* const* inputBuffers, [[maybe_unused]] Vamp::RealTime timestamp)
{
    auto const blockSize = mBlockSize;
    auto const* inputBuffer = inputBuffers[0];
    auto const sampleRateRatio = static_cast<double>(mModelSampleRate) / static_cast<double>(getInputSampleRate());

    if(std::abs(sampleRateRatio - 1.0) < 0.0001)
    {
        if((mBuffer.size() - mBufferWritePosition) < blockSize)
        {
            mBuffer.resize(mBufferWritePosition + blockSize, 0.0f);
        }
        std::copy(inputBuffer, inputBuffer + blockSize, std::next(mBuffer.begin(), static_cast<std::ptrdiff_t>(mBufferWritePosition)));
        mBufferWritePosition += blockSize;
    }
    else
    {
        auto const expectedSize = static_cast<size_t>(std::ceil(static_cast<double>(blockSize) * sampleRateRatio)) + 4;
        if((mBuffer.size() - mBufferWritePosition) < expectedSize)
        {
            mBuffer.resize(mBufferWritePosition + expectedSize, 0.0f);
        }
        auto const result = mResampler.process(blockSize, inputBuffer, expectedSize, mBuffer.data() + mBufferWritePosition);
        if(std::get<0>(result) != blockSize)
        {
            EfficientATErr("Resampler failed to consume all input samples! Consumed " << std::get<0>(result) << " out of " << blockSize);
        }
        mBufferWritePosition += std::get<1>(result);
    }

    FeatureSet fs;
    auto const hopSizeSamples = static_cast<size_t>(static_cast<double>(mModelSampleRate) * kSegmentHopSizeMs / 1000.0);

    while(mBufferWritePosition - mBufferReadPosition >= mModelSegmentLength)
    {
        auto const startPosition = mBufferReadPosition;
        std::copy_n(std::next(mBuffer.cbegin(), static_cast<std::ptrdiff_t>(startPosition)), mModelSegmentLength, mIntermediary.begin());

        static auto const linearThreshold = std::pow(10.0f, kMinimumEnergyThreshold / 20.0f);
        auto const isAboveThreshold = [&, this]()
        {
            size_t i = 0;
            while(i < mIntermediary.size())
            {
                auto const numSamples = std::min(mIntermediary.size() - i, hopSizeSamples);
                auto const startIt = std::next(mIntermediary.begin(), static_cast<std::ptrdiff_t>(i));
                auto const endIt = std::next(startIt, static_cast<std::ptrdiff_t>(numSamples));
                auto const sumSquares = std::accumulate(startIt, endIt, 0.0f, [](auto const sum, const auto amp)
                                                        {
                                                            return sum + amp * amp;
                                                        });
                if(std::sqrt(sumSquares / static_cast<float>(numSamples)) >= linearThreshold)
                {
                    return true;
                }
                i += numSamples;
            }
            return false;
        }();

        if(isAboveThreshold)
        {
            mFlatInputs.insert(mFlatInputs.end(), mIntermediary.cbegin(), mIntermediary.cend());
            mFlatIndices.push_back(mNumFrameProcessed);
            if(mFlatIndices.size() >= kBatchSize)
            {
                runModel(fs);
            }
        }

        mBufferReadPosition += hopSizeSamples;
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
    auto const hopSizeSamples = static_cast<size_t>(
        static_cast<double>(mModelSampleRate) * kSegmentHopSizeMs / 1000.0);

    static auto const linearThreshold = std::pow(10.0f, kMinimumEnergyThreshold / 20.0f);

    // Slide the window by hopSizeSamples until all remaining samples are consumed
    while(mBufferReadPosition < mBufferWritePosition)
    {
        auto const remainingSamples = mBufferWritePosition - mBufferReadPosition;
        auto const samplesToCopy = std::min(remainingSamples, mModelSegmentLength);

        std::fill(mIntermediary.begin(), mIntermediary.end(), 0.0f);
        std::copy_n(
            std::next(mBuffer.cbegin(), static_cast<std::ptrdiff_t>(mBufferReadPosition)),
            samplesToCopy,
            mIntermediary.begin());

        // Check energy only on the valid (non-zero-padded) samples
        auto const sumSquares = std::accumulate(
            mIntermediary.cbegin(),
            std::next(mIntermediary.cbegin(), static_cast<std::ptrdiff_t>(samplesToCopy)),
            0.0f,
            [](auto const sum, auto const amp)
            {
                return sum + amp * amp;
            });

        if(std::sqrt(sumSquares / static_cast<float>(samplesToCopy)) >= linearThreshold)
        {
            mFlatInputs.insert(mFlatInputs.end(), mIntermediary.cbegin(), mIntermediary.cend());
            mFlatIndices.push_back(mNumFrameProcessed);
            if(mFlatIndices.size() >= kBatchSize)
            {
                runModel(fs);
            }
        }

        mBufferReadPosition += hopSizeSamples;
        ++mNumFrameProcessed;
    }

    // Flush any remaining frames in the batch
    runModel(fs);

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
