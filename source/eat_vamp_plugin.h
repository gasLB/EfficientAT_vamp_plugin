#pragma once

#include <IvePluginAdapter.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <vamp-sdk/Plugin.h>

#include <array>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

// Configuration Constants
constexpr int EFFICIENTAT_VAMP_PLUGIN_PLUGIN_VERSION = 1;
constexpr const char* kSampleRateMethod = "sample_rate";
constexpr const char* kSegmentLengthMethod = "segment_length";
constexpr const char* kClassNamesMethod = "class_names";
constexpr const char* kForwardMethod = "forward";
constexpr size_t kBatchSize = 8;
constexpr double kSegmentHopSizeMs = 2500.0;      // 2.5 second hop
constexpr float kMinimumEnergyThreshold = -60.0f; // Silence gating in dB

namespace EfficientAT
{
    class Plugin
    : public Vamp::Plugin
    , public Ive::PluginExtension
    {
    public:
        explicit Plugin(float inputSampleRate);
        ~Plugin() override = default;

        bool initialise(size_t channels, size_t stepSize, size_t blockSize) override;
        void reset() override;

        std::string getIdentifier() const override;
        std::string getName() const override;
        std::string getDescription() const override;
        std::string getMaker() const override;
        int getPluginVersion() const override;
        std::string getCopyright() const override;

        InputDomain getInputDomain() const override;
        size_t getPreferredBlockSize() const override;
        size_t getPreferredStepSize() const override;

        OutputList getOutputDescriptors() const override;
        ParameterList getParameterDescriptors() const override;
        void setParameter(std::string paramid, float newval) override;
        float getParameter(std::string paramid) const override;

        OutputExtraList getOutputExtraDescriptors(size_t outputDescriptorIndex) const override;

        FeatureSet process(float const* const* inputBuffers, Vamp::RealTime timestamp) override;
        FeatureSet getRemainingFeatures() override;

    private:
        bool loadModel();
        void runModel(FeatureSet& set);
        void flushPendingMarker(FeatureSet& set);
        void pushInput(float const* inputBuffer, size_t numSamples);

        class Resampler
        {
        public:
            Resampler(double sourceSampleRate, double targetSampleRate);
            std::tuple<size_t, size_t> process(size_t numInputSamples, float const* inputBuffer, size_t numOutputSamples, float* outputBuffer);
            void reset();
            void setTargetSampleRate(double sampleRate);
            double getRatio() const noexcept;
            double getLatency() const noexcept;

        private:
            void updateFilter();
            float filterSample(float sample) noexcept;

            double mSourceSampleRate;
            double mTargetSampleRate;
            double mSubSamplePos = 1.0;
            size_t mIndexBuffer = 0;
            std::array<float, 5> mLastInputSamples{};

            // Anti-aliasing low-pass FIR filter (only used when downsampling)
            std::vector<float> mFilterCoefficients;
            std::vector<float> mFilterDelayLine;
            size_t mFilterPosition = 0;
        };

        Resampler mResampler;
        torch::jit::script::Module mModel;
        torch::Device mDevice{torch::kCPU};

        size_t mBlockSize = 0;
        size_t mModelIndex = 0;
        std::optional<size_t> mLoadedModelIndex;
        double mModelSampleRate = 32000.0;
        size_t mModelSegmentLength = 0;
        size_t mHopSize = 0;
        size_t mPaddingSize = 0;
        bool mIsResampling = false;
        size_t mResamplerLatency = 0;
        size_t mBufferWritePosition = 0;
        size_t mBufferReadPosition = 0;
        size_t mNumFrameProcessed = 0;
        size_t mAudioEnd = std::numeric_limits<size_t>::max(); // In model samples, known in getRemainingFeatures()

        std::optional<Feature> mPendingMarker;
        size_t mPendingMarkerFirstFrame = 0;
        size_t mPendingMarkerLastFrame = 0;

        std::vector<std::string> mModelClassNames;
        std::vector<float> mBuffer;
        std::vector<float> mIntermediary;
        std::vector<float> mFlatInputs;
        std::vector<size_t> mFlatIndices;
    };
} // namespace EfficientAT
