#pragma once

#include <IvePluginAdapter.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <vamp-sdk/Plugin.h>

#include <optional>
#include <string>
#include <vector>

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

        std::string getIdentifier() const override { return "efficientat"; }
        std::string getName() const override { return "EfficientAT"; }
        std::string getDescription() const override { return "Audio tagging and classification of AudioSet sound event classes using EfficientAT deep learning models."; }
        std::string getMaker() const override { return "Ircam"; }
        int getPluginVersion() const override { return 1; }
        std::string getCopyright() const override { return "EfficientAT. Vamp plug-in by Pierre Guillot, Gaston Fontagneu at Ircam."; }

        InputDomain getInputDomain() const override { return TimeDomain; }
        size_t getPreferredBlockSize() const override { return 256; }
        size_t getPreferredStepSize() const override { return 0; }

        OutputList getOutputDescriptors() const override;
        ParameterList getParameterDescriptors() const override;
        void setParameter(std::string paramid, float newval) override;
        float getParameter(std::string paramid) const override;

        OutputExtraList getOutputExtraDescriptors(size_t outputDescriptorIndex) const override;

        FeatureSet process(float const* const* inputBuffers, Vamp::RealTime timestamp) override;
        FeatureSet getRemainingFeatures() override;

    private:
        bool loadModel();
        void resample(bool isEnd);
        void analyse(FeatureSet& fs, bool isEnd);
        void runModel(FeatureSet& fs);

        torch::jit::script::Module mModel;
        torch::Device mDevice{torch::kCPU};
        size_t mModelIndex = 0;
        std::optional<size_t> mLoadedModelIndex;
        std::vector<std::string> mClassNames;
        double mModelSampleRate = 0.0;
        size_t mSegmentLength = 0;
        size_t mHopSize = 0;
        size_t mBlockSize = 0;

        // Resampling from the input sample rate to the model sample rate
        double mRatio = 1.0;
        std::vector<float> mFilter;
        std::vector<float> mInput;
        size_t mInputOffset = 0;
        size_t mNumInputSamples = 0;
        std::vector<float> mSamples;
        size_t mSamplesOffset = 0;
        size_t mNumSamples = 0;

        // Analysis
        size_t mNextFrame = 0;
        std::vector<float> mBatch;
        std::vector<size_t> mBatchFrames;
        std::optional<Feature> mMarker;
    };
} // namespace EfficientAT
