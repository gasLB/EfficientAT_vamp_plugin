## Notes

-> AI model is serialized into .onnx (ONNX Runtime) or .pt (TorchScript/LibTorch)
    -> ONXX is easier to use than LibTorch and no ABI problems

-> Models trained on AudioSet almost all expect 16 kHz mono audio, and many expect a Log-Mel Spectrogram as input rather than raw audio waveforms.
    -> bake the spectrogram directly into the model graph (e.g., using torchaudio.transforms.MelSpectrogram

### Compile

-> First compile by hand or small CMakeLists.txt
```
g++ -O2 -fPIC -shared MyPlugin.cpp plugins.cpp -lvamp-sdk -o my-ai-plugin.so
```

