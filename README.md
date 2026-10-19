---
title: Manufacturing MoE Demo
emoji: ⚙️
colorFrom: gray
colorTo: blue
sdk: gradio
app_file: app.py
python_version: "3.10.13"
sdk_version: "5.49.1"
suggested_hardware: cpu-basic
license: other
license_name: polyform-noncommercial-1.0.0
license_link: https://polyformproject.org/licenses/noncommercial/1.0.0
models:
- rasitcesur/Manufacturing-MoE
short_description: Time, energy and material estimates for manufacturing
---

# Manufacturing MoE Demo

This demo provides access to the 15 manufacturing operations in [Manufacturing-MoE Normal](https://huggingface.co/rasitcesur/Manufacturing-MoE). It accepts G-code and machine profiles for conventional manufacturing, or robot and process programs for robotics and welding.

The source package is published without pretrained weights. **Predictions become available only after training and uploading the five checkpoints to the model repository.** Until then, the app displays the supported processes, example input formats and the status of each model. It does not make predictions with random weights.

For machining, the model estimates time, electricity and removed material (or filament use for FDM). For general robotics it estimates movement time and electricity. The welding models cover arc power, robot and auxiliary loads, wire or filler where applicable, and shielding gas for GMAW/TIG. Resistance spot welding uses electrode force, current pulses and electrical contact behavior.

The underlying training data is synthetic and physically modeled. Outputs have not been validated against production machines, weld quality tests or independent robot dynamics benchmarks.

**License:** [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0). Commercial use requires separate permission.

The demo runs on the default CPU hardware. If the Space hardware is changed to a dedicated GPU or ZeroGPU, model inference can use CUDA when it is available. ZeroGPU uses short, per-request GPU allocations; training is performed outside the Space.
