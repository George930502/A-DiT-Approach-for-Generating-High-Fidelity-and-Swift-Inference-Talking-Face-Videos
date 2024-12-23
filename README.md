# A-DiT-Approach-for-Generating-High-Fidelity-and-Swift-Inference-Talking-Face-Videos
In this study, we present a novel diffusion transformer (DiT)-based approach to generate realistic talking face videos from a single portrait image, using either text or audio input. Our method addresses the challenge of preserving detailed facial expressions, including precise lip synchronization, subtle facial nuances, natural head movements, eye gaze, and blinking, through a custom face encoder designed to extract features from distinct facial components. The DiT model captures temporal dynamics and effectively manages complex sequences, ensuring scalability and computational efficiency throughout the video generation process. Our objective is to achieve state-of-the-art performance in realism and detail preservation. Following this, we aim to compress the model to enable swift-inference while maintaining high-fidelity output, facilitating its integration into applications such as psychological counseling systems and human-AI interactive systems.

## Pretrained Headpose Estimator (HopeNet)
[300W-LP, alpha 1, robust to image quality](https://drive.google.com/file/d/1m25PrSE7g9D2q2XJVMR6IA7RaCvWSzCR/view)  
Put ```hopenet_robust_alpha1.pkl``` under the pretrained folder
```plaintext
pretrained/
    hopenet_robust_alpha1.pkl
