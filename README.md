# BSAtlas: A Multimodal LLM for Medical Image Analysis and Conversational AI.(A Vision-Language Model for Medical Image Captioning)

[![Hugging Face Hub](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Card-blue)](https://huggingface.co/Azzahrae96/BSAtlas)

## Overview

BSAtlas is a multimodal AI model designed to provide accurate and clinically relevant descriptions of medical images. Built upon the robust Llama 3.2 11B architecture and fine-tuned with a Vision Transformer (ViT) module, BSAtlas excels at analyzing diverse medical imaging modalities and generating captions that assist medical professionals in diagnosis and treatment planning.

## Key Features

*   **Multimodal Understanding:** Combines textual and visual data processing for comprehensive analysis of medical images and related reports.
*   **Medical Image Expertise:** Fine-tuned on the ROCOv2-radiology dataset, providing specialized knowledge in interpreting various medical imaging modalities (X-rays, CT scans, MRIs, etc.).
*   **Accurate Caption Generation:** Generates detailed and clinically relevant captions that describe key findings in medical images.
*   **Parameter-Efficient Fine-Tuning:** Utilizes LoRA (Low-Rank Adaptation) adapters for efficient fine-tuning, enabling training on limited computational resources.
*   **Publicly Available:** The model, along with detailed documentation, is available on the Hugging Face Hub for easy access and integration.

## Model Architecture

BSAtlas comprises three key components:

1.  **Text Encoder:** Employs the Llama 3.2 11B transformer network to capture contextual relationships in textual inputs (radiology reports, clinical notes, etc.).
2.  **Image Encoder:** Uses a pre-trained Vision Transformer (ViT) to extract visual features from medical images by dividing them into patches and processing them through transformer layers.
3.  **Multimodal Fusion Mechanism:** Implements a cross-attention mechanism to align and integrate textual and visual embeddings, enabling the model to generate contextually relevant responses based on both modalities.

**LoRA Adapters:** The architecture leverages LoRA adapters for parameter-efficient fine-tuning.

## Dataset

The model was trained and evaluated using the ROCOv2-radiology dataset, accessible through the Hugging Face Datasets Hub. This dataset provides a diverse collection of medical images and their corresponding captions, enabling BSAtlas to learn the complex relationships between visual features and textual descriptions in the medical domain.

## Training and Fine-Tuning

*   **Pre-training:** The model was pre-trained on a large-scale multimodal dataset to establish a foundational understanding of both modalities (image and text).
*   **Fine-Tuning:** BSAtlas was fine-tuned on the ROCOv2-radiology dataset using the SFTTrainer from the Transformers Reinforcement Learning (TRL) library.
*   **Optimization:** The AdamW optimizer was used with a learning rate of 1e-4. Mixed-precision training (FP16 or BF16) was employed to reduce memory footprint and accelerate training.
*   **LoRA Adaptation:** LoRA adapters are utilized to enable parameter-efficient fine-tuning, allowing training on limited computational resources such as those available through Google Colab.

## Evaluation

BSAtlas's performance was evaluated using a combination of quantitative and qualitative metrics:

*   **Quantitative Metrics:**
    *   BERTScore: Measures the semantic similarity between generated captions and ground truth references, providing a more nuanced assessment than traditional n-gram overlap metrics.
    *   BLEU and ROUGE (Used mainly to validate that our test is similar to the benchmark, not great score indicators by themself)

*   **Qualitative Evaluation:**
    *   Expert review by radiologists: Assessed the accuracy, relevance, completeness, clarity, and helpfulness of the generated captions on a representative subset of the test data. The expert reviews were used to complement the automated metrics and provide a more comprehensive understanding of the model's clinical utility.

## Usage

Instructions on using the model, including code snippets for generating captions from medical images, are available on the Hugging Face Model Hub page: [https://huggingface.co/BSAtlas](https://huggingface.co/Azzahrae96/BSAtlas)

## Ethical Considerations

The development of BSAtlas followed strict ethical guidelines for AI in healthcare.

*   **Data Anonymization:** All datasets were anonymized to protect patient privacy.
*   **Assistant Tool:** The model's outputs are designed to assist rather than replace clinical decision-making.

## Future Work

*   Expand training dataset with more diverse medical imaging modalities and clinical scenarios.
*   Improve the model's ability to generate more detailed and nuanced captions.
*   Explore the use of reinforcement learning techniques to further optimize the model's performance.
*   Conduct further validation studies with healthcare professionals to assess the model's clinical utility and impact.

## References
