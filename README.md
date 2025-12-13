# Dynamic-Contextual-Fusion-for-Multi-Modal-Emotional-Intelligence-in-Conversational-Agents 
---
## 1. Project Overview

This project presents the **Dynamic Contextual Emotional Transformer (DCET)**, a novel, multi-stage cascaded deep learning framework designed for robust, real-time temporal tracking of emotional shifts in conversational settings. Addressing the limitations of existing multimodal models that often struggle to adapt to the dynamic nature of real-world interactions, DCET introduces a **Dynamic Contextual Fusion** mechanism. Our architecture not only identifies static emotional states but also adaptively weighs and integrates information from visual, audio, and textual modalities based on their real-time informativeness within the evolving conversational context. This is achieved through a combination of specialized unimodal feature extractors, cross-modal attention for dynamic fusion, and a recurrent memory network for maintaining long-term, per-speaker emotional trajectories. The ultimate goal is to pave the way for more empathetic and effective conversational agents capable of truly understanding and responding to human emotion in live, interactive environments like Augmented Reality (AR).

## 2. Key Features

*   **Dynamic Contextual Fusion:** Adaptive weighting of multimodal cues (visual, audio, text) based on their contextual reliability and informativeness.
*   **Temporal Emotion Tracking:** Utilizes a **Temporal Memory-Augmented Transformer** with a **Recurrent Memory Network** to capture emotional evolution and maintain long-term context across dialogue turns.
*   **Per-Speaker Trajectory Learning:** Tracks individual emotional shifts and histories within multi-party conversations.
*   **Real-time Viability:** Optimized for low-latency inference, suitable for live AR and conversational agent deployments.
*   **DVCEA Dataset:** Introduces a processed, feature-rich dataset derived from MELD, accelerating research into temporal multimodal emotion.
*   **Robustness:** Designed to gracefully handle missing or noisy modalities during inference.

<img width="12301" height="4660" alt="image" src="https://github.com/user-attachments/assets/89c755ae-4506-43c1-821f-b3bacef68c8c" />  

## 3. Architecture Overview (DCET Pipeline)

The DCET framework operates in a multi-stage fashion, both during training and real-time inference:  
### **Architecture Diagram:**  
<img width="5767" height="9538" alt="image" src="https://github.com/user-attachments/assets/424ab0fa-8efd-42f1-8306-3d69d99db224" />  

1.  **Stage 0: Preprocessing & Speaker-Focused Feature Generation:**
    *   Input: Raw MP4 video files (single utterance per file, derived from MELD scenes).
    *   Processes:
        *   **Person Tracking & Segmentation:** Identifies and isolates the `Utterance Speaker` (e.g., via ByteTrack + Mask2Former).
        *   **Visual Feature Extraction ($E_V$):** Speaker-focused facial ROIs fed into a pre-fine-tuned **ResNet-50** backbone.
        *   **Audio Feature Extraction ($E_A$):** Speaker's audio fed into a pre-fine-tuned **Wav2Vec2-base** encoder.
        *   **Text Feature Extraction ($E_T$):** Utterance text (from MELD) fed into a pre-fine-tuned **BERT-base** encoder.
    *   Output: Aligned sequences of unimodal (visual, audio) embeddings and single text embeddings for each utterance, forming the **DVCEA Dataset** in `.npy` format.

2.  **Stage 1: Adaptive Cross-Modal Fusion Module:**
    *   Input: Unimodal embeddings from Stage 0 for a single utterance.
    *   Processes: Leverages **cross-modal attention mechanisms** (e.g., text-conditioned attention for visual/audio) to dynamically fuse modalities into a single, emotion-aware **utterance-level multimodal embedding**.
    *   Goal: Implicitly learn to prioritize modalities based on their informativeness.

3.  **Stage 2: Temporal Memory-Augmented Transformer:**
    *   Input: Sequence of utterance-level multimodal embeddings (from Stage 1), augmented with `Speaker_ID` and positional embeddings, representing a full dialogue scene.
    *   Processes: Utilizes a **Recurrent Memory Transformer (RMT)**. Transformer encoder blocks process the sequence, while the **Recurrent Memory Network** maintains and updates long-term context and per-speaker emotional trajectories across utterances.
    *   Output: Contextualized utterance representations, fed into final prediction heads.

4.  **Prediction Heads:**
    *   Final MLPs for **Categorical Emotion Classification** (7 categories: anger, disgust, fear, joy, neutral, sadness, surprise) and **Continuous Sentiment Regression** (positive/negative/neutral scale).

## 4. Dataset

### **Dynamic-Visual-Conversational-Emotion-Action (DVCEA) Dataset**

The DVCEA Dataset is a novel, processed dataset derived from **1038 dialogue scenes with 9899 individual utterances** of the publicly available **MELD (Multimodal EmotionLines Dataset)**. It comprises high-level, speaker-focused, and time-aligned numerical embeddings in `.npy` format, ready for training temporal models.

*   **Motivation:** DVCEA uniquely enables direct and efficient research into temporal multimodal emotion tracking at the *feature level*, bypassing the computational overhead of raw data pipelines common with MELD or IEMOCAP.
*   **Content:** Contains utterance-level visual and audio feature sequences (from ResNet-50 and Wav2Vec2-base, respectively), utterance-level text embeddings (from BERT-base), and associated MELD labels (`Emotion`, `Sentiment`, `Speaker`, `Dialogue_ID`, `Utterance_ID`).
*   **Structure:** Features are chronologically grouped by `Dialogue_ID` and `Utterance_ID` to form continuous sequences representing full conversational scenes.
*   **Size:** Features derived from **1038 MELD scenes**, encompassing **[Insert Total Utterances Here] utterances** (approx. **[Insert Total Duration Here] hours** of conversational content).
*   **Limitations:** Inherits MELD's **scripted nature** from TV dialogues. Features are constrained by the capabilities of the pre-trained models used for extraction.

## 5. Getting Started

### Prerequisites

*   Python 3.x
*   PyTorch (or TensorFlow)
*   `opencv-python`
*   `transformers` library
*   `librosa`
*   `faiss-cpu` (for memory bank implementation)
*   `numpy`
*   Access to MELD dataset (for raw data processing, if regenerating DVCEA)
*   GPU(s) (NVIDIA recommended, with CUDA support)

### Installation

```bash
# Clone the repository
git clone https://github.com/DHEEPAK29/Dynamic-Contextual-Fusion-for-Multi-Modal-Emotional-Intelligence-in-Conversational-Agents.git 

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r 
