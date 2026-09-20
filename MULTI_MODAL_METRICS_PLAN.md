# Multi-Modal Model Training Metrics Plan

## **Complete Metric Table for 4 Classification Types**

| Metric | **Text** | **Image** | **Audio** | **Video** |
|--------|----------|-----------|-----------|-----------|
| **Primary Modality** | Tweets, comments, posts | Photos, drawings, scans | Speech, commands, audio clips | Multi-frames, video segments, transcripts |
| **Model Architecture** | DistilBERT/RoBERTa | CNN (ResNet, EfficientNet) | Wav2Vec, CNN on spectrograms | CNN-LSTM, ViT, SlowFast |
| **Typical Input Size** | 512-1024 tokens | 224×224 pixels | 16kHz audio, 1-30 seconds | 16-64 frames, 100ms-2fps |
| **Batch Size (GPU 8GB)** | 32-64 | 16-32 | 16-32 | 8-16 |
| **VRAM Required** | 4-6GB | 6-8GB | 4-6GB | 8-12GB |
| **Training Time (500 samples)** | 3-5 min | 5-8 min | 4-7 min | 8-12 min |
| **Training Time (16K samples)** | 45-90 min | 2-3 hours | 2-3 hours | 4-6 hours |
| **Dataset Size (Minimum)** | 500 samples | 1,000 images | 500 audio clips | 200 video segments |
| **Dataset Size (Recommended)** | 5,000-10,000 | 10,000-20,000 | 2,000-5,000 | 1,000-3,000 video segments |
| **Dataset Size (Optimal)** | 50,000+ | 50,000+ | 10,000+ | 5,000+ with diversity |
| **Epochs Needed** | 1-3 | 5-10 | 10-20 | 10-15 |
| **Typical Accuracy** | 85-95% | 75-90% | 70-85% | 75-88% |
| **Complexity** | Low-Medium | Medium-High | Medium | High |
| **Preprocessing** | Tokenization, lowercasing | Resize, normalize, augment | MFCC, spectrogram, augmentation | Frame extraction, temporal augment, audio extraction |

---

## **Dataset Size Recommendations by Modality**

### **Text Classification (Cyberbullying)**
- **Minimum**: 500 samples (quick demo achievable)
- **Recommended**: 5,000-10,000 samples
- **Optimal**: 50,000+ samples
- **Note**: 16,851-47,692 tweets available 

### **Image Classification**
- **Minimum**: 1,000 images (per class if multi-class)
- **Recommended**: 10,000-20,000 images
- **Optimal**: 50,000+ images (ImageNet-scale)
- **Note**: Requires data augmentation for best results

### **Audio Classification**
- **Minimum**: 500 audio clips (1-30 seconds each)
- **Recommended**: 2,000-5,000 audio clips
- **Optimal**: 10,000+ audio clips
- **Note**: Must balance across classes (bullying vs clean speech)

### **Video Classification**
- **Minimum**: 200 video segments (1-5 seconds each)
- **Recommended**: 1,000-3,000 video segments
- **Optimal**: 5,000+ video segments with diversity
- **Note**: Requires frame extraction and temporal feature engineering

---

##  **Key Planning Insights**

### **1. Text is Easiest & Fastest**
- Project already has text datasets working
- Quickest training time (3-5 min for 500 samples)
- Best accuracy results (85-95%)

### **2. Video Requires Most Resources**
- Highest VRAM demand (8-12GB)
- Longest training time
- Needs most dataset diversity
- **Consider**: Extract frames only, not full video processing

### **3. Audio Needs Moderate Resources**
- Similar VRAM to text
- Needs audio feature extraction (MFCC, spectrograms)
- **Consider**: Use existing speech datasets, not raw audio

### **4. Image Needs Medium Resources**
- CNNs well-established for image tasks
- Requires data augmentation (rotation, flip, color jitter)
- **Consider**: Use pretrained ImageNet models (ResNet50, EfficientNet)

---

##  **Resource Planning Checklist**

### **Before Starting Multi-Modal Training:**

- [ ] **VRAM**: 24-32GB total (or train sequentially)
- [ ] **Disk Space**: 10GB+ for all datasets
- [ ] **Time**: 8-15 hours (sequential) or 2-3 hours (parallel)
- **Datasets**:
  - [ ] Text: 5,000-50,000 samples  (already have)
  - [ ] Images: 10,000-50,000 images
  - [ ] Audio: 2,000-10,000 audio clips
  - [ ] Video: 1,000-5,000 video segments
- **Hardware**: GPU with 24GB+ VRAM OR train modalities one-at-a-time

---

## **Recommendation for this Project**

Given existing text datasets (16K-47K samples) and successful quick demo:

### **Phase 1: Text Only**  **(Already Complete)**
- Used existing 16K-47K Twitter/Toxicity datasets
- Achieve 85-95% accuracy
- Training in 15-90 minutes

### **Phase 2: Add Image Modality**
- Collect 10,000+ labeled images
- Use pretrained ResNet50/EfficientNet
- Add 2-3 hours training time
- Accuracy: 75-90%

### **Phase 3: Add Audio Modality**
- Collect 2,000-5,000 audio clips
- Extract MFCC/spectrogram features
- Add 2-3 hours training time
- Accuracy: 70-85%

### **Phase 4: Add Video Modality**
- Collect 1,000-3,000 video segments
- Extract frames + temporal features
- Add 4-6 hours training time
- Accuracy: 75-88%

### **Final Combined Model**
- **Total Training Time**: ~15-25 hours (sequential)
- **Total Datasets**: ~66K+ samples across 4 modalities
- **Expected Accuracy**: 75-95% (weighted by modality)

---

## **Strategic Recommendation**


1. **Phase 1**: Optimize text model with full 16K-47K datasets
2. **Phase 2**: Add Image modality if visual cyberbullying detection needed (memes, posted images)
3. **Phase 3**: Add Audio modality if detecting audio-based bullying (harassing calls, voice messages)
4. **Phase 4**: Add Video modality only if absolutely needed (rare for cyberbullying)
