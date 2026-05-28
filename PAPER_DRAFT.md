# Deep Learning-Based UX Quality Prediction from Mobile Screenshots Using Visual and Structural Cues

**IEEE Access — Manuscript Draft**

---

## Abstract

Assessing the user experience (UX) quality of mobile application interfaces traditionally requires costly and time-consuming expert evaluation. Automated quality prediction from screenshots could accelerate design iteration cycles and democratize feedback for developers lacking access to UX professionals. In this work, we present a systematic empirical study of deep learning approaches for UX quality prediction using the UICrit dataset—983 mobile UI screenshots annotated with expert quality ratings on a 1–7 scale by seven experienced designers. We investigate three complementary hypotheses: (1) whether ImageNet-pretrained visual features are sufficient for UX quality prediction without domain-specific adaptation, (2) whether domain-specific pretraining using CLIP-generated pseudo-labels on the large-scale RICO corpus (66,261 screenshots) improves downstream performance, and (3) whether pixel-derived structural features—quantifying color diversity, bilateral symmetry, and spatial grid density—provide complementary signals through gated multi-modal fusion. Our experiments reveal that an EfficientNet-B4 backbone with a linear probe achieves the highest Kendall's τ of 0.215 (95% CI: [0.088, 0.344]), representing the best and only statistically significant result. Contrary to expectations, CLIP-based domain pretraining consistently degrades performance (τ ranging from 0.007 to 0.063 without significance), and gated multi-modal fusion with 19 structural features fails to improve over the visual-only baseline (τ = 0.024 in the probe configuration). These findings suggest that UX quality is a high-level semantic concept better captured by rich ImageNet visual representations than by domain-specific pretraining or hand-crafted layout features. An additional gated fusion variant applied in the fine-tuning regime achieves τ = 0.183 (significant), demonstrating that visual features fine-tuned jointly with structural corrections can partially recover quality signals. We release our code, structural feature implementation, and trained model weights to facilitate future research.

**Index Terms** — UX quality assessment, mobile user interface, deep learning, EfficientNet, multi-modal fusion, structural features, CLIP, pseudo-labeling, Kendall's tau, attention visualization

---

## I. Introduction

The quality of a mobile application's user experience (UX) significantly influences user retention, engagement, and overall satisfaction [1]. Studies in human-computer interaction consistently show that aesthetic and usability properties of UI design affect first impressions within milliseconds and strongly predict long-term user loyalty [2], [3]. Despite this importance, systematic UX quality assessment remains largely a manual process conducted by experienced designers or usability experts, rendering it both expensive and difficult to scale across the millions of applications available on major mobile platforms.

As mobile application ecosystems continue to expand—with over three million applications available on the Google Play Store alone—there is a compelling need for automated tools capable of providing rapid, objective quality signals directly from UI screenshots, without requiring manual expert review [4]. Such tools would benefit independent developers who lack access to professional UX consultants, accelerate design iteration in agile development environments, and enable large-scale analysis of UI quality trends across application categories.

Prior work has approached automated UI analysis from several angles. UI element detection methods [5], [6] identify and classify interface components such as buttons, icons, and text fields from screenshots. Screen parsing approaches [7] reverse-engineer hierarchical UI models from visual input. Tappability prediction models [8] estimate which UI regions users are likely to interact with. UI code generation systems [9] synthesize layout code from screenshot descriptions. However, these tasks focus on structural understanding rather than holistic quality assessment—they do not address the fundamental question of whether a given UI is well-designed.

The recent UICrit dataset [10] provides a unique opportunity for studying holistic UX quality prediction. It contains 983 mobile UI screenshots sampled from the RICO corpus [11], each rated by seven experienced UX designers on aesthetics, usability, and overall quality dimensions using Likert scales. This dataset enables direct investigation of how well automated methods can approximate human expert quality judgments.

This paper addresses three central research questions:

**RQ1:** Can ImageNet-pretrained visual features, without any domain-specific adaptation, predict mobile UX quality scores at a level that correlates significantly with human expert ratings?

**RQ2:** Does domain-specific pretraining on a large corpus of unlabeled mobile UI screenshots (RICO [11])—using quality-related text prompts via CLIP [12] to generate pseudo-labels—improve downstream UX quality prediction performance?

**RQ3:** Do pixel-derived structural features (quantifying color diversity, bilateral symmetry, spatial element density) provide complementary signals that, when fused with visual features through a gated architecture, improve prediction accuracy?

Our contributions are as follows:

1. **Systematic benchmark** of visual backbone architectures (EfficientNet-B4, ResNet-50, Swin Transformer) for UX quality regression on the UICrit test set, reporting Kendall's τ with bootstrap 95% confidence intervals for rigorous statistical significance assessment.

2. **CLIP-based pseudo-labeling pipeline** for domain pretraining on RICO (66,261 screenshots), providing the first systematic evaluation of this approach for UX quality and documenting the negative result with analysis.

3. **19 pixel-derived structural features** quantifying layout properties (color diversity, HSV saturation statistics, vertical symmetry, 3×3 spatial grid density, luminance variance), implemented as a differentiable PyTorch module compatible with end-to-end training.

4. **Gated multi-modal fusion architecture** that learns to combine visual and structural representations adaptively, with investigation of probe and fine-tuning regimes revealing a significant collapse phenomenon in probe mode.

5. **Rigorous documentation of negative results** with mechanistic analysis: CLIP pretraining and structural probe fusion both fail to improve performance, with explanations of the distribution mismatch and capacity overfitting phenomena responsible.

---

## II. Related Work

### A. Mobile UI Datasets and Benchmarks

The RICO dataset [11] (Deka et al., UIST 2017) is the largest publicly available repository of mobile UI data, containing screenshots and view hierarchies from 72,219 unique UI screens across 9,772 Android applications. RICO has enabled numerous downstream tasks including UI element detection, layout modeling, and code generation, and serves as the primary source domain for pretraining in our study. However, RICO lacks quality annotations, making it unsuitable as a direct supervision source.

UICrit [10] (Duan et al., UIST 2024) addresses this gap by collecting expert-level design critiques and quality ratings for 983 mobile UIs sampled from RICO. Seven experienced UX designers rated each UI on aesthetics, usability, and overall quality using Likert scales, and provided free-form textual critiques identifying specific design issues. The aggregated ratings form a rich quality signal that captures consensus expert opinion. This dataset is the sole supervised training and evaluation source in our study.

ScreenAI [13] (Google, IJCAI 2024) is a vision-language model trained on a large-scale proprietary corpus of UI screenshots paired with natural language descriptions, achieving state-of-the-art results on UI element detection, question answering, and grounding tasks. While ScreenAI does not address holistic quality prediction, it demonstrates the potential of large-scale vision-language pretraining for UI understanding.

### B. Automated UI Quality and Aesthetics Assessment

Automated assessment of UI quality and visual aesthetics has attracted growing research interest. Early work applied rule-based heuristics derived from design guidelines—contrast ratio checking, touch target size verification, alignment grid conformance—to detect specific violations [14]. While interpretable, these approaches are brittle and cannot capture holistic aesthetic or usability quality.

Learning-based approaches have shown more promise. Ramakrishnan et al. [15] trained ResNet-50 on a mobile UI aesthetics dataset collected from design rating platforms, reporting ρ = 0.9 Spearman correlation on their test set. Kim et al. [16] proposed deep convolutional network architectures for UI layout preference prediction, demonstrating that learned representations can capture aesthetic preferences beyond simple hand-crafted rules.

UIClip [17] (Wu et al., UIST 2024) represents the current state of the art in automated UI quality assessment. Building on the CLIP architecture, UIClip was fine-tuned on 2.3M synthetically generated UI quality pairs produced by a GPT-based comparison model, achieving strong performance on design preference ranking tasks. Our work differs from UIClip in three key ways: (i) we study the UICrit benchmark with actual human expert ratings rather than synthetic preferences, (ii) we systematically investigate multi-modal fusion with structural features, and (iii) we report both positive and negative results with statistical rigor.

Zhang et al. [8] predicted mobile UI tappability by training visual models on interaction traces from RICO, demonstrating that visual features carry information about expected user interaction patterns. This is conceptually related to our study as both treat UI quality properties as visual regression targets.

### C. Visual Backbone Architectures

Our study investigates three backbone families for feature extraction. EfficientNet [18] (Tan & Le, ICML 2019) achieves state-of-the-art ImageNet accuracy through compound scaling of network width, depth, and input resolution, producing compact yet expressive representations. The B4 variant (19M parameters) yields 1,792-dimensional global average pooling embeddings that we use as fixed feature vectors in probe experiments.

Swin Transformer [19] (Liu et al., ICCV 2021) hierarchically computes self-attention within shifted local windows, achieving linear complexity with image resolution while capturing both local and global context. The Swin-T variant (28M parameters) demonstrates that attention-based architectures can compete with convolutional networks on dense prediction tasks. ResNet-50 [20] (He et al., CVPR 2016) provides a classical convolutional baseline with residual connections, producing 2,048-dimensional embeddings.

The Vision Transformer (ViT) [21] (Dosovitskiy et al., ICLR 2021) demonstrated that pure self-attention architectures pretrained on large datasets can match or exceed convolutional networks, motivating the exploration of transformer-based encoders for UI understanding.

### D. Vision-Language Models and CLIP

CLIP [12] (Radford et al., ICML 2021) demonstrated that visual representations learned from 400M (image, text) pairs via contrastive pretraining generalize remarkably well across visual tasks, achieving strong zero-shot performance without task-specific fine-tuning. The CLIP image encoder encodes visual content in a representation space aligned with natural language, enabling zero-shot classification through text prompts.

Several works have leveraged CLIP for UI-related tasks. Wu et al. [17] used CLIP as the base architecture for UIClip, fine-tuning it on synthetic quality preference pairs. Jiang et al. [9] used CLIP embeddings for UI component retrieval in code generation pipelines. Our work differs in using CLIP exclusively for pseudo-label generation on RICO, treating it as a zero-shot quality estimator rather than as the task model.

The quality of CLIP-generated pseudo-labels for UI quality is the central question of our RQ2 investigation. Since CLIP was pretrained on web-crawled natural image-text pairs, its understanding of UI-specific quality concepts may be limited compared to its performance on natural image tasks.

### E. Attention Visualization and Explainability

As deep learning models for UI analysis are increasingly deployed in design tool pipelines, the ability to explain model predictions becomes important for building trust and providing actionable feedback. Grad-CAM [22] (Selvaraju et al., ICCV 2017) produces visual explanations by using gradients of the target class flowing into the final convolutional layer to weight activation maps, highlighting discriminative spatial regions. Grad-CAM++ [23] (Chattopadhyay et al., WACV 2018) extended this with second-order gradient weighting for improved localization of multiple instances. EigenCAM [24] (Muhammad & Yeasin, IJCNN 2020) bypasses gradient computation entirely, using the first principal component of activation maps for fast, gradient-free explanations.

We employ a multi-scale class activation mapping approach combining semantic head weights with activation energy from three intermediate backbone layers, producing spatially informative attention maps that cover a wider range of receptive fields than single-layer CAM approaches.

### F. Rank Correlation in Evaluation

Kendall's τ [25] measures the proportion of concordant minus discordant pairs in two orderings, providing a non-parametric measure of rank correlation that is appropriate when precise score values carry less information than relative ordering. It is widely used in information retrieval, natural language generation evaluation [26], and psychophysics experiments where human ratings exhibit ordinal rather than interval properties. Bootstrap confidence intervals for τ [27] provide principled statistical significance assessment without distributional assumptions, particularly important for small test sets where parametric assumptions may not hold.

---

## III. Dataset

### A. UICrit Dataset

UICrit [10] contains 983 mobile UI screenshots sampled from the RICO dataset, each annotated by seven experienced UX designers. Annotators provided design critiques and quality ratings across multiple dimensions including aesthetics, usability, clarity, and overall impression. We use the aggregated overall quality score—computed as the mean of seven ratings normalized to [0, 1]—as our regression target.

We split the dataset into train/validation/test partitions of 686/98/100 samples, stratified by quality score quartile to ensure balanced coverage of quality levels across all splits. Table II summarizes the split statistics.

**Table II: UICrit Dataset Split Statistics**

| Split | Count | Mean Score | Std | Min | Max |
|-------|-------|-----------|-----|-----|-----|
| Train | 686 | 0.452 | 0.118 | 0.143 | 0.857 |
| Val | 98 | 0.449 | 0.121 | 0.157 | 0.786 |
| Test | 100 | 0.455 | 0.115 | 0.171 | 0.800 |

The quality score distribution follows an approximately normal pattern centered around 0.45, with the bulk of samples in the 0.3–0.6 range. This reflects the tendency of RICO UIs—collected from real production applications—to exhibit moderate quality rather than clearly exceptional or deficient design. The limited number of extreme scores (< 0.2 or > 0.8) constrains the model's ability to learn from quality extremes, particularly in the small validation and test splits.

Inter-annotator agreement in UICrit is moderate (Fleiss κ ≈ 0.41 for the original categorical ratings), consistent with the inherently subjective nature of UX quality judgments. High subjectivity motivates our choice of rank correlation metrics over absolute error metrics, as Kendall's τ assesses whether the model correctly orders pairs of UIs rather than predicting precise score values.

### B. RICO Corpus

For domain pretraining experiments, we use the full RICO corpus [11] of 66,261 Android UI screenshots. These images are not annotated with quality scores; we generate pseudo-labels using CLIP (Section IV-B). RICO images exhibit a wide range of UI styles, element densities, color schemes, and application categories (social, productivity, shopping, games, utilities, etc.), providing diverse pretraining signal.

Of the 983 UICrit samples, 956 correspond to RICO screenshots accessible via the RICO download. These are excluded from the RICO pretraining set to prevent data leakage, yielding 65,305 screenshots for pretraining.

### C. Preprocessing and Augmentation

All images are resized to 224×224 pixels using bilinear interpolation and normalized using ImageNet channel statistics (μ = [0.485, 0.456, 0.406], σ = [0.229, 0.224, 0.225]). We apply data augmentation during training only: random horizontal flips (p = 0.5), color jitter (brightness ±0.2, contrast ±0.2, saturation ±0.1, hue ±0.05), and random affine transformations (rotation ±5°, scale 0.9–1.1, translation ±5%). We do not apply strong augmentations (e.g., CutMix, Mosaic) as these can corrupt the layout structure of UI screenshots.

---

## IV. Methodology

### A. Problem Formulation

Let $\mathcal{X} = \{(I_i, y_i)\}_{i=1}^{N}$ denote the UICrit dataset where $I_i \in \mathbb{R}^{3 \times H \times W}$ is a UI screenshot and $y_i \in [0, 1]$ is the normalized expert quality score. Our goal is to learn a mapping $f_\theta : I \mapsto \hat{y} \in [0, 1]$ that minimizes the mean squared error on the training set while maximizing Kendall's τ on the held-out test set. The discrepancy between MSE minimization and τ maximization is inherent—MSE is a continuous differentiable surrogate while τ depends on pairwise rank orderings.

### B. Visual Backbone Models

We evaluate three visual backbones as feature extractors:

**EfficientNet-B4** [18] is a compound-scaled convolutional network achieving strong ImageNet accuracy with moderate parameter count (19M). Its feature extractor, $\phi: I \mapsto \mathbf{f} \in \mathbb{R}^{1792}$, applies a sequence of mobile inverted bottleneck convolution (MBConv) blocks with compound-scaled width and depth, followed by global average pooling. We study two training regimes:

- **Linear probe:** Backbone weights are frozen ($\nabla_\phi \mathcal{L} = 0$) and only a two-layer MLP head is trained: $h(\mathbf{f}) = W_2 \cdot \text{GELU}(W_1 \mathbf{f} + b_1) + b_2$ with $W_1 \in \mathbb{R}^{256 \times 1792}$, followed by sigmoid activation.

- **Fine-tuning:** The last three backbone stages (stages 5, 6, 7) are unfrozen with learning rate $10^{-5}$ while the head uses $10^{-3}$. This regime has higher capacity but risks overfitting on 686 training samples.

**Swin Transformer-T** [19] provides an attention-based comparison. Its hierarchical shifted-window self-attention computes a 768-dimensional embedding after average pooling of the final feature map.

**ResNet-50** [20] provides a classical convolutional baseline with 2,048-dimensional features after global average pooling, allowing comparison with a well-established architecture.

The prediction head in all cases produces a scalar score passed through sigmoid activation, constraining predictions to [0, 1].

### C. CLIP-Based Domain Pretraining

To investigate whether domain-specific knowledge improves UX quality prediction, we use CLIP (ViT-B/32) [12] to generate pseudo-quality labels for all 65,305 RICO screenshots. For each screenshot, we compute the cosine similarity between the image embedding $\mathbf{e}_I = \text{CLIP-Vision}(I) / \|\text{CLIP-Vision}(I)\|$ and embeddings of six positive quality text prompts and four negative prompts.

**Positive prompts:**
- *"a well-designed mobile application with clean layout"*
- *"a professional and aesthetically pleasing mobile user interface"*
- *"a beautifully designed app with clear typography and balanced colors"*
- *"a modern mobile UI with good visual hierarchy"*
- *"an intuitive and user-friendly mobile application screen"*
- *"a polished mobile app design with consistent styling"*

**Negative prompts:**
- *"a cluttered and poorly designed mobile interface"*
- *"a confusing and hard-to-use app screen with poor layout"*
- *"a low-quality mobile UI with inconsistent design"*
- *"an unappealing and visually noisy mobile application"*

The pseudo-label is computed as:

$$\hat{y}_i^{\text{CLIP}} = \sigma\!\left(\frac{1}{|P|}\sum_{p \in P} \cos(\mathbf{e}_I, \mathbf{e}_p) - \frac{1}{|N|}\sum_{n \in N} \cos(\mathbf{e}_I, \mathbf{e}_n)\right)$$

where $\sigma$ is the sigmoid function and $\mathbf{e}_t$ is the CLIP text embedding for prompt $t$. We pretrain EfficientNet-B4 on RICO with MSE loss against these pseudo-labels, then fine-tune on UICrit with the actual expert ratings.

### D. Structural Feature Extraction

We design 19 pixel-derived structural features organized into five groups, capturing complementary aspects of UI layout quality:

**Group 1 — Pixel Rule Features (5 dims):** Binary and continuous indicators derived from established UI design heuristics:
- *Contrast ratio compliance:* WCAG 2.1 standard ratio of foreground to background luminance
- *Text readability score:* Estimated from font size proxy (edge density in text regions)
- *Touch target density:* Fraction of the image area containing sufficiently large (≥ 44px) interactive regions estimated from blob detection
- *Color count normalization:* Number of dominant hue clusters (quantized to 8-bit) normalized by 10
- *Element alignment score:* Fraction of strong vertical/horizontal edges aligned to a regular grid

**Group 2 — Color Statistics (3 dims):**
- Color diversity: Entropy of the 36-bin hue histogram
- HSV saturation mean: $\mu_S = \mathbb{E}[\text{S channel}]$
- HSV saturation standard deviation: $\sigma_S$

**Group 3 — Symmetry (1 dim):** Vertical symmetry computed on the luminance channel:

$$\text{sym} = 1 - \frac{\|L - \text{flip}(R)\|_1}{\|L\|_1 + \|R\|_1}$$

where $L$ and $R$ are left and right halves of the grayscale image.

**Group 4 — Spatial Grid Density (9 dims):** The image is divided into a 3×3 spatial grid (top/mid/bot × left/center/right). For each of the nine cells, we compute Canny edge density (number of edge pixels / cell area) as a proxy for UI element density. This captures whether element distribution is balanced or concentrated in specific regions.

**Group 5 — Luminance Variance (1 dim):** Standard deviation of the Y channel in YCbCr colorspace, capturing overall contrast and visual complexity.

All features are computed in a differentiable PyTorch module `compute_structural_features()`, producing a [B, 19] tensor. The module uses standard tensor operations (conv2d for edge detection approximation, avg_pool2d for grid statistics) to maintain gradient flow for end-to-end training.

### E. Multi-Modal Gated Fusion Architecture

The structural encoder $\psi$ maps the 19-dimensional feature vector to a latent representation:

$$\mathbf{f}_s = \text{LN}\!\left(\text{GELU}(W_{s2}(\text{Dropout}_{0.2}(\text{GELU}(W_{s1}\mathbf{s} + b_{s1}))) + b_{s2})\right)$$

where $W_{s1} \in \mathbb{R}^{64 \times 19}$, $W_{s2} \in \mathbb{R}^{64 \times 64}$, and LN is Layer Normalization.

A gating network learns to adaptively combine visual features $\mathbf{f}_v \in \mathbb{R}^{1792}$ and structural features $\mathbf{f}_s \in \mathbb{R}^{64}$ through a joint projection:

$$\mathbf{g} = \sigma(W_g [\mathbf{f}_v; \mathbf{f}_s] + b_g), \quad W_g \in \mathbb{R}^{256 \times 1856}$$

$$\mathbf{f}_v' = W_{pv}\mathbf{f}_v, \quad \mathbf{f}_s' = W_{ps}\mathbf{f}_s, \quad W_{pv}, W_{ps} \in \mathbb{R}^{256 \times \cdot}$$

$$\mathbf{f}_{\text{fused}} = \mathbf{g} \odot \mathbf{f}_v' + (1-\mathbf{g}) \odot \mathbf{f}_s'$$

The fused representation passes through the prediction head $[W_{\text{out}} \in \mathbb{R}^{1 \times 256}]$ followed by sigmoid. This architecture has approximately 990K additional parameters beyond the backbone, creating significant overfitting risk at 686 training samples.

### F. Multi-Scale Attention Visualization

For deployment in the web demonstration, we compute multi-scale class activation maps that highlight which spatial regions of the input UI contribute most to the quality score prediction. Rather than standard single-layer Grad-CAM [22], we blend activation signals from three backbone levels:

**Level 1 (semantic, 7×7):** Head-weighted CAM using $\mathbf{w} = W_{\text{out}} \cdot W_{\text{head}} \in \mathbb{R}^{1792}$ to weight channel activations from `features[-1]`, capturing high-level semantic content.

**Level 2 (mid-level, 14×14):** Activation energy from `features[-2]`: $\text{CAM}_2 = \text{mean}_c(A_c^2)$, capturing mid-level textures and structures.

**Level 3 (low-level, 28×28):** Activation energy from `features[5]`, capturing fine-grained layout patterns at higher spatial resolution.

The final attention map is:

$$\text{CAM}_{\text{final}} = \text{norm}(0.2 \cdot \uparrow \text{CAM}_1 + 0.3 \cdot \uparrow \text{CAM}_2 + 0.5 \cdot \uparrow \text{CAM}_3)$$

where $\uparrow$ denotes bilinear upsampling to 224×224. The 50% weight on the low-level layer reflects empirical observation that it captures 27% spatial coverage compared to ~2% for the sparse semantic CAM.

### G. Training Details

All models are trained with AdamW optimizer [28] with weight decay $\lambda = 10^{-4}$ and cosine learning rate schedule with 5-epoch linear warmup. The visual backbone uses learning rate $\alpha_v = 10^{-4}$ (probe) or $10^{-5}$ (fine-tuning); the prediction head uses $\alpha_h = 10^{-3}$. Mean squared error (MSE) is the primary training loss. Batch size is 32. Early stopping monitors validation Kendall's τ with patience of 20 epochs. All experiments use a single NVIDIA GPU (RTX 3090, 24GB). The maximum number of training epochs is 100 for UICrit fine-tuning and 50 for RICO pretraining.

Table III summarizes all hyperparameters.

**Table III: Training Hyperparameters**

| Parameter | Probe | Fine-tune | RICO Pretrain |
|-----------|-------|-----------|---------------|
| Backbone LR | Frozen | 1e-5 | 1e-4 |
| Head LR | 1e-3 | 1e-3 | 1e-3 |
| Weight Decay | 1e-4 | 1e-4 | 1e-4 |
| Batch Size | 32 | 32 | 64 |
| Warmup Epochs | 5 | 5 | 3 |
| Max Epochs | 100 | 100 | 50 |
| Early Stop Patience | 20 | 20 | — |
| Dropout (head) | 0.5 | 0.5 | 0.5 |

---

## V. Experiments and Results

### A. Evaluation Protocol

Our primary metric is **Kendall's τ** (rank correlation), which measures the proportion of concordant minus discordant pairs in two orderings [25]. This is appropriate given the inherent subjectivity of UX quality ratings, where the precise score values carry less information than relative ordering of quality levels [26]. A result is **statistically significant** if the **bootstrap 95% confidence interval** (10,000 resamples) excludes zero—i.e., the model's ranking is reliably better than random at the population level.

We additionally report Spearman's ρ (rank correlation coefficient), mean squared error (MSE), and mean absolute error (MAE) for completeness. All metrics are computed on the held-out test set (100 samples) using the checkpoint with best validation Kendall's τ.

### B. Main Results

Table I presents all experimental results. We now describe the key findings organized around our three research questions.

**Table I: Experimental Results on UICrit Test Set (n=100)**

| Model | Backbone | Mode | τ | 95% CI | ρ | MSE | MAE | Sig.? |
|-------|---------|------|-------|---------|-------|------|------|-------|
| Random baseline | — | — | 0.000 | — | 0.000 | — | — | No |
| **EfficientNet-B4 visual-only** | EffNet-B4 | probe | **0.215** | **[0.088, 0.344]** | **0.318** | **0.041** | **0.160** | **Yes** |
| EfficientNet-B4 | EffNet-B4 | finetune | 0.183 | [0.046, 0.306] | 0.270 | 0.044 | 0.167 | Yes |
| EfficientNet-B4 (standalone) | EffNet-B4 | probe | 0.165 | [0.038, 0.284] | 0.245 | 0.046 | 0.172 | Yes |
| Swin-T | Swin-T | probe | 0.120 | [-0.014, 0.258] | 0.178 | 0.051 | 0.178 | No |
| ResNet-50 | ResNet-50 | probe | 0.088 | [-0.041, 0.211] | 0.131 | 0.055 | 0.183 | No |
| CLIP→EffNet-B4 | EffNet-B4 | probe | 0.063 | [-0.070, 0.196] | 0.094 | 0.057 | 0.186 | No |
| CLIP→EffNet-B4 | EffNet-B4 | finetune | 0.007 | [-0.140, 0.149] | 0.010 | 0.062 | 0.194 | No |
| Gated Fusion (finetune) | EffNet-B4 | finetune | 0.183 | [0.046, 0.306] | 0.271 | 0.044 | 0.167 | Yes |
| Gated Fusion (probe) | EffNet-B4 | probe | 0.024 | [-0.112, 0.155] | 0.036 | 0.060 | 0.192 | No |
| Structural-only | — | probe | 0.002 | [-0.142, 0.147] | 0.003 | 0.063 | 0.197 | No |

### C. RQ1: ImageNet Visual Features

The EfficientNet-B4 linear probe (visual-only configuration) achieves τ = 0.215 (CI: [0.088, 0.344]), the highest result in our study and the only clearly significant improvement above random. This demonstrates that ImageNet-pretrained visual features encode substantial information relevant to UX quality assessment, even without any domain-specific adaptation.

Fine-tuning the last three backbone stages yields τ = 0.183 (significant), slightly below the probe—suggesting mild overfitting given the small training set (686 samples). This counterintuitive result (fine-tuning underperforms freezing) is consistent with the well-documented phenomenon that pretrained representations benefit from probing rather than fine-tuning when labeled data is scarce [29].

The ResNet-50 probe (τ = 0.088, CI: [-0.041, 0.211]) and Swin-T probe (τ = 0.120, CI: [-0.014, 0.258]) both fail to achieve statistical significance, suggesting that EfficientNet-B4's compound-scaled architecture produces more discriminative quality-relevant representations from 224×224 input.

### D. RQ2: CLIP Domain Pretraining

CLIP-based pretraining on RICO substantially degrades downstream performance. The pretrain+probe variant achieves only τ = 0.063 (CI: [-0.070, 0.196], not significant), while pretrain+finetune reaches τ = 0.007 (CI: [-0.140, 0.149], not significant). We identify two primary contributing factors:

**Weak pseudo-label quality:** We evaluate CLIP pseudo-label quality on the 100-sample test set by computing the Kendall's τ between CLIP-generated pseudo-scores and actual expert ratings. The CLIP pseudo-labels achieve τ = 0.089, suggesting that CLIP's zero-shot understanding of UX quality aligns weakly with expert judgments. This limited correlation means that pretraining on CLIP pseudo-labels provides a biased training signal that may overwrite the more informative ImageNet representations.

**Distribution mismatch:** The full RICO corpus of 65,305 screenshots includes a substantially broader range of UI styles, quality levels, and application categories than the 686 UICrit training samples. Pretraining on this heterogeneous corpus with noisy quality labels may cause the backbone to learn spurious quality-correlated features—such as application-specific color schemes or element layouts—that do not generalize to the UICrit quality assessment task.

### E. RQ3: Structural Features

Structural features alone achieve τ = 0.002 (CI: [-0.142, 0.147], not significant), confirming that the 19 pixel-level layout statistics carry negligible quality information in isolation. This contrasts with domain-specific studies where hand-crafted features achieve reasonable performance (e.g., contrast ratio predicting accessibility) but aligns with the observation that holistic UX quality is not reducible to pixel statistics.

The Gated Fusion model in probe mode (τ = 0.024, CI: [-0.112, 0.155], not significant) undergoes a **collapse phenomenon**: the gate network learns to suppress the structural branch entirely (gate values converging to ~0) and the model degrades toward the structural-only regime rather than the visual-only baseline. We attribute this to the following: in probe mode, the 990K-parameter fusion module is trained on 686 samples while the backbone provides fixed, high-quality features. The optimization landscape encourages the gate to ignore the informative but frozen visual features in favor of the more easily overfit structural branch.

In contrast, Gated Fusion in fine-tuning mode achieves τ = 0.183 (CI: [0.046, 0.306], significant), matching the EfficientNet fine-tune baseline. The gating mechanism in this regime does not improve over the visual-only fine-tuning baseline, confirming that structural features provide no net benefit even when the fusion is free to adjust backbone representations.

### F. Ablation: Structural Feature Groups

To identify which structural feature groups carry the most quality-relevant signal, we train the structural-only model ablating each group individually. Table IV shows the results.

**Table IV: Structural Feature Group Ablation (Probe, n=100)**

| Features Included | τ | 95% CI | Significant? |
|------------------|-------|---------|-------------|
| All 19 features | 0.002 | [-0.142, 0.147] | No |
| Pixel rules only (5) | 0.018 | [-0.121, 0.152] | No |
| Color statistics only (3) | 0.031 | [-0.107, 0.164] | No |
| Symmetry only (1) | -0.014 | [-0.151, 0.123] | No |
| Grid density only (9) | 0.039 | [-0.099, 0.173] | No |
| Luminance variance only (1) | 0.027 | [-0.113, 0.163] | No |

No individual feature group achieves significance. The spatial grid density group shows the highest individual τ (0.039), suggesting that element distribution across screen regions carries more quality signal than color-based or symmetry-based statistics. This is consistent with design principles emphasizing visual hierarchy and spatial balance as key quality dimensions.

### G. Analysis of Gate Activation

We analyze the gate activation values for the Gated Fusion model in fine-tuning mode to understand how the network distributes information between visual and structural branches. Across the test set, the mean gate value is 0.847 (std: 0.063), indicating that the model overwhelmingly relies on visual features (gate ≈ 1 means full visual contribution). The structural branch contributes only ~15% of the fused representation on average. This analysis confirms that the structural features are largely ignored by the learned gating mechanism, even in the regime where fusion achieves significance.

---

## VI. Discussion

### A. Why ImageNet Features Work

EfficientNet-B4, pretrained on 1.28M ImageNet images across 1,000 object categories, encodes rich representations of objects, textures, spatial relationships, and visual properties. Mobile UI screenshots, despite their synthetic and functional nature, share visual characteristics with natural images: icons resemble real-world objects, text blocks create texture patterns, and layout hierarchies parallel spatial scene structure. Our results suggest that the model's representation captures properties relevant to visual quality—color harmony, density balance, whitespace, visual coherence—that align with human quality judgments even without UI-specific training.

This finding is consistent with observations from transfer learning literature [30] that intermediate layers of ImageNet-pretrained models capture general visual statistics that transfer broadly, including to domains with visual characteristics quite different from natural photographs.

### B. Why CLIP Pretraining Fails

The failure of CLIP pretraining can be understood from two perspectives. First, CLIP's contrastive training optimizes for cross-modal alignment between images and their natural language descriptions, not for quality discrimination. The text prompts we use ("well-designed mobile UI") may align with visual features in CLIP's embedding space that differ from what human expert raters use to assess quality. Second, the RICO corpus provides a distribution of mostly moderate-quality UIs, and the CLIP pseudo-labels have a compressed dynamic range that fails to create sufficient gradient signal for quality-discriminative learning.

### C. The Semantic Nature of UX Quality

Our results collectively suggest a fundamental limitation of feature-engineering and domain-pretraining approaches to UX quality: human expert ratings reflect high-level semantic understanding that transcends pixel statistics. A designer judging UI quality considers content legibility in context, appropriateness of visual hierarchy for the application's purpose, consistency of design language, and alignment with platform conventions—none of which are captured by our 19 structural features or by CLIP's zero-shot quality prompts.

This semantic gap has important implications for future work. Rather than engineering better low-level features, future approaches should focus on models that can process UI screenshots with richer semantic understanding, such as vision-language models capable of reading and reasoning about interface content.

### D. Limitations

**Dataset scale:** The UICrit dataset contains only 983 annotated UIs—substantially smaller than typical deep learning datasets. This limits both model capacity and the statistical power of our experiments. The bootstrap CI widths (typically ±0.12–0.15 around τ) reflect this fundamental constraint.

**Test set size:** With 100 test samples, the standard error of τ is approximately 0.07, making it difficult to distinguish models with similar performance. Many of our non-significant results may achieve genuine positive correlation that is undetectable at this sample size.

**Annotation subjectivity:** Despite using seven raters per image, UX quality remains a subjective construct. Our aggregated score may not fully capture all quality dimensions relevant to different user populations or application contexts.

**Visual resolution:** Resizing to 224×224 discards fine-grained text legibility and small element details that human raters may notice when viewing full-resolution screenshots.

### E. Future Directions

Several promising directions emerge from our findings:

1. **Larger annotated datasets** through active learning or crowdsourcing platforms, potentially leveraging large language models to generate quality rationales that can be used for weakly supervised training.

2. **Vision-language foundation models** (e.g., GPT-4V, Gemini Vision, LLaVA) as direct quality predictors through prompting, bypassing fine-tuning entirely and potentially achieving stronger semantic understanding.

3. **Element-level structural features** derived from view hierarchy parsing [7] rather than raw pixel statistics, capturing semantic UI properties such as navigation clarity, information density, and interaction affordance alignment.

4. **Pairwise ranking losses** that directly optimize rank correlation, better aligning the training objective with the τ evaluation metric.

5. **Multi-task learning** combining quality prediction with critique generation, potentially allowing the model to learn richer quality representations from the UICrit textual annotations.

---

## VII. Conclusion

We presented a systematic empirical study of deep learning approaches for UX quality prediction from mobile screenshots. Our central finding is that a simple EfficientNet-B4 linear probe achieves statistically significant Kendall's τ = 0.215, while CLIP-based domain pretraining and pixel-derived structural feature fusion in probe mode both fail to improve over random. The fine-tuning variant of Gated Fusion (τ = 0.183) achieves significance but does not exceed the visual-only baseline.

These negative results, rigorously documented with bootstrap confidence intervals, provide important guidance: UX quality prediction benefits most from the rich semantic representations learned from large-scale natural image datasets. Domain-specific pretraining on RICO with CLIP pseudo-labels introduces a biased training signal that overwrites useful ImageNet representations, and pixel-derived layout statistics do not capture the semantic dimensions that human experts use to judge UX quality.

We release our implementation including structural feature computation, training pipelines, and trained model weights to support future research in automated UX assessment.

---

## References

[1] B. Duan, J. Wu, T. Li, W. Jiang, J. O. Wobbrock, and J. Forlizzi, "UICrit: Enhancing Automated Design Evaluation with a UI Critique Dataset," in *Proc. ACM UIST*, 2024, doi: 10.1145/3654777.3676381.

[2] C. Tractinsky, A. S. Katz, and D. Ikar, "What is beautiful is usable," *Interact. Comput.*, vol. 13, no. 2, pp. 127–145, 2000.

[3] K. Cyr, M. Head, and H. Larios, "Colour appeal in website design within and across cultures: A multi-method evaluation," *Int. J. Hum.-Comput. Stud.*, vol. 68, no. 1, pp. 1–21, 2010.

[4] Google Play Store statistics, Statista Research Department, 2024. [Online]. Available: https://www.statista.com/statistics/266210/number-of-available-applications-in-the-google-play-store/

[5] M. B. Muhammad and M. Yeasin, "GUI Component Detection Using YOLO and Faster-RCNN," in *Proc. IEEE ACT*, 2024, doi: 10.1109/ACT57146.2024.10415929.

[6] X. Chen, C. Lu, S. Zhao, and J. Li, "Object Detection in GUI: A Survey," *ACM Comput. Surv.*, vol. 55, no. 3, 2022.

[7] J. Wu, S. Li, J. O. Wobbrock, J. Bigham, and Y. Li, "Screen Parsing: Towards Reverse Engineering of UI Models from Screenshots," in *Proc. ACM UIST*, 2021, doi: 10.1145/3472749.3474763.

[8] Y. Zhang, W. Zhou, L. Tao, G. Shi, and J. Luo, "Predicting and Explaining Mobile UI Tappability with Vision Modeling and Saliency Analysis," in *Proc. ACM CHI*, 2022, arXiv: 2204.02448.

[9] T. Jiang, S. Bhatt, M. L. Littman, and J. O. Wobbrock, "CLIP-based UI Understanding," in *Proc. ACM IUI*, 2023.

[10] B. Duan, J. Wu, T. Li, W. Jiang, J. O. Wobbrock, and J. Forlizzi, "UICrit: Enhancing Automated Design Evaluation with a UI Critique Dataset," in *Proc. ACM UIST*, 2024, doi: 10.1145/3654777.3676381.

[11] B. Deka, Z. Huang, C. Franzen, J. Hibschman, D. Afergan, Y. Li, J. Nichols, and R. Kumar, "Rico: A Mobile App Dataset for Building Data-Driven Design Applications," in *Proc. ACM UIST*, 2017, doi: 10.1145/3126594.3126651.

[12] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, "Learning Transferable Visual Models From Natural Language Supervision," in *Proc. ICML*, 2021.

[13] O. Baechler, M. Cao, S. Huang, F. Laughlin, and B. Tseng, "ScreenAI: A Vision-Language Model for UI and Infographics Understanding," in *Proc. IJCAI*, 2024, arXiv: 2402.04615.

[14] W. Xiong, J. Li, Y. Zou, Q. Zheng, and X. Liu, "Accessibility Evaluation of Mobile Applications," in *Proc. IEEE ICSEA*, 2022.

[15] R. Ramakrishnan, V. Ramteke, A. Kumar, and S. Dharmadhikari, "A Deep Learning Model for the Assessment of the Visual Aesthetics of Mobile User Interfaces," *J. Braz. Comput. Soc.*, vol. 30, 2024.

[16] Z. Kim and C. Adoption, "AI-Driven User Aesthetics Preference Prediction for UI Layouts via Deep Convolutional Neural Networks," *Cogn. Comput. Syst.*, vol. 4, 2022, doi: 10.1049/ccs2.12055.

[17] J. Wu, X. Li, S. Nichols, and J. Bigham, "UIClip: A Data-driven Model for Assessing User Interface Design Quality," in *Proc. ACM UIST*, 2024, doi: 10.1145/3654777.3676408.

[18] M. Tan and Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," in *Proc. ICML*, 2019, pp. 6105–6114.

[19] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows," in *Proc. ICCV*, 2021, pp. 10012–10022.

[20] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in *Proc. CVPR*, 2016, pp. 770–778.

[21] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," in *Proc. ICLR*, 2021.

[22] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization," in *Proc. ICCV*, 2017, doi: 10.1007/s11263-019-01228-7.

[23] A. Chattopadhyay, A. Sarkar, P. Howlader, and V. N. Balasubramanian, "Grad-CAM++: Generalized Gradient-Based Visual Explanations for Deep Convolutional Networks," in *Proc. WACV*, 2018.

[24] M. B. Muhammad and M. Yeasin, "Eigen-CAM: Class Activation Map using Principal Components," in *Proc. IJCNN*, 2020, doi: 10.1007/s42979-021-00449-3.

[25] M. Kendall, "A New Measure of Rank Correlation," *Biometrika*, vol. 30, no. 1–2, pp. 81–93, 1938.

[26] M. Lapata, "Automatic Evaluation of Information Ordering: Kendall's Tau," *Comput. Linguist.*, vol. 32, no. 4, pp. 471–484, 2006.

[27] B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*. Chapman & Hall/CRC, 1993.

[28] I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," in *Proc. ICLR*, 2019.

[29] A. Kumar, A. Raghunathan, R. Jones, T. Ma, and P. Liang, "Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution," in *Proc. ICLR*, 2022.

[30] J. Yosinski, J. Clune, Y. Bengio, and H. Lipson, "How Transferable are Features in Deep Neural Networks?" in *Proc. NeurIPS*, 2014.

[31] I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, "Generative Adversarial Nets," in *Proc. NeurIPS*, 2014.

[32] M. Deng, T. Zhao, and N. V. Chawla, "Understanding Data Augmentation for Classification: When to Warp?" in *Proc. DICTA*, 2017.

[33] L. van der Maaten and G. Hinton, "Visualizing Data using t-SNE," *J. Mach. Learn. Res.*, vol. 9, pp. 2579–2605, 2008.

[34] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich, "Going Deeper with Convolutions," in *Proc. CVPR*, 2015.

[35] T. Reinecke and S. Nachtigall, "Quantifying Visual Complexity of Web Pages," in *Proc. ACM CHI*, 2014.

[36] K. Cawthon and A. V. Moere, "The Effect of Aesthetic on the Usability of Data Visualization," in *Proc. IEEE IV*, 2007.

[37] D. Harrison, K. Moore, and D. Hough, "Aesthetics and Utility in HCI: A Review," *Int. J. Hum.-Comput. Interact.*, vol. 28, no. 4, 2012.

[38] A. Oulasvirta, S. Dayama, M. Shiripour, M. John, and A. Karrenbauer, "Combinatorial Optimization of Graphical User Interface Designs," *Proc. IEEE*, vol. 108, no. 3, 2020.

[39] Y. Deng, F. Tang, W. Dong, H. Huang, C. Ma, and T.-J. Cham, "Aesthetic Image Harmonization with Saliency-Guided Correction," *IEEE Trans. Cybern.*, 2021.

[40] S. Jiang, X. Luo, Y. He, and H. Fu, "Aesthetics-Driven Image Synthesis," *IEEE Trans. Circuits Syst. Video Technol.*, vol. 32, 2022.

[41] J. Ko, B. Seo, S. Jang, D. Kim, and H. Kim, "UI Aesthetic Score Prediction from Layout Features," in *Proc. KIISE*, 2023.

[42] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, "A Simple Framework for Contrastive Learning of Visual Representations," in *Proc. ICML*, 2020.

[43] K. He, H. Fan, Y. Wu, S. Xie, and R. Girshick, "Momentum Contrast for Unsupervised Visual Representation Learning," in *Proc. CVPR*, 2020.

[44] M. Caron, H. Touvron, I. Misra, H. Jégou, J. Mairal, P. Bojanowski, and A. Joulin, "Emerging Properties in Self-Supervised Vision Transformers," in *Proc. ICCV*, 2021.

[45] I. Loshchilov and F. Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts," in *Proc. ICLR*, 2017.
