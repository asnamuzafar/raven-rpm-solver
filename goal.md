Stage A — Visual Aspect (Perception Layer)

The first stage of the system processes the visual content of the RPM puzzles. Each RPM problem consists of a 3×3 grid where the bottom-right panel is missing. The task is to analyze the eight visible context panels along with eight candidate answer images and select the one that completes the grid according to the underlying visual pattern or logical rule. The developer must implement a visual encoder, typically a Convolutional Neural Network (CNN), that takes each image as input and outputs a fixed-length feature vector representing its high-level visual characteristics. ResNet-18 or a similar lightweight architecture is preferred because it balances accuracy with computational efficiency and is widely used in vision-reasoning research.

The encoder should process:
• the 8 context panels (top-left to bottom-middle)
• the 8 answer options

Each image, once passed through the CNN, should produce a feature vector (e.g., 512-dimensional). These feature vectors are the essential raw materials for all later reasoning components. The output of this stage is therefore:
(1) a set of extracted visual features, and (2) clean, preprocessed images ready for further transformation.
No logic or reasoning happens here — only visual understanding.

⸻

Stage B — Tokenizer (Symbolic Abstraction Layer)

The tokenizer converts the continuous feature vectors from Stage A into structured, interpretable symbolic attributes. RPM puzzles are built around visual properties such as shape, size, number of objects, orientation, and color. To allow later reasoning modules to work with explicit concepts rather than raw vectors, this stage predicts discrete symbolic labels for each panel. For example, the tokenizer should output attributes like:
• object shape (square, circle, triangle, polygon)
• size (small, medium, large)
• color (black, gray, white)
• number of objects (1–9)
• optional attributes such as rotation or texture

The tokenizer can be implemented as a small Multi-Layer Perceptron (MLP) classifier or through a lightweight object-centric model such as Slot Attention (optional). The essential requirement is that each panel obtains a set of symbolic attributes that accurately describe visual content. These attributes will allow the symbolic and deep learning reasoning engines to detect relationships between panels.

The output of Stage B is:
a symbolic representation of each image, forming a matrix of attributes that mirrors the structure of the 3×3 puzzle. This enables rule-based logic to operate on the puzzle in later stages.

⸻

Stage C — Reasoner (Primary Deep Learning Reasoning Engine)

This stage performs the actual reasoning task — predicting which of the eight candidate answers correctly completes the 3×3 puzzle. The preferred approach (as suggested by the supervisor) is to build a deep learning–based reasoning engine, such as a Transformer or a feature-fusion neural network. The developer should treat the eight context feature vectors (from Stage A) and the eight candidate answer vectors as a sequence. The model should analyze relationships across all panels simultaneously and learn to infer the underlying visual rule without explicit symbolic instruction.

Two implementations are acceptable:
	1.	A Transformer Encoder that treats each panel embedding as a token and uses attention to learn relationships.
	2.	A simpler MLP-based relational module that combines features pairwise to infer patterns.

The model outputs a probability distribution over the eight candidate options, selecting the most likely answer. This reasoning engine is expected to perform well on standard RPM tasks and serve as one of the main comparison points for symbolic and hybrid methods.

The output of Stage C is:
a deep-learning-based prediction and an internal representation that encodes learned relationships between panels.

⸻

Stage D — Other Models (Baselines and Hybrid Variants)

To evaluate the effectiveness of the primary reasoning engine, we include several additional models. These models are simplified or alternative approaches used to benchmark the system and provide comparative insights. A CNN-Direct model serves as the most basic baseline, where features from each panel are passed through a classifier without any relational reasoning. Another baseline is a Neural Relation Network (RN), which processes every pair of panels to infer relations — a classic architecture for relational tasks. A Transformer-only baseline can also be included, offering insight into how attention-based reasoning performs without symbolic components.

A symbolic rule-based reasoner should also be implemented. This model uses the symbolic attributes from Stage B and applies explicit logical rules such as progression, constant, XOR, and distribution. It does not rely on deep learning for reasoning and serves as the transparent, explainable baseline.

A hybrid model can optionally be created, where the deep learning reasoner proposes likely rules, and the symbolic engine verifies them or vice versa. Each model gives different strengths: some generalize better, some are more interpretable, and some are faster.

The output of Stage D is:
multiple alternative predictions, rule traces (if symbolic), and performance statistics for each baseline model.

⸻

Stage E — Comparison and Evaluation Module

This stage systematically compares all models developed in Stages C and D. The developer must implement a unified evaluation script that runs each model on the same set of RPM tasks and measures performance across several dimensions. These include:
• accuracy on standard test sets
• accuracy on unseen puzzle configurations (generalization)
• sample efficiency (performance with limited training data)
• rule-trace fidelity (how well symbolic models match ground truth rules)
• explanation quality (interpretability)
• computational cost (inference time, parameters)

The comparison should produce tables, plots, or CSV logs that clearly show how the deep learning reasoner performs relative to symbolic and hybrid models. This gives the research contribution credibility and enables experimental analysis in the dissertation.

The output of Stage E is:
a complete comparative analysis of all reasoning approaches, highlighting strengths, weaknesses, and generalization behavior.

⸻

Stage F — Simulator (Interactive Demonstration Tool)

The final stage is an interactive puzzle-solving simulator, ideally implemented in Streamlit. The simulator should allow users to upload or select an RPM puzzle, view the 3×3 grid, and observe how each reasoning engine (deep learning, symbolic, hybrid) processes it. The application should display the predicted answer from each model along with its explanation. For deep learning models, this may include attention maps or relevance scores; for symbolic models, a transparent rule trace should be shown (e.g., “shapes increase across the row”).

The simulator acts as both a demonstration and a validation tool, showing how the system works internally. It enhances user trust and supports the project’s interpretability goals. It also becomes part of your dissertation as evidence of an interactive, functional implementation.

The output of Stage F is:
a fully functional demonstration interface that visualizes model reasoning and allows real-time puzzle solving.

We dont have to follow this to the T, like model selections etc we are free to do what we think works