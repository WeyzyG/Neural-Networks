# Initial prompt
Role:
You are a senior full-stack engineer and ML educator building an advanced browser-based Fashion MNIST classification platform for e-commerce applications.

Context:

Create a production-ready web application deployable on GitHub Pages that performs complete client-side Fashion MNIST classification using TensorFlow.js and tfjs-vis. The application should demonstrate real-world business value for fashion marketplaces with comprehensive analytics and model management.

Data will be provided via file inputs: fashion_train.csv and fashion_test.csv in standard MNIST format (label + 784 pixels). Additionally support unlabeled data for production classification.

CSV format: each row = label (0-9) followed by 784 pixel values (0-255) with no header. Unlabeled data should have 784 pixels only.

Implement robust file-based model operations: save as downloadable model.json + weights.bin, load from user-selected files.

Include three main operational phases: Business Analysis, Model Training & Validation, and Production Deployment with real-time classification capabilities.

Instruction:
Output exactly three fenced code blocks, in this order, labeled "index.html", "data-loader.js", and "app.js", implementing all features below without any extra prose.

index.html

Include CDNs:
- TensorFlow.js latest
- tfjs-vis latest  
- Chart.js for EDA visualizations
- Modern CSS with gradient backgrounds and responsive design

Three main sections with clear visual separation:
1. BUSINESS SOLUTION OVERVIEW - Market analysis, client showcase, value proposition
2. MODEL TRAINING & VALIDATION - Data upload, EDA, model controls, training logs
3. PRODUCTION DEPLOYMENT - Real-time classification, batch processing, export functionality

Advanced UI Components:
- File inputs for train/test/unlabeled data with validation
- Comprehensive model controls (Train, Evaluate, Test Samples, Save/Load, Reset, Visualization)
- Real-time training progress with accuracy metrics
- EDA dashboard with class distribution, pixel statistics, data quality checks
- Prediction preview with visual feedback (correct/incorrect indicators)
- Model architecture and performance metrics display
- Production classification interface with export capabilities

data-loader.js

Implement robust file parsing system:
- Handle both labeled (label + 784 pixels) and unlabeled (784 pixels) CSV formats
- Stream processing for large files with progress reporting
- Automatic data validation and error handling
- Tensor normalization (/255) and reshaping to [N,28,28,1]
- One-hot encoding for labels (depth=10)
- Memory-efficient tensor management with automatic disposal

Key functions:
- loadCSVFile(file): Universal parser for both labeled and unlabeled data
- loadTrainFromFiles(file), loadTestFromFiles(file): Specialized loaders
- loadUnlabeledFromFiles(file): For production classification data
- splitTrainVal(): Validation split with configurable ratio
- getRandomTestBatch(): Sampling for preview functionality
- draw28x28ToCanvas(): High-quality image rendering with scaling

app.js

Comprehensive application architecture:

Business Logic Layer:
- Three-phase workflow: Analysis → Training → Deployment
- Real-time EDA generation with Chart.js visualizations
- Advanced CNN model with configurable architecture
- Production-ready classification pipeline

Model Architecture (Optimized CNN):
- Multiple convolutional blocks with batch normalization
- Strategic dropout layers for regularization  
- Adaptive learning based on backend capabilities (WebGL/CPU)
- Progressive training with real-time metrics

UI Management:
- Dynamic progress reporting during training
- Interactive prediction previews with accuracy scoring
- Model performance dashboard with quality ratings
- Production classification with batch processing

Advanced Features:
- Automatic backend detection (WebGL fallback to CPU)
- Memory management with tensor disposal scopes
- Error recovery and user-friendly messaging
- Export functionality for classification results
- Model persistence with version tracking

Performance & Safety:
- Comprehensive error handling throughout pipeline
- Memory leak prevention with systematic tensor disposal
- Responsive UI with async/await patterns
- Production-grade reliability for business use

Formatting:
- Produce exactly three fenced code blocks labeled "index.html", "data-loader.js", and "app.js"
- Browser-compatible JavaScript with modern ES6+ features
- Detailed, business-focused comments explaining key algorithms
- Professional CSS with gradient designs and responsive layouts
- No external text outside the specified code blocks
