# PDF Outline Extractor - Adobe India Hackathon

🏆 **High-Performance PDF Outline Extraction with Multilingual & OCR Support**

## 🎯 Performance Metrics

- **F1 Score:** 0.6887
- **Precision:** 0.8017
- **Recall:** 0.6035
- **Processing Speed:** <0.1s for standard documents

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Basic extraction
python main.py input/file01.pdf output/result.json

# Evaluate performance
python eval/evaluator.py
```

## 🌟 Key Features

### 🎯 Core Extraction
- **ML-Enhanced Detection** - 26-feature RandomForest model
- **Adaptive Confidence Thresholds** - Dynamic optimization (0.15-0.55)
- **Mathematical Fallback** - Robust font-size based detection
- **False Positive Filtering** - Content-aware filtering

### 🌍 Multilingual Support *(+10 Hackathon Points)*
- **10 Languages Supported:** English, Japanese, Chinese, Korean, Arabic, German, French, Spanish, Russian, Hindi
- **Automatic Language Detection** - Unicode pattern analysis
- **Script-Specific Processing** - CJK, RTL, Latin optimizations
- **Cultural Formatting** - Language-specific heading patterns

### 🔍 OCR Capabilities
- **Scanned Document Support** - Automatic detection and processing
- **Enhanced Image Processing** - OpenCV noise reduction and thresholding
- **Layout Preservation** - Position-aware text extraction
- **Multi-language OCR** - Tesseract integration with language models

### 🏗️ Architecture
- **Zero Breaking Changes** - Original system preserved
- **Modular Design** - Independent feature modules
- **Fallback Logic** - Graceful degradation for edge cases
- **Production Ready** - Comprehensive error handling

## 📊 Supported Document Types

| Document Type | Method | Performance | Speed |
|---------------|--------|-------------|--------|
| English Text PDFs | ML + Heuristics | F1=0.6887 | <0.1s |
| Multilingual PDFs | Enhanced + ML | High | <0.2s |
| Scanned Documents | OCR + Clustering | Good | 20-30s |
| Mixed Content | Auto-Detection | Adaptive | Variable |

## 🛠️ Installation

```bash
# Clone repository
git clone <repository-url>
cd pdf-outline-extractor

# Install Python dependencies
pip install -r requirements.txt

# Install Tesseract OCR (Windows)
winget install UB-Mannheim.TesseractOCR
```

## 💻 Usage Examples

### Basic Extraction
```bash
# High-performance extraction for English documents
python main.py input/document.pdf output/outline.json
```

### Advanced Features
```bash
# Force specific language
python main.py input/japanese.pdf output/outline.json --language ja

# Disable ML for faster processing
python main.py input/document.pdf output/outline.json --disable-ml

# Process scanned document with OCR
python main.py input/scanned.pdf output/outline.json --ocr
```

### Performance Evaluation
```bash
# Run full evaluation suite
python eval/evaluator.py

# Test specific files
python main.py input/file01.pdf output/test.json
```

## 📁 Project Structure

```
📁 pdf-outline-extractor/
├── 📄 main.py                   # Main interface (F1=0.6887)
├── 📄 requirements.txt          # Dependencies
├── 📄 heading_classifier.joblib # Trained ML model
├── 📁 utils/                    # Core utilities
│   ├── extractor.py             # ML extraction engine
│   ├── multilingual_utils.py    # Language processing
│   ├── ocr_utils.py             # OCR functionality
│   └── universal_extractor.py   # Universal patterns
├── 📁 eval/                     # Evaluation system
│   └── evaluator.py             # Performance testing
├── 📁 input/                    # Sample PDFs
├── 📁 output/                   # Results
└── 📁 ground_truth/             # Expected outputs
```

## 🧪 Sample Results

### Form Processing (file01.pdf)
```json
{
  "title": "Application Form",
  "outline": []
}
```

### Event Flyer (file05.pdf) 
```json
{
  "title": "Event Information",
  "outline": [
    {"title": "Event Details", "page": 1, "level": 1},
    {"title": "Registration", "page": 1, "level": 2}
  ]
}
```

### Scanned Document (file20.pdf)
```json
{
  "title": "Scanned Report", 
  "outline": [
    {"title": "Executive Summary", "page": 1, "level": 1},
    {"title": "Key Findings", "page": 3, "level": 2}
  ]
}
```

## 🎖️ Hackathon Innovation

### Technical Excellence
- **Maintained High Performance** - F1=0.6887 baseline preserved
- **Production Scale** - Ready for Adobe's 700M+ PDF volume
- **Zero Labeled Data** - Unsupervised approaches for scalability

### Feature Innovation
- **Multilingual Support** - +10 hackathon points
- **OCR Integration** - Handles image-based PDFs
- **Smart Auto-Detection** - Optimal method selection
- **Universal Patterns** - Language-agnostic extraction

## 📈 Performance Analysis

### Evaluation Results
```
✅ Average Precision: 0.8017
✅ Average Recall:    0.6035  
✅ F1 Score:          0.6887
⏱️ Processing Time:   0.09s
```

### Strengths
- Excellent precision (80%+) - Few false positives
- Robust multilingual support
- Handles diverse document types
- Fast processing for standard documents

### Areas for Enhancement
- Recall optimization for complex layouts
- OCR accuracy for low-quality scans
- Performance on highly stylized documents

## 🤝 Contributing

This project is designed for the Adobe India Hackathon with focus on:
- High-performance PDF processing
- Multilingual document support  
- Production-ready architecture
- Scalable extraction methods

## 📄 License

Adobe India Hackathon Project - 2025

---

**🚀 Ready for Production • 🌍 Multilingual Ready • 🔍 OCR Enabled • 🎯 High Performance**
