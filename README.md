# Enhanced PDF Outline Extractor

A hybrid ML + Visual PDF outline extraction system with multilingual support for accurate heading detection and document structure analysis.

## Features

- **Hybrid Extraction**: Combines machine learning models with advanced visual analysis
- **Multilingual Support**: Native support for Hindi (Devanagari), English, and mixed-language documents  
- **Smart Quality Assessment**: Advanced confidence scoring and quality-based filtering
- **Form Field Detection**: Automatically filters out form fields and irrelevant content
- **Clean JSON Output**: Standardized schema with title and hierarchical outline structure

## Docker Requirements & Execution

### Build Command
```bash
docker build --platform linux/amd64 -t pdf-extractor:latest .
```

### Run Command  
```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none pdf-extractor:latest
```

### Container Specifications
- **Architecture**: AMD64 (x86_64) compatible
- **Network**: Offline operation (no internet calls)
- **Model Size**: ≤ 200MB
- **Input**: `/app/input` directory containing PDF files
- **Output**: `/app/output` directory with corresponding JSON files

The container automatically processes all PDFs from the input directory and generates `filename.json` for each `filename.pdf`.

## Output Format

The system generates JSON files with the following schema:

```json
{
  "title": "Document Title",
  "outline": [
    {"level": "H1", "text": "Chapter 1", "page": 0},
    {"level": "H2", "text": "Introduction", "page": 1}
  ]
}
```

Where:
- `title`: Extracted document title
- `outline`: Array of heading objects with level (H1, H2, H3, etc.), text content, and 0-indexed page number

## Technical Approach

### Hybrid Processing Pipeline
1. **ML Extraction**: Feature-engineered machine learning model for heading classification
2. **Visual Analysis**: Advanced font and layout analysis for structure detection  
3. **Multilingual Processing**: Language detection and script-specific processing
4. **Quality Assessment**: Confidence scoring and intelligent filtering
5. **Output Generation**: Schema-compliant JSON with hierarchical structure

### Multilingual Support
- **Hindi (Devanagari)**: Full support with form field filtering
- **English (Latin)**: Complete processing capabilities
- **Mixed Documents**: Bilingual document support
- **Language Detection**: Automatic script identification

## Dependencies

All required dependencies are listed in `requirements.txt` and will be automatically installed during Docker build.
