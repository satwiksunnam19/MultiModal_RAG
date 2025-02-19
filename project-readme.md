# Multimodal RAG with Qwen2.5-VL

A Streamlit-based application that implements Multimodal Retrieval-Augmented Generation (RAG) using Qwen2.5-VL for processing PDF documents and answering queries about their content.

## Features

- PDF document upload and processing
- Image-based document indexing using ColPaLI embeddings
- Vector similarity search using Qdrant
- Natural language question answering with Qwen2.5-VL
- Interactive chat interface with streaming responses
- Efficient batch processing of document pages

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Qdrant server running locally on port 6333

## Installation

1. Clone the repository:
```bash
git clone git@github.com:satwiksunnam19/MultiModal_RAG.git
cd MultiModal_RAG
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure you have a Qdrant instance running locally:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

## Project Structure

- `app.py`: Main Streamlit application file
- `rag.py`: Core RAG implementation including embedding, retrieval, and generation
- `requirements.txt`: Project dependencies

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Upload a PDF document using the sidebar interface

3. Wait for the document to be processed and indexed

4. Start asking questions about the document content in the chat interface

## Technical Details

### Components

1. **Document Processing**
   - Converts PDF pages to images
   - Resizes images for optimal processing
   - Maintains a session-based file cache

2. **Embedding System**
   - Uses ColPaLI v1.2 for generating image embeddings
   - Processes images in batches for efficiency
   - Supports both image and text query embeddings

3. **Vector Database**
   - Uses Qdrant for storing and retrieving embeddings
   - Implements cosine similarity search
   - Supports on-disk storage for large documents

4. **Language Model**
   - Utilizes Qwen2.5-VL for multimodal understanding
   - Processes both images and text in context
   - Generates natural language responses

### Performance Optimizations

- Batch processing for embedding generation
- Efficient image resizing with caching
- Session-based document caching
- Streaming responses for better user experience

## Configuration

The application uses several configurable parameters:

- `collection_name`: Name of the Qdrant collection
- `vector_dim`: Dimension of the embedding vectors (default: 128)
- `batch_size`: Size of batches for processing (default: 16)

## Limitations

- Currently supports only PDF documents
- Requires local Qdrant instance
- Memory usage scales with document size
- Requires CUDA-capable GPU for optimal performance

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.


## Acknowledgments

- Qwen team for the Qwen2.5-VL model
- ColPaLI team for the embedding model
- Qdrant team for the vector database
- Streamlit team for the web framework
