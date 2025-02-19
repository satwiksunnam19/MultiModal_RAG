import torch
import time
from PIL import Image
from qdrant_client import models
from qdrant_client import QdrantClient
from qwen_vl_utils import process_vision_info
from colpali_engine.models import ColPali, ColPaliProcessor
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info
import base64
from io import BytesIO
from tqdm import tqdm

def resize_image_once(image_path, new_path=None, size=(700, 2100)):
    """
    Resizes and saves a single image to new_path to avoid repeating
    the same operation on every query.
    """
    if not new_path:
        new_path = image_path  # Overwrite if no new path is given
    
    with Image.open(image_path) as img:
        img_resized = img.resize(size, Image.LANCZOS)
        img_resized.save(new_path)

def stream_response(response):
    """
    Remove or reduce time.sleep() to eliminate artificial slowing.
    """
    words = response[0].split()
    for word in words:
        yield word + " "

def batch_iterate(lst, batch_size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

class EmbedData:
    def __init__(self, embed_model_name="vidore/colpali-v1.2", batch_size=16):
        """
        Use a larger default batch_size for better throughput.
        """
        self.embed_model_name = embed_model_name
        self.embed_model, self.processor = self._load_embed_model()
        self.batch_size = batch_size
        self.embeddings = []

    def _load_embed_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embed_model = ColPali.from_pretrained(
            self.embed_model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True, 
            cache_dir="Qwen_MultiModal_RAG/hf_cache",
        )
        processor = ColPaliProcessor.from_pretrained(self.embed_model_name)
        return embed_model, processor

    def get_query_embedding(self, query):
        with torch.no_grad():
            query_input = self.processor.process_queries([query]).to(self.embed_model.device)
            query_embedding = self.embed_model(**query_input)
        return query_embedding[0].cpu().float().numpy().tolist()

    def generate_embedding(self, images):
        with torch.no_grad():
            batch_images = self.processor.process_images(images).to(self.embed_model.device)
            image_embeddings = self.embed_model(**batch_images).cpu().float().numpy().tolist()
        return image_embeddings

    def embed(self, images):
        self.images = images
        self.all_embeddings = []
        for batch_images in tqdm(batch_iterate(images, self.batch_size), desc="Generating embeddings"):
            batch_embeddings = self.generate_embedding(batch_images)
            self.embeddings.extend(batch_embeddings)

class QdrantVDB_QB:
    def __init__(self, collection_name, vector_dim=128, batch_size=16):
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.vector_dim = vector_dim

    def define_client(self):
        self.client = QdrantClient(url="http://localhost:6333", prefer_grpc=False)

    def create_collection(self):
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                on_disk_payload=True,
                vectors_config=models.VectorParams(
                    size=self.vector_dim,
                    distance=models.Distance.COSINE,
                    on_disk=True,
                    # Only set multiVectorConfig if needed
                    # multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM),
                ),
            )

    def ingest_data(self, embeddata):
        """
        Ingest in larger batches to reduce overhead.
        Optionally store file path or smaller preview instead of full base64.
        """
        idx = 0
        for batch_embeddings in tqdm(batch_iterate(embeddata.embeddings, self.batch_size), desc="Ingesting data"):
            points = []
            for j, embedding in enumerate(batch_embeddings):
                # Using image path or small thumbnail is more efficient than full base64
                # For demonstration, keep base64, but you can skip or store smaller size.
                image_bs64 = image_to_base64(embeddata.images[idx + j])
                current_point = models.PointStruct(
                    id=idx + j,
                    vector=embedding,
                    payload={"image": image_bs64},
                )
                points.append(current_point)

            self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
            idx += len(batch_embeddings)

class Retriever:
    def __init__(self, vector_db, embeddata):
        self.vector_db = vector_db
        self.embeddata = embeddata

    def search(self, query):
        query_embedding = self.embeddata.get_query_embedding(query)
        query_result = self.vector_db.client.query_points(
            collection_name=self.vector_db.collection_name,
            query=query_embedding,
            limit=4,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    ignore=True,
                    rescore=True,
                    oversampling=2.0
                )
            )
        )
        return query_result

class RAG:
    def __init__(self, retriever, llm_name="Qwen/Qwen2.5-VL-3B-Instruct"):
        self.llm_name = llm_name
        self._setup_llm()
        self.retriever = retriever

    def _setup_llm(self):
        """
        Remove force_download=True unless you actually need to redownload
        every time.
        """
        self.model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct",
            self.llm_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir="Qwen_MultiModal_RAG/hf_cache",
        )
        self.processor = AutoProcessor.from_pretrained(
            self.llm_name,
            cache_dir="Qwen_MultiModal_RAG/hf_cache",
        )

    def generate_context(self, query):
        result = self.retriever.search(query)
        return f"./images/page{result.points[0].id}.jpg"

    def query(self, query):
        image_context = self.generate_context(query=query)

        resize_image_once(image_context)  

        qa_prompt_tmpl_str = f"""The user has asked the following question:

        Query: {query}

        Some images are available to you for this question. You have
        to understand these images thoroughly and extract all relevant 
        information that will help you answer the query.
        """

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_context},
                    {"type": "text", "text": qa_prompt_tmpl_str},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False, streaming=True
        )
        return stream_response(output_text)
