from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sentence_transformers.quantization import quantize_embeddings

# 1. Modell und Tokenizer laden
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# Beispielabfrage und Dokumente
query = 'Represent this sentence for searching relevant passages: A man is slicing potatoes'
docs = [
    query,
    "A woman is peeling potato.",
    "A woman is slicing an onion.",
    "A woman is cutting a potato.",
]

# 2. Encode
embeddings = model.encode(docs)

# Optional: Quantize the embeddings
binary_embeddings = quantize_embeddings(embeddings, precision="ubinary")

# Ähnlichkeiten berechnen
similarities = cos_sim(embeddings[0], embeddings[1:])
print('similarities:', similarities)
