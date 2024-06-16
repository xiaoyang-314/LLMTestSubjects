from mixedbread_ai.client import MixedbreadAI
import matplotlib.pyplot as plt
import numpy as np

mxbai = MixedbreadAI(api_key="emb_39818b04d52147ef0840e2f3fe42a5157c0e0a941990e922")

english_sentences = [
    'A man is looking at a computer monitor.',
    'A man is using a laptop.',
    'Boy with glasses typing on a computer.',
    'A boy looking at a computer screen.'
]

res = mxbai.embeddings(
    input=english_sentences,
    model="mixedbread-ai/mxbai-embed-large-v1",
    normalized=True,
    dimensions=512
)

# Einbettungsvektoren abrufen und in NumPy-Arrays konvertieren
encoded_embeddings = np.array([item.embedding for item in res.data])
print(res.dimensions, encoded_embeddings)
