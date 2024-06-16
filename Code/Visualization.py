# Einbettungsvektoren abrufen und in NumPy-Arrays konvertieren
encoded_embeddings = np.array([item.embedding for item in res.data])
print(res.dimensions, encoded_embeddings)

# Visualisierung der Einbettungsvektoren
plt.figure(figsize=(10, 6))
for i, embedding in enumerate(encoded_embeddings):
    plt.plot(range(512), embedding, label=f'Sentence {i+1}: {english_sentences[i]}')
plt.xlabel('Embedding Dimension')
plt.ylabel('Value')
plt.title('Visualization of Embedding Vectors')
plt.legend()
plt.grid(True)
plt.show()
