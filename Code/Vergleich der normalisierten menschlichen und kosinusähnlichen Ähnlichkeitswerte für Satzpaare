import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Datenvorbereitung
data = {
    'Pair': [
        'Pair 1a', 'Pair 1b', 'Pair 2a', 'Pair 2b', 'Pair 3a', 'Pair 3b', 
        'Pair 4a', 'Pair 4b', 'Pair 5a', 'Pair 5b', 'Pair 6a', 'Pair 6b',
        'Pair 7a', 'Pair 7b', 'Pair 8a', 'Pair 8b', 'Pair 9a', 'Pair 9b', 
        'Pair 10a', 'Pair 10b'
    ],
    'Human Similarity': [
        2.2, 1.2, 2.6, 2.0, 1.643, 1.75, 2.75, 2.25, 3.0, 4.0, 4.2, 2.0,
        1.8, 4.2, 0.6, 1.2, 4.0, 3.8, 2.75, 3.2
    ],
    'Cosine Similarity': [
        0.6808, 0.6758, 0.6420, 0.6331, 0.5536, 0.5758, 0.7888, 0.7996, 0.6883, 0.8186, 
        0.8813, 0.7766, 0.7163, 0.8603, 0.5943, 0.5451, 0.8751, 0.7153, 0.7372, 0.7958
    ]
}

df = pd.DataFrame(data)

# Normalisierung der Werte, um sie auf eine gemeinsame Skala zu bringen
scaler = MinMaxScaler()

# Daten anpassen und transformieren
df[['Human Similarity Normalized', 'Cosine Similarity Normalized']] = scaler.fit_transform(df[['Human Similarity', 'Cosine Similarity']])

# Diagramm der normalisierten Werte
plt.figure(figsize=(12, 8))
plt.plot(df['Pair'], df['Human Similarity Normalized'], marker='o', linestyle='-', color='b', label='Human Similarity (Normalized)')
plt.plot(df['Pair'], df['Cosine Similarity Normalized'], marker='s', linestyle='--', color='r', label='Cosine Similarity (Normalized)')

plt.xlabel('Satzpaare')
plt.ylabel('Normalisierte Ähnlichkeitswerte')
plt.title('Vergleich der normalisierten menschlichen und kosinusähnlichen Ähnlichkeitswerte für Satzpaare')
plt.xticks(rotation=90)
plt.legend()
plt.grid(True)

# Diagramm anzeigen
plt.tight_layout()
plt.show()

import ace_tools as tools; tools.display_dataframe_to_user(name="Normalized Similarity Scores", dataframe=df)
