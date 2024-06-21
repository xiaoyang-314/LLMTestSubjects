import matplotlib.pyplot as plt
import pandas as pd

# Datenvorbereitung mit gruppierten Sätzen
data = {
    'Group': [
        'Group 1', 'Group 1', 'Group 1', 
        'Group 2', 'Group 2', 'Group 2', 
        'Group 3', 'Group 3', 'Group 3', 
        'Group 4', 'Group 4', 'Group 4', 
        'Group 5', 'Group 5', 'Group 5', 
        'Group 6', 'Group 6', 'Group 6', 
        'Group 7', 'Group 7', 'Group 7', 
        'Group 8', 'Group 8', 
        'Group 9', 'Group 9', 'Group 9', 
        'Group 10', 'Group 10', 'Group 10'
    ],
    'Sentence': [
        'A woman is peeling potato.', 
        'A woman is slicing an onion.', 
        'A woman is cutting a potato.', 
        'A girl is riding a horse.', 
        'A man is riding a bicycle.', 
        'A monkey is riding a bike.', 
        'A man is playing the flute.', 
        'The man is playing the piano.', 
        'The man is playing the violin.', 
        'Woman is adding sugar to meat.', 
        'A woman is adding oil on fishes.', 
        'A woman is pouring water on fish.', 
        'Two dogs in snow.', 
        'Two dogs playing in the snow.', 
        'Two black dogs in the snow.', 
        'Dogs are swimming in a pool.', 
        'A dog is looking into swimming pool.', 
        'A dog is walking around a pool.', 
        'Four people sitting at a table.', 
        'A group of people sitting at a restaurant table.', 
        'A group of people sitting around a table with food on it.', 
        'A boat floats in the water.', 
        'The seagull is floating on the water.', 
        'A boy with a broken arm is sleeping.', 
        'A woman is resting in a floating raft.', 
        'A woman relaxes in an tube.', 
        'A man is using a laptop.', 
        'Boy with glasses typing on a computer.', 
        'A boy looking at a computer screen.'
    ],
    'Similarity': [
        0.6808, 0.5519, 0.7610, 0.6420, 0.5100, 0.3809, 0.5536, 0.5801, 0.5576,
        0.7888, 0.6877, 0.6072, 0.6883, 0.6905, 0.6661, 0.6808, 0.5519, 0.7610,
        0.7345, 0.8561, 0.7893, 0.7984, 0.6542, 0.8856, 0.4221, 0.3209, 0.7981, 
        0.5952, 0.7114
    ]
}

df = pd.DataFrame(data)

# Sortieren der Daten nach Ähnlichkeit innerhalb jeder Gruppe
df_sorted = df.sort_values(by=['Group', 'Similarity'], ascending=[True, False])

# Sicherstellen der Gruppenreihenfolge von 1 bis 10
group_order = [f'Group {i}' for i in range(1, 11)]
df_sorted['Group'] = pd.Categorical(df_sorted['Group'], categories=group_order, ordered=True)
df_sorted = df_sorted.sort_values(by=['Group', 'Similarity'], ascending=[True, False])

# Plotten mit sortierten Sätzen
plt.figure(figsize=(14, 10))

# Originalsätze für jede Gruppe
original_sentences = {
    'Group 1': 'A man is slicing potatoes.',
    'Group 2': 'A man is riding a horse.',
    'Group 3': 'The man is playing the guitar.',
    'Group 4': 'A woman is adding spices on a meat.',
    'Group 5': 'Two dogs in a yard.',
    'Group 6': 'Two dogs swim in a pool.',
    'Group 7': 'Two people sitting at a table at a restaurant.',
    'Group 8': 'A bird lands in the water.',
    'Group 9': 'A boy with a broken arm is resting in a bed.',
    'Group 10': 'A man is looking at a computer monitor.'
}

groups = df_sorted.groupby('Group')
for name, group in groups:
    plt.plot(group['Sentence'], group['Similarity'], marker='o', linestyle='-', label=f"{name}: {original_sentences[name]}")

plt.xlabel('Sentences')
plt.ylabel('Similarity Scores')
plt.title('Sorted Similarity Scores for Sentence Pairs using mxbai-embed-large-v1')
plt.xticks(rotation=90)
plt.legend(title='Groups', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Layout anpassen, um Platz für die Legende zu schaffen
plt.tight_layout(rect=[0, 0, 0.85, 1])

# Diagramm anzeigen
plt.show()
