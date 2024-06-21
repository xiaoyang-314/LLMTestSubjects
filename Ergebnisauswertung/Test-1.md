## Satzpaare Analyse

### Paar 1
**Sätze:**
1. A man is slicing potatoes.
2. A woman is peeling potato.

**Bewertungen:**
- Menschlicher Ähnlichkeitswert: 2,2
- Kosinus-Ähnlichkeit: 0,6808

---

**Sätze:**
1. A woman is slicing an onion.
2. A woman is cutting a potato.

**Bewertungen:**
- Menschlicher Ähnlichkeitswert: 1,2
- Kosinus-Ähnlichkeit: 0,6758

### Paar 2
**Sätze:**
1. A man is riding a horse.
2. A girl is riding a horse.

**Bewertungen:**
- Menschlicher Ähnlichkeitswert: 2,6
- Kosinus-Ähnlichkeit: 0,6420

---

**Sätze:**
1. A man is riding a bicycle.
2. A monkey is riding a bike.

**Bewertungen:**
- Menschlicher Ähnlichkeitswert: 2,0
- Kosinus-Ähnlichkeit: 0,6331

### Paar 3
**Sätze:**
1. The man is playing the guitar.
2. A man is playing the flute.

**Bewertungen:**
- Menschlicher Ähnlichkeitswert: 1,643
- Kosinus-Ähnlichkeit: 0,5536

---

**Sätze:**
1. The man is playing the piano.
2. The man is playing the violin.

**Bewertungen:**
- Menschlicher Ähnlichkeitswert: 1,75
- Kosinus-Ähnlichkeit: 0,5758

### Paar 4
**Sätze:**
1. A woman is adding spices on a meat.
2. Woman is adding sugar to meat.

**Bewertungen:**
- Menschlicher Ähnlichkeitswert: 2,75
- Kosinus-Ähnlichkeit: 0,7888

---

**Sätze:**
1. A woman is adding oil on fishes.
2. A woman is pouring water on fish.

**Bewertungen:**
- Menschlicher Ähnlichkeitswert: 2,25
- Kosinus-Ähnlichkeit: 0,7996

### Paar 5
**Sätze:**
1. Two dogs in a yard.
2. Two dogs in snow.

**Bewertungen:**
- Menschlicher Ähnlichkeitswert: 3,0
- Kosinus-Ähnlichkeit: 0,6883

---

**Sätze:**
1. Two dogs playing in the snow.
2. Two black dogs in the snow.

**Bewertungen:**
- Menschlicher Ähnlichkeitswert: 4,0
- Kosinus-Ähnlichkeit: 0,8186

### Paar 6
**Sätze:**
1. Two dogs swim in a pool.
2. Dogs are swimming in a pool.

**Bewertungen:**
- Menschlicher Ähnlichkeitswert: 4,2
- Kosinus-Ähnlichkeit: 0,8813

---

**Sätze:**
1. A dog is looking into swimming pool.
2. A dog is walking around a pool.

**Bewertungen:**
- Menschlicher Ähnlichkeitswert: 2,0
- Kosinus-Ähnlichkeit: 0,7766

### Paar 7
**Sätze:**
1. Two people sitting at a table at a restaurant.
2. Four people sitting at a table.

**Bewertungen:**
- Menschlicher Ähnlichkeitswert: 1,8
- Kosinus-Ähnlichkeit: 0,7163

---

**Sätze:**
1. A group of people sitting at a restaurant table.
2. A group of people sitting around a table with food on it.

**Bewertungen:**
- Menschlicher Ähnlichkeitswert: 4,2
- Kosinus-Ähnlichkeit: 0,8603

### Paar 8
**Sätze:**
1. A bird lands in the water.
2. A boat floats in the water.

**Bewertungen:**
- Menschlicher Ähnlichkeitswert: 0,6
- Kosinus-Ähnlichkeit: 0,5943

---

**Sätze:**
1. The seagull is floating on the water.
2. Large cruise ship floating on the water.

**Bewertungen:**
- Menschlicher Ähnlichkeitswert: 1,2
- Kosinus-Ähnlichkeit: 0,5451

### Paar 9
**Sätze:**
1. A boy with a broken arm is resting in a bed.
2. A boy with a broken arm is sleeping.

**Bewertungen:**
- Menschlicher Ähnlichkeitswert: 4,0
- Kosinus-Ähnlichkeit: 0,8751

---

**Sätze:**
1. A woman is resting in a floating raft.
2. A woman relaxes in an tube.

**Bewertungen:**
- Menschlicher Ähnlichkeitswert: 3,8
- Kosinus-Ähnlichkeit: 0,7153

### Paar 10
**Sätze:**
1. A man is looking at a computer monitor.
2. A man is using a laptop.

**Bewertungen:**
- Menschlicher Ähnlichkeitswert: 2,75
- Kosinus-Ähnlichkeit: 0,7372

---

**Sätze:**
1. Boy with glasses typing on a computer.
2. A boy looking at a computer screen.

**Bewertungen:**
- Menschlicher Ähnlichkeitswert: 3,2
- Kosinus-Ähnlichkeit: 0,7958

---

![output](https://github.com/xiaoyang-314/ObjectSemantischeAehnlichkeit/assets/170884230/164e2c40-b573-4a70-8c4d-25f0ca64fb4c)

### Quantitative Analysis

1. **Normalization**:
   - Die menschlichen Ähnlichkeitswerte reichen von 0,6 bis 4,2, und die Kosinus-Ähnlichkeitswerte reichen von etwa 0,5451 bis 0,8813.
   - Durch die Normalisierung werden beide Wertemengen auf einen Bereich von 0 bis 1 skaliert, was einen direkten Vergleich ermöglicht.

2. **Korrelationsanalyse**:
   - Die normalisierten menschlichen Ähnlichkeitswerte und die Kosinus-Ähnlichkeitswerte können verglichen werden, um zu sehen, wie gut sie korrelieren. Der berechnete Korrelationskoeffizient beträgt **0.793**, was auf eine starke positive Korrelation hinweist.

### Berechnungsprozess

1. **Normalisierung**:
   - Zuerst wurden die menschlichen Ähnlichkeitswerte und die Kosinus-Ähnlichkeitswerte mit Hilfe des MinMaxScaler aus der Bibliothek sklearn normalisiert.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# Fit and transform the data
df[['Human Similarity Normalized', 'Cosine Similarity Normalized']] = scaler.fit_transform(df[['Human Similarity', 'Cosine Similarity']])
```

2. **Korrelationskoeffizient**:
   - Danach wurde der Korrelationskoeffizient zwischen den normalisierten Werten berechnet.

```python
import numpy as np

# Calculate correlation coefficient
correlation = np.corrcoef(df['Human Similarity Normalized'], df['Cosine Similarity Normalized'])[0, 1]
correlation
```

   - Der berechnete Korrelationskoeffizient beträgt **0.793**.

### Qualitative Analysis

1. **Trends und Muster**:
   - Generell weisen Paare mit höheren menschlichen Ähnlichkeitswerten auch höhere Kosinus-Ähnlichkeitswerte auf und umgekehrt. Dies deutet darauf hin, dass die Kosinus-Ähnlichkeit in gewissem Maße mit der menschlichen Einschätzung übereinstimmt.
   - Bestimmte Paare, wie „Two dogs swim in a pool“ und „Dogs are swimming in a pool“, erhielten hohe Ähnlichkeitswerte von sowohl Menschen als auch der Kosinus-Ähnlichkeit, was auf eine starke Übereinstimmung hinweist.

2. **Abweichungen**:
   - Einige Satzpaare zeigen Abweichungen zwischen menschlichen und Kosinus-Ähnlichkeitswerten. Zum Beispiel hat „A man is slicing potatoes“ vs. „A woman is peeling potato“ einen relativ niedrigen menschlichen Ähnlichkeitswert (2,2), aber einen hohen Kosinus-Ähnlichkeitswert (0,6808). Dies könnte darauf zurückzuführen sein, dass die Kosinus-Ähnlichkeit lexikalische oder syntaktische Ähnlichkeiten erfasst, die Menschen übersehen oder als weniger wichtig erachten.

3. **Kontextuelle Unterschiede**:
   - Menschliche Ähnlichkeitsbewertungen berücksichtigen oft den Kontext und ein tieferes semantisches Verständnis. Zum Beispiel haben „A man is riding a bicycle“ vs. „A monkey is riding a bike“ einen menschlichen Ähnlichkeitswert von 2,0, aber einen Kosinus-Ähnlichkeitswert von 0,6331. Der Unterschied in den Subjekten (man vs. monkey) ist für Menschen bedeutender als für die Kosinus-Ähnlichkeit, die möglicherweise mehr auf die Ähnlichkeit von Verb und Objekt achtet.

### Zusammenfassung

- **Korrelation**: Es gibt eine positive Korrelation zwischen den normalisierten menschlichen Ähnlichkeitswerten und den Kosinus-Ähnlichkeitswerten, was darauf hinweist, dass die Kosinus-Ähnlichkeit als vernünftiger Stellvertreter für menschliche Einschätzungen dienen kann, obwohl sie nicht vollständig übereinstimmen.
- **Muster und Abweichungen**: Menschliche Einschätzungen berücksichtigen den Kontext und die Semantik tiefer, was zu einigen Abweichungen führt, während die Kosinus-Ähnlichkeit mehr auf lexikalische und syntaktische Ähnlichkeiten angewiesen ist.

### Anwendung auf Modell „mxbai-embed-large-v1“

Das Ziel dieses Experiments ist es, die semantische Ähnlichkeitsfunktion des Modells „mxbai-embed-large-v1“ zu testen. Dabei wurde festgestellt, dass das Modell in vielen Fällen menschliche Ähnlichkeitsbewertungen gut widerspiegelt, es jedoch in bestimmten kontextabhängigen Fällen zu Abweichungen kommt. Die starke Korrelation zwischen den normalisierten Werten legt nahe, dass das Modell grundsätzlich eine gute semantische Ähnlichkeitsfunktion bietet, jedoch möglicherweise in der Feinabstimmung der kontextuellen und semantischen Unterschiede verbessert werden könnte.
