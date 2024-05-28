
## Aufgabe:

Ziel dieses Projekts ist, die Funktionalität des Modells `mxbai-embed-large-v1` bei der semantischen Suche zu testen.

### Test 1:

Ein Beispiel für eine zu testende Aufgabe wäre eine Abfrage wie „A man is eating a piece of bread“.  Es soll herausgefunden werden, basierend auf der Cosine Similarity, welcher der folgenden Sätze der Abfrage semantisch am ähnlichsten ist:
- 'A man is eating food.'
- 'A man is eating pasta.'
- 'The girl is carrying a baby.'
- 'A man is riding a horse.'

### Test 2:

Das Ziel dieser Aufgabe ist es, die eingebetteten Vektoren von zwei (oder mehreren) zu vergleichenden Sätzen zu erhalten. Zum Beispiel die folgenden Sätze:
- 'What is the capital of Australia?'
- 'Canberra is the capital of Australia.'

## Datenbasis:

STS Benchmark ist ein Standarddatensatz zur Bewertung der semantischen Textähnlichkeit, der viele Satzpaare mit von Menschen kommentierten Ähnlichkeitswerten enthält.

- **Medium:** Text (Satzpaare)
- **Sprache:** Englisch
- **Quelle:** [STS Benchmark](https://huggingface.co/datasets/mteb/stsbenchmark-sts?row=19)

## Modell:

- **OpenLLM:** Verwendetes Modell: `mxbai-embed-large-v1`

## Experimentdesign

### Vorgehensweise:

#### Test 1:

1. Sätze sammeln und in einer Textdatei speichern: Auswahl von 10 Abfragen und zu vergleichenden Dokumenten aus dem STS Benchmark.
2. Verwendung von `sentence_transformers`:
   - Eingabe der Abfragen mit einem spezifischen Prompt.
3. Berechnung der Cosine Similarity für jeden Satz und die Abfrage.

#### Test 2:

1. Sätze sammeln und in einer Textdatei speichern: Auswahl von 10 Satzpaaren und Hinzufügen der zu testenden Sätze in eine Liste.
2. Verwendung des Modells via API, um die eingebetteten Vektoren für jeden Satz zu erhalten und diese in einem NumPy-Array zu speichern.

### Ergebnisse beurteilen:

#### Test 1:

Vergleich der Cosine Similarity (-1 bis 1) mit der manuellen Bewertung des STS Benchmarks (Score 0 bis 5). Analyse der Unterschiede in den Ergebnissen.

#### Test 2:

Visualisierung:
- Verwendung von Matplotlib, um die eingebetteten Vektoren aller Sätze auf einem einzigen Diagramm darzustellen, um ihre eingebetteten Merkmale zu vergleichen.
- Die eingebetteten Vektoren jedes Satzes werden in unterschiedlichen Farben und mit Satzlabels dargestellt, um die Unterschiede visuell zu erkennen.
- Dies ermöglicht die Analyse und Visualisierung der eingebetteten Vektoren mehrerer Sätze.

