# Estensione NLP di DÆLTA

Questa cartella raccoglie l'estensione NLP del progetto DÆLTA e gli script
operativi collegati alla generazione delle immagini attaccate. L'estensione
non modifica l'attacco DÆLTA né il punteggio ufficiale AADD: aggiunge una
lettura semantica separata del cambiamento introdotto dall'attacco.

L'idea è confrontare, per ogni coppia, l'immagine fake originale con la sua
versione `attacked` e verificare quali attributi visivi continuano a essere
riconosciuti da un modello visione-linguaggio.

## In breve

`nlp_semantic_eval.py` usa **Qwen2.5-VL-3B-Instruct** sullo stesso prompt per
annotare separatamente le due immagini. Per ciascuna produce una caption
neutra in inglese e cinque attributi controllati:

- sorriso visibile;
- occhiali;
- posa della testa;
- copricapo;
- tipo di sfondo.

~~~text
original fake ─┐
               ├─ Qwen + stesso prompt ─→ caption neutra + 5 attributi
attacked image ─┘
                                        └─ confronto per coppia ─→ metriche
~~~

Le metriche restano separate e non vengono combinate con lo score AADD:

| Metrica | Cosa misura |
| --- | --- |
| **Semantic Consistency** | Quota di attributi chiari dell'originale che rimangono uguali nell'immagine attacked. Una coppia entra nella media solo con almeno 3 attributi valutabili. |
| **BERTScore F1** | Somiglianza semantico-linguistica tra caption attacked e caption originale, calcolata con `roberta-large`. |
| **Concept Retention** | Sovrapposizione dei concetti lessicali rilevanti nelle due caption; è una diagnostica complementare, non una misura di qualità visiva. |

Un BERTScore alto non implica che tutti gli attributi categoriali siano
conservati: proprio questo divario è uno dei risultati dell'esperimento.

## Risultato di riferimento

L'output da usare per relazione, presentazione e paper è
[`nlp_semantic_results-All`](nlp_semantic_results-All/). Il confronto completo
copre **1.600 coppie** e ha prodotto:

| Risultato | Valore |
| --- | ---: |
| Semantic Consistency | **0,6062** su 1.057 coppie valutabili |
| BERTScore F1 | **0,9507** su 1.508 coppie con caption valide |
| Concept Retention | **0,2612** sulle stesse 1.508 coppie |

Gli attributi meno conservati sono `background_type` (**46,7%**) e
`head_pose` (**57,6%**); `eyeglasses` (**95,7%**) e `headwear` (**78,1%**) sono
più stabili. Questo indica che il soggetto e il contenuto generale restano
spesso descrivibili in modo simile, mentre posa e sfondo sono più sensibili
all'attacco.

La prima elaborazione completa ha richiesto **44 ore, 50 minuti e 7 secondi**
su una NVIDIA GeForce RTX 2060 da 6 GB con offloading su CPU. Durante quella
run, 66 coppie a 1792 px hanno causato un errore di memoria GPU; sono state
recuperate senza rieseguire le annotazioni valide, ridimensionando solo
l'input del VLM in memoria a 1024 px. I file sorgente non sono stati
modificati.

## File principali

| File | Ruolo |
| --- | --- |
| [`nlp_semantic_eval.py`](nlp_semantic_eval.py) | Script principale e corrente: annota immagini, usa la cache JSONL, calcola le tre metriche e salva i risultati finali. Include audit greedy opzionale e recupero sicuro delle annotazioni fallite. |
| [`requirements-nlp.txt`](requirements-nlp.txt) | Dipendenze aggiuntive dell'estensione NLP (`transformers`, `accelerate`, `bert-score`). |
| [`compute_bertscore_from_cache.py`](compute_bertscore_from_cache.py) | Utility delle fasi iniziali: ricostruisce metriche da cache già esistenti senza interrogare Qwen. È utile come traccia sperimentale, ma non genera l'output di riferimento corrente. |
| [`train-ae-daelta.py`](train-ae-daelta.py) | Training dell'autoencoder dual-branch DÆLTA, con rami RGB e DCT. Non fa parte della metrica NLP. |
| [`attack-ae-daelta.py`](attack-ae-daelta.py) | Inferenza dell'attacco DÆLTA: genera le immagini attacked a partire dalle fake. Non calcola metriche NLP. |
| [`AgiuntePaper.md`](AgiuntePaper.md) | Guida alle aggiunte NLP inserite nel paper e alla loro collocazione. |

## Output degli esperimenti

| Cartella | Stato e contenuto |
| --- | --- |
| [`nlp_semantic_results-All`](nlp_semantic_results-All/) | **Output di riferimento.** Contiene `pairs.csv`, `summary.json` e la cache completa delle annotazioni. |
| [`nlp_semantic_results-gready`](nlp_semantic_results-gready/) | Pilot di 20 coppie e audit di ripetibilità greedy. Il nome contiene un refuso storico: il test è *greedy*, non “gready”. |
| [`nlp_semantic_results-test`](nlp_semantic_results-test/) | Test iniziali sulle immagini originali: prompt, JSON e qualità delle caption. |
| [`nlp_semantic_results-attacked`](nlp_semantic_results-attacked/) | Annotazioni iniziali delle sole immagini attacked. |
| [`nlp_semantic_results-paired`](nlp_semantic_results-paired/) | Precedente confronto creato dall'utility da cache; utile come storico, non come risultato finale. |
| `__pycache__/` | Cache automatica di Python; può essere ignorata. |

Nell'output di riferimento:

- `pairs.csv` contiene una riga per coppia, caption, attributi, metriche e
  stato di valutazione;
- `summary.json` contiene medie, denominatori, risultati per attributo e
  configurazione di inferenza;
- `annotations_*.jsonl` conserva le annotazioni elementari e permette di
  riprendere una run senza rifare le inferenze già riuscite.

## Esecuzione

Dopo avere preparato l'ambiente base AADD, installare le dipendenze NLP:

~~~powershell
pip install -r .\requirements-nlp.txt
~~~

La sezione `QUICK CONFIGURATION` all'inizio di
[`nlp_semantic_eval.py`](nlp_semantic_eval.py) definisce cartelle di input,
output e numero di immagini. Con la configurazione corrente, dalla cartella
`NLP` il comando è:

~~~powershell
python .\nlp_semantic_eval.py
~~~

La cache evita di ricalcolare le annotazioni già presenti. Per una nuova run
completa, verificare prima percorsi e cartella di output: un'esecuzione VLM su
1.600 coppie è onerosa. Non usare `--overwrite-cache` su un risultato che si
vuole conservare.



