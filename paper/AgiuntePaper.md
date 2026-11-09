Guida per la revisione delle aggiunte NLP nel paper

Questo documento indica  le parti del paper in cui sono state
aggiunte o modificate sezioni riguardanti l'estensione NLP. Il paper da
visionare è paper/sigconf.tex. Le aggiunte NLP sono state inserite come
valutazione supplementare: non modificano l'attacco DÆLTA, i risultati sui
detector o il punteggio ufficiale della challenge AADD.

Abstract

Nell'abstract è stata aggiunta una breve sintesi della nuova analisi NLP. In
particolare, viene anticipato il confronto tra le caption dell'immagine
originale e dell'immagine attacked e vengono riportati i due risultati centrali:
BERTScore F1 pari a 0.9507 e Semantic Consistency pari a 0.6062 sulle 1.057
coppie valutabili. Il messaggio da verificare è che caption molto simili non
implicano necessariamente che posa e sfondo siano percepiti allo stesso modo.

Introduzione

Nell'introduzione è stato inserito il motivo per cui è stata aggiunta la parte
NLP. Le metriche già presenti, come SSIM, LPIPS e Challenge Score, misurano la
qualità visiva e il successo dell'attacco, ma non descrivono in modo esplicito
quali elementi della scena diventano meno chiari. È stata quindi introdotta una
valutazione semantica separata, usata solo per interpretare meglio l'effetto
dell'attacco.

Related Work

Nel Related Work è stato aggiunto un breve collegamento con la letteratura su
valutazione vision-language, BERTScore e domande visive atomiche. Questa parte
spiega anche una scelta importante: l'audit NLP non viene presentato come un
nuovo stato dell'arte, perché non esiste una leaderboard direttamente
confrontabile per coppie di immagini originale e attacked.

Experimental Settings - Supplementary VLM-based semantic audit

In questa nuova sottosezione è stato inserito il protocollo dell'audit NLP.
Qwen2.5-VL-3B-Instruct osserva in modo indipendente l'immagine originale e
l'immagine attacked, usando lo stesso prompt neutro. Per ogni immagine produce
una caption e cinque attributi: sorriso visibile, occhiali, posa della testa,
copricapo e tipo di sfondo.

Nella stessa parte è stata aggiunta la formula della Semantic Consistency. Il
confronto usa solo gli attributi chiari nell'immagine originale. Il valore
unknown indica che un attributo non è leggibile; se compare solo dopo
l'attacco viene considerato come informazione non conservata. Una coppia entra
nella media quando nell'originale sono chiari almeno tre attributi su cinque.

Sempre in questa sottosezione sono state descritte BERTScore e Concept
Retention. BERTScore controlla se le caption sono linguisticamente simili;
Concept Retention controlla invece se i dettagli lessicali dell'originale
ricompaiono nella caption attacked. Le tre misure non vengono sommate e non
entrano nel punteggio AADD.

Experimental Evaluations - Supplementary Semantic Audit

In questa nuova sottosezione sono stati inseriti i risultati della run completa
su 1.600 coppie. La prima tabella riporta Semantic Consistency, BERTScore,
Concept Retention e il numero di coppie valide per ciascuna misura. La seconda
tabella riporta la conservazione dei cinque attributi uno per uno.

Questa è la parte più importante da controllare. Il risultato mostra che
occhiali e copricapo restano generalmente più stabili, mentre posa della testa
e tipo di sfondo sono le categorie meno conservate. Il paper spiega inoltre che
il BERTScore alto e la Semantic Consistency più bassa non sono in contrasto:
il modello mantiene spesso una descrizione generale della persona, ma tende a
semplificare o cambiare alcuni dettagli della scena.

Nella stessa sottosezione sono stati aggiunti il pilot iniziale e il controllo
di ripetibilità greedy. Le tre inferenze ripetute sulle 40 immagini del pilot
hanno prodotto sempre gli stessi attributi, caption e JSON. È stata inoltre
riportata la verifica visiva qualitativa dell'artefatto di overlay o doppia
esposizione. Il testo precisa che questa osservazione è una conferma manuale
qualitativa e non uno studio umano formale.

Scope and limitations

In questa parte sono stati aggiunti i limiti dell'estensione NLP. La metrica
misura la percezione di un solo VLM con un solo prompt, copre soltanto cinque
attributi e non rappresenta una ground truth umana. Questa precisazione serve a
mantenere l'interpretazione dei risultati corretta e prudente.

Conclusioni e sviluppi futuri

Nelle conclusioni è stato aggiunto un richiamo al risultato NLP principale:
caption molto simili possono coesistere con una minore stabilità di posa e
sfondo. Sono stati indicati anche possibili sviluppi futuri, come un secondo
VLM indipendente, prompt equivalenti e giudizi umani strutturati. Questi punti
sono solo proposte future e non risultati già ottenuti.

Reproducibility Statement

Nella dichiarazione di riproducibilità è stato aggiunto che il repository
contiene anche il codice e i risultati dell'audit semantico NLP.

Per approfondire i dati e le scelte metodologiche, i documenti di supporto sono
nella cartella Ricerca/NLP_SEMANTIC_EXTENSION/. Il documento principale con i
risultati completi è RESULTS_V1.md.
