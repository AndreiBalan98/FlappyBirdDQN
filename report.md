# Raport Proiect: Flappy Bird DQN

## 1. Introducere si Obiective

* **Scop**: Crearea unui agent de Reinforcement Learning capabil sa joace Flappy Bird folosind doar imaginea (pixelii).
* **Target**: Obtinerea unui scor de peste 30 de puncte.
* **Tehnologii**: Python, PyTorch, Gymnasium.

## 2. Preprocesarea Datelor

Pentru a face antrenarea eficienta, imaginile brute de 512x288 sunt transformate astfel:

* **Grayscale**: Eliminam culorile inutile.
* **Resize**: Reducem la 84x84 pixeli.
* **Normalizare**: Valorile pixelilor trec de la [0, 255] la [0, 1].
* **Frame Stacking**: Grupam ultimele 4 cadre consecutive intr-un singur input (shape 4x84x84) pentru ca agentul sa poata percepe directia si viteza pasarii.

## 3. Arhitectura Retelei (CNN)

Am folosit o retea neuronala convolutionala (CNN) cu urmatoarea structura:

1. **3 Straturi Convolutionale**: Extrag trasaturile vizuale (tuburi, pozitia pasarii).
2. **2 Straturi Fully Connected**: Primul strat are 512 neuroni, iar ultimul are 2 neuroni (corespunzatori actiunilor: *stay* sau *jump*).
3. **Activare**: ReLU pentru straturile intermediare.

## 4. Algoritmul si Strategia de Invatare

* **DQN (Deep Q-Network)**: Reteaua invata sa estimeze valoarea Q (cat de buna este o actiune intr-o anumita stare).
* **Experience Replay**: Folosim un buffer de 100.000 de experiente din care extragem batch-uri aleatorii (64 de sample-uri) pentru a sparge corelatia dintre cadre consecutive.
* **Target Network**: O a doua retea, actualizata mai rar (la 500 pasi), folosita pentru stabilitatea calculului erorii (loss).
* **Epsilon-Greedy**: Agentul incepe prin a explora total (epsilon=1.0) si ajunge treptat la o explorare minima (epsilon=0.01).

## 5. Antrenare si Parametri

* **Learning Rate**: 5e-4.
* **Discount Factor (Gamma)**: 0.99 (agentul prioritizeaza recompensele pe termen lung).
* **Optimizer**: Adam cu Huber Loss (mai stabil la valori extreme).
* **Durata**: Antrenat pe parcursul a 5.000 de episoade.

## 6. Rezultate si Evaluare

* **Dupa 20 episoade**: (100 episoade)
============================================================
statistici:
   reward mediu:  -1.43 ± 0.73
   reward max:    -0.90
   reward min:    -2.50
   length mediu:  53
============================================================

* **Dupa 250 episoade**: (100 episoade)
============================================================
statistici:
   reward mediu:  -0.58 ± 0.90
   reward max:    2.30
   reward min:    -1.80
   length mediu:  53
============================================================

* **Dupa 500 episoade**: (100 episoade)
============================================================
statistici:
   reward mediu:  1.47 ± 1.46
   reward max:    3.20
   reward min:    -0.90
   length mediu:  54
============================================================

---
