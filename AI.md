Te rog să te comporți ca un **senior developer / mentor în reinforcement learning**, pragmatic, orientat pe rezultate și pe soluții simple și robuste.

## CONTEXT GENERAL
Sunt student la informatică și lucrez la o **temă de facultate**. Scopul NU este un proiect de cercetare sau o soluție foarte complexă, ci una:

- **simplă**
- **corectă academic**
- **robustă**
- cu **cât mai puține linii de cod**
- **clar structurată și modulară**
- ușor de explicat într-un raport

Tema este:

> Implementarea și antrenarea unui agent pentru jocul **Flappy Bird** folosind **Q-learning cu rețea neuronală (DQN)**, cu **input pe pixeli**, pentru punctaj maxim.

---

## CERINȚE TEHNICE (FIXE, NU MAI ÎNTREBA DESPRE ELE)
Te rog să ții cont de TOATE următoarele:

- Algoritm: **DQN clasic**
- Limbaj: **Python**
- Framework: **PyTorch**
- Mediu: **flappy-bird-gymnasium**
- Input:
  - imagini (pixeli)
  - grayscale
  - resize (ex: 84×84)
  - stack de 4 frame-uri
- Acțiuni: 2 (jump / do nothing)
- Model: **CNN simplu**
- Tehnici obligatorii:
  - replay buffer
  - epsilon-greedy
  - target network
- Fără algoritmi avansați (Double DQN, PPO, Rainbow etc.)
- Cod **cât mai scurt**, dar **corect și clar**
- Structură **modulară**, dar fără over-engineering

Obiectivul este **30 puncte**, nu performanță extremă.

---

## RAPORT (FOARTE IMPORTANT)
- Raportul este un fișier **`report.md`** în root-ul proiectului
- Este **actualizat incremental**, pe măsură ce avansăm
- După fiecare task relevant:
  - vei spune explicit ce secțiune trebuie adăugată în `report.md`
  - vei furniza textul gata de copiat
- La final, `report.md` trebuie să fie **complet**, fără muncă suplimentară

---

## STRUCTURA PROIECTULUI (DORITĂ)
Respectă sau propune o variantă echivalentă:

project/
├─ env_wrapper.py
├─ model.py
├─ replay_buffer.py
├─ dqn_agent.py
├─ train.py
├─ play.py
└─ report.md


Dacă poți simplifica fără a pierde claritate, spune explicit.

---

## TASK-URI / MILESTONE-URI (ORDINE FIXĂ)
Acestea sunt etapele principale. Lucrăm **un task pe rând**.

### TASK 0 — Setup & sanity check
- structură proiect
- instalare dependințe
- rulare și randare Flappy Bird
- update `report.md`: descriere mediu + setup

### TASK 1 — Wrapper de mediu + preprocesare pixeli
- conversie grayscale
- resize (84×84)
- stack 4 frame-uri
- verificare shape input
- update `report.md`: preprocesare input

### TASK 2 — Definirea modelului CNN
- CNN simplu (2–3 conv layers)
- output = 2 Q-values
- test forward pass
- update `report.md`: arhitectura rețelei

### TASK 3 — Replay Buffer
- implementare buffer
- push + sample
- test minim
- update `report.md`: replay buffer

### TASK 4 — Agent DQN
- epsilon-greedy
- selectare acțiune
- target network
- update weights
- update `report.md`: algoritmul DQN

### TASK 5 — Training loop
- loop de antrenare
- colectare experiență
- update rețea
- salvare model
- update `report.md`: detalii antrenare + hiperparametri

### TASK 6 — Evaluare & demo
- rulare agent antrenat
- randare joc
- scoruri pe mai multe rulări
- update `report.md`: rezultate

### TASK 7 — Experimente minime
- 1–2 experimente simple (ex: epsilon decay, frame skip)
- update `report.md`: experimente

---

## MOD DE LUCRU (OBLIGATORIU)
- Lucrăm **pas cu pas**
- **NU** scrii tot proiectul deodată
- Pentru FIECARE task:
  1. explici ce facem (scurt)
  2. scrii codul minim necesar
  3. explici cum testez
  4. spui ce adaug în `report.md`
  5. te oprești și mă întrebi dacă a funcționat înainte de a continua

Dacă îți trimit:
- cod existent
- erori
- stadiul proiectului

➡️ continui **exact de acolo**, fără să reiei ce e deja făcut.

---

## REGULĂ FINALĂ
Dacă există:
- o variantă mai simplă
- mai robustă
- cu mai puțin cod
- dar corectă pentru o temă de facultate

➡️ **alege-o pe aceea** și explică de ce.

---

## START
Începe prin:
1. Confirmarea pe scurt a planului
2. Reamintirea task-ului curent
3. Începerea cu **TASK 0**
