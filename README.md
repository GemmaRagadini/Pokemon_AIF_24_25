# Pokemon_AIF_24_25

.Studiare il documento messo nel  link: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9618985
Qui si trovano le regole del gioco.

.Provare il codice vedere cosa fa capire le politiche utilizzate(probabilmente random).

.Pensare ad una nuova politica di gioco.

.Iniziare a scrivere il competitor. 
(cose da fare scritte in maniera coarse grained andare più nel dettaglio)

## Come lanciare (Gemma) 
Su due terminali: 

1. python3 -m venv amb (nella cartella pokemon-vgc-engine-master)
2. source amb/bin/activate
3. export PYTHONPATH=$PYTHONPATH:/home/gemmaraga/Desktop/AIF/Pokemon_AIF_24_25/pokemon-vgc-engine-master
4. python3 example/Example_BattleEcosystem.py

## Tempi  
Tempo impiegato per 10 turni, 2 giocatori: 
- Minimax ALpha Beta: 58.18 secondi (10-0), 140.05 (7-3)
- Minimax: 418.98 secondi (10-0) , 447.12 secondi (10-0)
      