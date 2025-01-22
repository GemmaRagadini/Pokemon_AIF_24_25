# Pokemon_AIF_24_25

The project runs over an existing project called pokemon-vgc-engine-master. This project contains various directory with parts of the game. The section that have been modified are the section pokemon-vgc-engine-master/vgc/behaviour in particular there are new files that are:

-*MyPolicy.py*: this file contains the implementation of our custom policy 

-*otherPolicies.py*: this file contains the implementation of the other policies in particular the Minimax and Minimax with alpha beta pruning and killer move heuristic woth the two different evaluation functions

-*evalFunctions.py*: this file contains the two evaluation functions that are been usen in the implementation of the Minimax policies

The other files that have been modified are in the section pokemon-vgc-engine-master/example and in particular they are called:

-*Example_Competitor.py*: In this file there are the competitor that used the different policies that we implemented in the file listed previously

-*Example_BattleEcosystem.py*: In this file we implemented the test face where we implemented the single matches between competitor and the tournament between all the players we have definied

## How to run 

Once you cloned the progect using 

!git clone https://github.com/GemmaRagadini/Pokemon_AIF_24_25.git

you move to the directory 

%cd Pokemon_AIF_24_25/pokemon-vgc-engine-master

then you can install the requirements needed to run the project using the comand

1. python3 -m ve

      
