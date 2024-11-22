# Pokémon VGC AI Framework 3.0.5.7

[[_TOC_]]

## Changelog 3.0.5.7 (New)

What's new?

* Previous baseline tree-search agents were fixed; A new agent, TunedTreeTraversal was added.
* Fixed broken dependencies from the elo package installation, migrated to elopy.
* PkmBattleEnv was migrated from gym to gymnasium.
* New terminal-based agents for Pokémon battling, team selection and team building.
* New gui-based agents for Pokémon battling.
* Integration of baseline team building agents from the VGC-Agent projects.
* Meta-Game Balance Track upgraded with MetaEvaluator to allow different evaluation criteria.
* requirements.txt updated.
* Readme.md updated, tutorials were fixed to correspond to technical changes of 3.0.

## Installation (New)

If you wish to use a clone of this project standalone, we recommend to install the dependencies using requirements.txt

```
git clone https://gitlab.com/DracoStriker/pokemon-vgc-engine.git
cd pokemon-vgc-egine
pip install -r requirements.txt
```

If you wish to install vgc as a package in your venv, then you can:

```
git clone https://gitlab.com/DracoStriker/pokemon-vgc-engine.git
cd pokemon-vgc-egine
pip install .
```

## Project Organization

The `/vgc` module is the core implementation of the VGC AI Framework.

In the `/test` folder is contained some unit tests from the core framework modules.

In the `/example` folder it can be found multiple examples for how to use the framework, to train or test isolated
agents or behaviours or run full ecosystems with independent processes controlling each agent.

In the `/organization` folder it can be found the multiple entry points for the main ecosystem layers in the VGC AI
Framework.

In the `/competition` folder can be found previous years entries.

## Tutorials

### Set a Pokémon  Battle in the Pokémon  Battle Env (gym)

Set Pokémon battles is just to set a simple gym environment loop. The `PkmBattleEnv` is parametrized
by two `PkmTeam`, each will be piloted by its respective `BattlePolicy` agent.

```python
team0, team1 = PkmTeam(), PkmTeam()
agent0, agent1 = RandomPlayer(), RandomPlayer()
env = PkmBattleEnv((team0, team1),
                   encode=(agent0.requires_encode(), agent1.requires_encode())  # set new environment with teams
n_battles = 3  # total number of battles
t = False
battle = 0
while battle < n_battles:
    s, _ = env.reset()
    while not t:  # True when all pkms of one of the two PkmTeam faint
        a = [agent0.get_action(s[0]), agent1.get_action(s[1])]
        s, _, t, _, _ = env.step(a)  # for inference, we don't need reward
        env.render()
    t = False
    battle += 1
print(env.winner)  # winner id number
```

`s` is a duple with the game state encoding for each agent. `r` is a duple with the reward for each agent.

To create custom `PkmTeam` you can just input a list of `Pkm`.

Agents may require the standard game state encoding for their observations. Agents' `BattlePolicy` encode such
information in the `requires_encode()` method. We pass the required encoding protocol to the environment.

```python
team = PkmTeam([Pkm(), Pkm(), Pkm()])  # up to three!
```

The `PkmTeam` represents a in battle team, which is a subset of a `PkmFullTeam`. The later is used for team building
settings. You can obtain a battle team from a full team by providing the team indexes.

```python
full_team = FullPkmTeam([Pkm(), Pkm(), Pkm(), Pkm(), Pkm(), Pkm()])
team: PkmTeam = full_team.get_battle_team([1, 4, 5])
```

### Create a Pokémon  Roster and Meta

A `PkmRoster` represents the entirety of unit selection for a team build competition. It is defined as
`List[PkmTemplate]`. A `PkmTemplate` represents a Pokémon species. It defines the legal stats combinations and move set
for that Pokémon species.

```python
roster = [PkmTemplate(), PkmTemplate(), PkmTemplate()]
```

To get a `Pkm` instance from a `PkmTemplate` you just need to provide the moves indexes.

```python
templ = PkmTemplate()
pkm = templ.gen_pkm([1, 2, 5, 3])  # 4 max!
```

To create a meta is as simple as initializing.

```python
meta_data = StandardMetaData()
meta_data.set_moves_and_pkm(self, roster, move_roster)
```

The `StandardMetaData` assumes that the `move_roster` contains `PkmMove` that have the field `move_id` ordered and with
values from 0 to n-1, where n is the number of moves. All existing `PkmMove` in `PkmTemplate`s in the `roster` should
also be present in the `move_roster`.

### Query Meta

```python
class MetaData(ABC):
    ...

    def get_global_pkm_usage(self, pkm_id: PkmId) -> float

        def get_global_move_usage(self, move: PkmMove) -> float

        def get_pair_usage(self, pkm_ids: Tuple[PkmId, PkmId]) -> float

        def get_team(self, t) -> Tuple[PkmFullTeam, bool]

        def get_n_teams(self) -> int
```

Several standard methods can be used to query usage rate information of isolated moves, Pokémon and teams.

### Create My Battle Policy

The battle policy must inherit from `BattlePolicy` (example bellow). The team build policy must inherit from
`TeamBuildPolicy`.

```python
class MyVGCBattlePolicy(BattlePolicy):

    def get_action(self, g: GameState) -> int:
        # get my team
        my_team = g.teams[0]
        my_active = my_team.active
        my_active_type = my_active.type
        my_active_moves = my_active.moves

        # get opp team
        opp_team = g.teams[1]
        opp_active = opp_team.active
        opp_active_type = opp_active.type

        # get best move
        damage: List[float] = []
        for move in my_active_moves:
            damage.append(estimate_damage(move.type, my_active_type, move.power, opp_active_type))
        move_id = int(np.argmax(damage))
        return move_id
```

If you want to receive the `GameState` then your `BattlePolicy.requires_encode` must return `False`. If you want to
receive automatically the standard encoded game state as `get_action(self, s: List[float])` your
`BattlePolicy.requires_encode` must return `True`.

### Forward Model

The `GameState` provided to you is in reality a `PkmBattleEnv` object (which inherits from `GameState`), so you can
forward the game state using the gym method `step` providing the joint action. Note that only public or predicted
information will be available (if a move is unknown it will be replaced by a normal type `PkmMove`, and same for the
`Pkm`), with no effects and a base move power and hp.

```python
 def get_action(self, g) -> int:  # g: PkmBattleEnv
    my_action = 0
    opp_action = 0
    s, _, _, _ = g.step([my_action, opp_action])
    g = s[0]  # my game state view (first iteration)
    my_action = 1
    opp_action = 1
    s, _, _, _ = g.step([my_action, opp_action])
    g = s[0]  # my game state view (second iteration)
```

### Create My Team Build Policy

At the beginning of a championship, or during a meta-game balance competition, `set_roster` is called providing the
information about the available roster. You can use that opportunity to store the roster or to make some preprocessing
about the `Pkm` win rates.

```python
class MyVGCBuildPolicy(TeamBuildPolicy):
    """
    Agents that selects teams randomly.
    """

    def __init__(self):
        self.roster = None

    def set_roster(self, roster: PkmRoster):
        self.roster = roster

    def get_action(self, meta: MetaData) -> PkmFullTeam:
        roster = list(self.roster)
        pre_selection: List[PkmTemplate] = [roster[i] for i in random.sample(range(len(roster)), DEFAULT_TEAM_SIZE)]
        team: List[Pkm] = []
        for pt in pre_selection:
            team.append(pt.gen_pkm(random.sample(range(len(pt.move_roster)), DEFAULT_PKM_N_MOVES)))
        return PkmFullTeam(team)
```

### Create My VGC AI Agent

To implement a VGC competitor agent you need to create an implementation of the class `Competitor` and override its
multiple methods that return the various types of behaviours that will be called during an ecosystem simulation.
Example:

```python
class MyVGCCompetitor(Competitor):

    def __init__(self):
        self.my_battle_policy = MyVGCBattlePolicy()
        self.my_team_build_policy = MyVGCBuildPolicy()

    @property
    def battle_policy(self) -> BattlePolicy:
        return self.my_battle_policy

    @property
    def team_build_policy(self) -> TeamBuildPolicy:
        return self.my_team_build_policy

    @property
    def name(self) -> str:
        return "My VGC AI competition"
```

### Set Competition Managers and a Tree Championship

A `CompetitorManager` binds and manages a `Competitor` with its current `PkmFullTeam` and respective performance (ELO
rating). These can be used in the context of a `TreeChampionship` or any full Ecosystem track.

```python
roster = RandomPkmRosterGenerator().gen_roster()
competitors = [ExampleCompetitor('Player ' + str(i)) for i in range(N_PLAYERS)]
championship = TreeChampionship(roster, debug=True)
for competitor in competitors:
    championship.register(CompetitorManager(competitor))  # add competitor to the tournament and set his team
championship.new_tournament()  # create a tournament tree
winner = championship.run()  # run tournament
print(winner.competitor.name + " wins the tournament!")  # fetch winner
```

The `TeamBuildPolicy` from the `Competitor` is called to request the agent to choose its team.

### Run Your own Full Competitions

The `ChampionshipEcosystem` is used to simulate a Championship Competition Track. You just need to instantiate a
`PkmRoster`, `MetaData`, and register the competitors wrapped under their `CompetitorManager`. You must set both the
number of championship epochs and how many battle epochs run inside each championship epoch.

```python
generator = RandomPkmRosterGenerator()
roster = generator.gen_roster()
move_roster = generator.base_move_roster
meta_data = StandardMetaData()
meta_data.set_moves_and_pkm(roster, move_roster)
ce = ChampionshipEcosystem(roster, meta_data, debug=True)
battle_epochs = 10
championship_epochs = 10
for i in range(N_PLAYERS):
    cm = CompetitorManager(ExampleCompetitor("Player %d" % i))
    ce.register(cm)
ce.run(n_epochs=battle_epochs, n_league_epochs=championship_epochs)
print(ce.strongest.name)  # determine winner by checking the highest ELO rating!
```

### Visualize Battles

See and use examples provided in `vgc/ux`. Run `vgc/ux/PkmBattleClientTest.py` and `vgc/ux/PkmBattleUX.py` in that
order.

## Documentation

The full documentation from API, Framework architecture to the Competition Tracks and
Rules can be found in the [Wiki](https://gitlab.com/DracoStriker/pokemon-vgc-engine/-/wikis/home).

## Citation

The technical document can be found in the following link:

https://ieeexplore.ieee.org/document/9618985

Please cite this work if used.

```
@INPROCEEDINGS{9618985,

  author={Reis, Simão and Reis, Luís Paulo and Lau, Nuno},

  booktitle={2021 IEEE Conference on Games (CoG)}, 

  title={VGC AI Competition - A New Model of Meta-Game Balance AI Competition}, 

  year={2021},

  volume={},

  number={},

  pages={01-08},

  doi={10.1109/CoG52621.2021.9618985}}
```

## TODO

* Improve Game State encoding performance.
* Integrate more meta-game evaluators.
* Implement In-Game Balance Track.
* GUI interface for team selection.
* GUI interface for team building.