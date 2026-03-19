import mesa
from mesa.datacollection import DataCollector
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.discrete_space.cell_agent import BasicMovement, HasCell


class RuleBasedAgent(mesa.Agent, HasCell, BasicMovement):
    def __init__(self, model, name, persona, initial_stance, initial_belief):
        super().__init__(model=model)
        self.name = name
        self.stance = initial_stance
        self.belief_score = initial_belief

    def step(self):
        neighbors = [
            agent
            for cell in self.cell.neighborhood
            for agent in cell.agents
        ]

        for neighbor in neighbors:
            if neighbor.stance == "believer":
                self.belief_score += 0.1 * (1 - self.belief_score)
            elif neighbor.stance == "skeptic":
                self.belief_score -= 0.1 * self.belief_score
            else:
                self.belief_score += self.random.uniform(-0.02, 0.02)

        self.belief_score = max(0.0, min(1.0, self.belief_score))

        if self.belief_score > 0.7:
            self.stance = "believer"
        elif self.belief_score < 0.3:
            self.stance = "skeptic"
        else:
            self.stance = "neutral"


class RuleBasedModel(mesa.Model):
    def __init__(self, width=5, height=5):
        super().__init__()

        self.grid = OrthogonalMooreGrid(
            (width, height), capacity=1, torus=True, random=self.random
        )

        agent_configs = [
            {
                "name": "Maria",
                "persona": "A cautious schoolteacher who values evidence and critical thinking.",
                "initial_stance": "skeptic",
                "initial_belief": 0.2,
            },
            {
                "name": "Carlos",
                "persona": "An anxious shopkeeper who worries about health risks.",
                "initial_stance": "believer",
                "initial_belief": 0.8,
            },
            {
                "name": "Priya",
                "persona": "A young university student studying chemistry.",
                "initial_stance": "skeptic",
                "initial_belief": 0.15,
            },
            {
                "name": "James",
                "persona": "A retired factory worker who distrusts the factory owners.",
                "initial_stance": "believer",
                "initial_belief": 0.85,
            },
            {
                "name": "Aisha",
                "persona": "A community health worker.",
                "initial_stance": "neutral",
                "initial_belief": 0.5,
            },
            {
                "name": "Tom",
                "persona": "A local journalist always looking for a story.",
                "initial_stance": "neutral",
                "initial_belief": 0.45,
            },
            {
                "name": "Lin",
                "persona": "A grandmother who has lived here for 50 years.",
                "initial_stance": "neutral",
                "initial_belief": 0.55,
            },
            {
                "name": "David",
                "persona": "A government water inspector.",
                "initial_stance": "skeptic",
                "initial_belief": 0.1,
            },
            {
                "name": "Sofia",
                "persona": "A social media enthusiast who shares things quickly.",
                "initial_stance": "believer",
                "initial_belief": 0.75,
            },
            {
                "name": "Raj",
                "persona": "A doctor at the local clinic.",
                "initial_stance": "skeptic",
                "initial_belief": 0.2,
            },
            {
                "name": "Emma",
                "persona": "A stay-at-home parent concerned about children's health.",
                "initial_stance": "neutral",
                "initial_belief": 0.6,
            },
            {
                "name": "Mike",
                "persona": "A laid-back bartender who hears all sorts of gossip.",
                "initial_stance": "neutral",
                "initial_belief": 0.4,
            },
        ]

        for config in agent_configs:
            agent = RuleBasedAgent(
                model=self,
                name=config["name"],
                persona=config["persona"],
                initial_stance=config["initial_stance"],
                initial_belief=config["initial_belief"],
            )
            agent.move_to(self.grid.select_random_empty_cell())

        self.datacollector = DataCollector(
            model_reporters={
                "believers": lambda m: sum(
                    1 for a in m.agents if a.stance == "believer"
                ),
                "skeptics": lambda m: sum(
                    1 for a in m.agents if a.stance == "skeptic"
                ),
                "neutrals": lambda m: sum(
                    1 for a in m.agents if a.stance == "neutral"
                ),
                "avg_belief": lambda m: sum(a.belief_score for a in m.agents)
                / len(m.agents),
            },
            agent_reporters={
                "belief_score": "belief_score",
                "stance": "stance",
            },
        )

    def step(self):
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)
