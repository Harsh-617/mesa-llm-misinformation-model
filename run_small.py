import matplotlib.pyplot as plt

from misinformation_model.agents import CitizenAgent
from misinformation_model.model import MisinformationModel


class SmallMisinformationModel(MisinformationModel):
    def __init__(self, **kwargs):
        # Skip MisinformationModel.__init__ and call mesa.Model.__init__ directly
        super(MisinformationModel, self).__init__()

        from dotenv import load_dotenv
        from mesa.datacollection import DataCollector
        from mesa.discrete_space import OrthogonalMooreGrid

        load_dotenv()

        self.llm_model = kwargs.get("llm_model", "ollama/llama3.2:3b")
        self.rumor = (
            "The town's water supply has been contaminated with dangerous "
            "chemicals from the nearby factory."
        )

        width = kwargs.get("width", 5)
        height = kwargs.get("height", 5)
        self.grid = OrthogonalMooreGrid((width, height), capacity=1, torus=True, random=self.random)

        agent_configs = [
            {
                "name": "Maria",
                "persona": "A cautious schoolteacher who values evidence and critical thinking. You don't believe things easily.",
                "initial_stance": "skeptic",
                "initial_belief": 0.2,
            },
            {
                "name": "Carlos",
                "persona": "An anxious shopkeeper who worries about health risks. You tend to believe warnings about safety.",
                "initial_stance": "believer",
                "initial_belief": 0.8,
            },
            {
                "name": "Aisha",
                "persona": "A community health worker who has seen real contamination cases before. You take such claims seriously but want proof.",
                "initial_stance": "neutral",
                "initial_belief": 0.5,
            },
            {
                "name": "Tom",
                "persona": "A local journalist always looking for a story. You're curious but need verification before reporting.",
                "initial_stance": "neutral",
                "initial_belief": 0.45,
            },
        ]

        for config in agent_configs:
            agent = CitizenAgent(
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


def main():
    print("Starting Small Misinformation Spread Simulation (4 agents, 3 steps)...")
    model = SmallMisinformationModel()

    num_steps = 3
    for i in range(num_steps):
        print(f"\n--- Step {i + 1} ---")
        model.step()

        for agent in model.agents:
            print(f"  {agent.name:8s} | stance: {agent.stance:8s} | belief: {agent.belief_score:.2f}")

        believers = sum(1 for a in model.agents if a.stance == "believer")
        skeptics = sum(1 for a in model.agents if a.stance == "skeptic")
        neutrals = sum(1 for a in model.agents if a.stance == "neutral")
        print(f"\n  Summary: {believers} believers, {skeptics} skeptics, {neutrals} neutrals")

    # Plot model-level results
    model_data = model.datacollector.get_model_vars_dataframe()

    plt.figure(figsize=(10, 6))
    plt.plot(model_data["believers"], color="red", label="Believers", marker="o")
    plt.plot(model_data["skeptics"], color="blue", label="Skeptics", marker="s")
    plt.plot(model_data["neutrals"], color="gray", label="Neutrals", marker="^")
    plt.xlabel("Step")
    plt.ylabel("Number of Agents")
    plt.title("Misinformation Spread Over Time (Small Model)")
    plt.legend()
    plt.grid(True)
    plt.savefig("results_small.png")
    print("\nPlot saved to results_small.png")
    plt.show()

    # Print final agent belief scores
    agent_data = model.datacollector.get_agent_vars_dataframe()
    print("\n--- Final Belief Scores ---")
    last_step = agent_data.xs(num_steps - 1, level="Step")
    for agent_id, row in last_step.iterrows():
        print(f"  Agent {agent_id:3d} | stance: {row['stance']:8s} | belief: {row['belief_score']:.2f}")


if __name__ == "__main__":
    main()
