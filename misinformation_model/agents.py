from mesa.discrete_space.cell_agent import BasicMovement, HasCell
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.reasoning.react import ReActReasoning

from misinformation_model.tools import (  # noqa: F401 — import so @tool registers them
    check_neighbors,
    challenge_rumor,
    spread_rumor,
    update_belief,
)


class CitizenAgent(LLMAgent, HasCell, BasicMovement):
    def __init__(self, model, name, persona, initial_stance, initial_belief):
        super().__init__(
            model=model,
            reasoning=ReActReasoning,
            llm_model=model.llm_model,
            system_prompt=(
                f"You are {name}, a citizen in a small community. {persona}"
            ),
        )
        self.name = name
        self.stance = initial_stance
        self.belief_score = initial_belief

    def step(self):
        prompt = (
            f"Your current belief score is {self.belief_score:.2f} and your stance "
            f'is "{self.stance}".\n'
            f'The rumor going around is: "{self.model.rumor}"\n\n'
            f"First, use check_neighbors to see who is nearby. "
            f"Then, pick one neighbor and either spread_rumor if you believe it "
            f"or challenge_rumor if you doubt it. "
            f"Finally, reflect on the situation and use update_belief to adjust "
            f"your belief score."
        )
        plan = self.reasoning.plan(prompt=prompt)
        self.apply_plan(plan)
