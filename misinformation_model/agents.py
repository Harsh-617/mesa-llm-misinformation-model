from mesa_llm.llm_agent import LLMAgent
from misinformation_model.tools import (
    check_neighbors,
    challenge_rumor,
    spread_rumor,
    update_belief,
)


class CitizenAgent(LLMAgent):
    def __init__(self, model, name, persona, initial_stance, initial_belief):
        super().__init__(model)
        self.name = name
        self.persona = persona
        self.stance = initial_stance
        self.belief_score = initial_belief
        self.system_prompt = (
            f"You are {self.name}, a citizen in a small community. "
            f"{self.persona}"
        )
        self.tools = [check_neighbors, spread_rumor, challenge_rumor, update_belief]

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
        self.execute(prompt=prompt)
