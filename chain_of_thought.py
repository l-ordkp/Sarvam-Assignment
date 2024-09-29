import dspy
from generation import gemini_api_key

gemini = dspy.Google("models/gemini-1.0-pro",api_key=gemini_api_key, temperature=0.7)
dspy.settings.configure(lm=gemini, max_tokens=1024)
# Custom generator class with Chain of Thought reasoning

class Generator(dspy.Module):
    def __init__(self):
        super().__init__()
        # Define the ChainOfThought template for reasoning and generation
        self.generate = dspy.ChainOfThought("query: str, context: str -> response: str")

    def forward(self, query: str, context: str) -> str:
        # Apply the Chain of Thought reasoning and generate the response
        cot_result = self.generate(query=query, context=context)
        
        # Combine the context (retrieved text) with the generated response
        response = cot_result.response + context
        return response
def load_generator():
    # Instantiate and return the Generator module
    return Generator()


