from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field
from graph.chains.llm import get_llm

llm = get_llm()


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


parser = PydanticOutputParser(pydantic_object=GradeHallucinations)

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score true or false. True means that the answer is grounded in / supported by the set of facts.
     Return only JSON that matches these instructions: {format_instructions}"""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

hallucination_grader: RunnableSequence = hallucination_prompt | llm | parser
