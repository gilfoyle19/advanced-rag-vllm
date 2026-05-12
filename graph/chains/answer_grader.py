from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field
from graph.chains.llm import get_llm


class GradeAnswer(BaseModel):

    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


llm = get_llm()
parser = PydanticOutputParser(pydantic_object=GradeAnswer)

system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score true or false. True means that the answer resolves the question.
     Return only JSON that matches these instructions: {format_instructions}"""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

answer_grader: RunnableSequence = answer_prompt | llm | parser
