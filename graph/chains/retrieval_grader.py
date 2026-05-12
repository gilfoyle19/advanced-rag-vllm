from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from graph.chains.llm import get_llm

llm = get_llm()


class GradeDocuments(BaseModel):
    """Binary score whether the document is relevant to the question, yes or no
    Gives pydantic object with binary score"""

    binary_score: str = Field(
        description="Binary score whether the document is relevant to the question, yes or no"
    )


parser = PydanticOutputParser(pydantic_object=GradeDocuments)

system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
    Return only JSON that matches these instructions: {format_instructions}"""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

retrieval_grader = grade_prompt | llm | parser
