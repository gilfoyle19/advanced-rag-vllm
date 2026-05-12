from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from graph.chains.llm import get_llm

llm = get_llm()
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a careful RAG assistant. Answer the question using only the provided context. "
            "If the context does not contain the answer, say that you do not know. Keep the answer concise.",
        ),
        ("human", "Question:\n{question}\n\nContext:\n{context}"),
    ]
)

generation_chain = prompt | llm | StrOutputParser()
