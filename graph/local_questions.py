LOCAL_DOCUMENT_CUES = (
    "according to the document",
    "according to the pdf",
    "according to the report",
    "according to this document",
    "according to this pdf",
    "according to this report",
    "in the document",
    "in the pdf",
    "in the report",
    "this document",
    "this pdf",
    "this report",
    "the document",
    "the pdf",
    "the report",
    "chapter ",
    "section ",
)


def is_local_document_question(question: str) -> bool:
    normalized = question.lower()
    return any(cue in normalized for cue in LOCAL_DOCUMENT_CUES)
