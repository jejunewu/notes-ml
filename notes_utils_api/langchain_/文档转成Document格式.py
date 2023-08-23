from langchain.schema.document import Document

doc = Document(page_content="你好吗？我很好！", metadata={"source": "local"})

print(doc)