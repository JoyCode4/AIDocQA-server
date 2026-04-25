from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader("sample.pdf")

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

prompt = PromptTemplate(
    template="""
    You are a helpful assistant that can answer questions about the document.
    Answer ONLY from the provided document context.
    If the context is insufficient, just say you don't know.

    {context}

    Question: {question}
    """,
    input_variables=["context", "question"]
)
# this will be dynamic as question changes all above will be same for document uploaded by user
question = "What is smart contract as per the document?"
retrieved_docs = retriever.invoke(question)

context = "\n\n".join([doc.page_content for doc in retrieved_docs])
final_prompt = prompt.format(context=context, question=question)

answer = llm.invoke(final_prompt)
print(answer.content)

# Todo save the chat history and messages in a database and chat history will be saved in a database and messages will be saved in a database
class ChatHistory(BaseModel):
    role:Literal["user", "assistant"],
    content:str = Field(..., min_length=1),
    created_at:datetime = Field(default_factory=datetime.now)

chat_history=[]
chat_history.append(ChatHistory(role="user", content="What is smart contract as per the document?"))
chat_history.append(ChatHistory(role="assistant", content=answer.content))
print(chat_history)
