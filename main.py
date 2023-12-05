from dotenv import load_dotenv
load_dotenv()
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA

# Loader
loader = PyPDFLoader("unsu.pdf")
pages = loader.load_and_split()

# Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap  = 20,
    length_function = len,
    is_separator_regex = False,
)
texts = text_splitter.split_documents(pages)

# Embeddings
embeddings_model = OpenAIEmbeddings()

# Chroma
vectordb = Chroma.from_documents(texts, embeddings_model)

# load from directory
# vectordb = Chroma.from_documents(texts, embeddings_model, persist_directory="./chroma_db")
# vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings_model)

# Question
question = "아내가 먹고 싶어하는 음식은 무엇이야?"
# 관련 문서를 찾는다.
# llm = ChatOpenAI(temperature=0)
# retriever_from_llm = MultiQueryRetriever.from_llm(
#     retriever=vectordb.as_retriever(), llm=llm
# )
# docs = retriever_from_llm.get_relevant_documents(query=question)

llm = ChatOpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever())
result = qa_chain({"query": question})
print(result)
