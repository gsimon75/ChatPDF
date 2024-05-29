import os

from chromadb.errors import NoIndexException
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFium2Loader
from langchain.chains.question_answering import load_qa_chain
# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI


class PDFQuery:
    def __init__(self, openai_api_key = None) -> None:
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        # self.llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
        self.llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
        self.chain = load_qa_chain(self.llm, chain_type="stuff")
        self.vectordb = Chroma(embedding_function=self.embeddings, persist_directory="./chroma_db")
        self.db = self.vectordb.as_retriever()

    def ask(self, question: str) -> str:
        try:
            docs = self.db.get_relevant_documents(question)
            response = self.chain.run(input_documents=docs, question=question)
        except NoIndexException:
            response = "Please, add a document."

        return response

    def ingest(self, file_path: os.PathLike) -> None:
        loader = PyPDFium2Loader(file_path)
        documents = loader.load()
        splitted_documents = self.text_splitter.split_documents(documents)
        self.vectordb.add_documents(splitted_documents)

    def forget(self) -> None:
        self.db = None
        self.chain = None