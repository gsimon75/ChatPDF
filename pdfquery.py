import os

# from chromadb.errors import NoIndexException
from langchain_core.prompts import PromptTemplate, format_document
from langchain.chains import LLMChain  # deprecated
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import PyPDFium2Loader
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_community.llms import OpenAIChat
from langchain_core.runnables import RunnableLambda
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pydantic import BaseModel, Field
from streamlit.runtime.uploaded_file_manager import UploadedFile


class DocumentTag(BaseModel):
    page_content: str = Field(description="The content")
    orig_source: str = Field(description="The source where the content came from")


document_prompt = PromptTemplate(
    input_variables=['page_content'],
    output_parser=None,
    partial_variables={},
    template='{page_content}',
    template_format='f-string',
    validate_template=True
)

def stuff_whatever(input):
    docs = input["input_documents"]
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    result = "\n\n".join(doc_strings)
    return {
        "context": result,
        "question": input["question"]
    }

class PDFQuery:
    def __init__(self, openai_api_key = None) -> None:
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        # self.llm = OpenAIChat(temperature=0, openai_api_key=openai_api_key) # .with_structured_output(DocumentTag)
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
        # self.structured_llm = self.llm.with_structured_output(DocumentTag) # FIXME: needs package upgrade to recent

        _prompt = ChatPromptTemplate(
            input_variables=["question", "context"],
            output_parser=None,
            partial_variables={},
            messages=[
                SystemMessagePromptTemplate(
                    prompt=PromptTemplate(
                        input_variables=["context"],
                        output_parser=None,
                        partial_variables={},
                        template="""
Use the following pieces of context to answer the users question.
If you don"t know the answer, just say that you don"t know, don"t try to make up an answer.
----------------
{context}""",
                        template_format="f-string",
                        validate_template=True
                    ),
                    additional_kwargs={}
                ),
                HumanMessagePromptTemplate(
                    prompt=PromptTemplate(
                        input_variables=["question"],
                        output_parser=None,
                        partial_variables={},
                        template="{question}",
                        template_format="f-string",
                        validate_template=True
                    ),
                    additional_kwargs={}
                )
            ]
        )

#        llm_chain = LLMChain(
#            llm=self.llm,
#            prompt=_prompt,
#            verbose=None,
#            callback_manager=None
#        )
#        self.chain = StuffDocumentsChain(
#            llm_chain=llm_chain,
#            document_variable_name="context",
#            verbose=None,
#            callback_manager=None
#        )

        self.chain = RunnableLambda(stuff_whatever) | _prompt | self.llm

        self.vectordb = Chroma(embedding_function=self.embeddings, persist_directory="./chroma_db")
        self.db = self.vectordb.as_retriever()

    def dump_vectordb(self) -> None:
        print("Dumping the DB:")
        print(self.vectordb.get())

    def ask(self, question: str) -> str:
        docs = self.db.invoke(question)

        response = self.chain.invoke(
            {
                "input_documents": docs,
                "question": question
            }
        )
        return response.content

    def ingest(self, file_path: os.PathLike, file: UploadedFile) -> None:
        loader = PyPDFium2Loader(file_path)
        documents = loader.load()
        for doc in documents:
            doc.metadata["source_name"] = file.name
            doc.metadata["source_type"] = file.type
        splitted_documents = self.text_splitter.split_documents(documents)
        self.vectordb.add_documents(splitted_documents)
        # self.vectordb.persist()  # deprecated

    def forget(self) -> None:
        self.db = None
        self.chain = None