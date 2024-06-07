import os
from typing import Any

# from chromadb.errors import NoIndexException
from langchain_core.prompts import PromptTemplate, format_document
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import PyPDFium2Loader
# from langchain_community.llms import OpenAIChat
from langchain_core.runnables import RunnableLambda, RunnableConfig
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pydantic import BaseModel, Field
from streamlit.runtime.uploaded_file_manager import UploadedFile


class DocumentTag(BaseModel):
    page_content: str = Field(description="The content")
    orig_source: str = Field(description="The source where the content came from")


class PDFQuery:
    def __init__(self, openai_api_key=None) -> None:
        os.environ["OPENAI_API_KEY"] = openai_api_key

        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.vectordb = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")
        db_retriever = self.vectordb.as_retriever()

        document_prompt = PromptTemplate(
            input_variables=["page_content", "metadata"],
            input_types={
                "page_content": str,
                "metadata": dict[str, Any],
            },
            output_parser=None,
            partial_variables={},
            template="{source_name}: {page_content}",  # NOTE: all variables except for `page_content` are resolved from metadata
            template_format="f-string",
            validate_template=False  # https://github.com/langchain-ai/langchain/issues/22668
        )

        def f_stuff_documents(docs, config: RunnableConfig):
            """
            StuffDocumentsChain is not a chainable Runnable, so now we do it manually,
            by wrapping this function with a RunnableLambda
            """
            return {
                "question": config["metadata"]["question"],
                "context": "\n\n".join(
                    [format_document(doc, document_prompt) for doc in docs]
                )
            }

        stuff_documents = RunnableLambda(f_stuff_documents)

        prompt = ChatPromptTemplate(
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
If you don't know the answer, just say that you don't know, don't try to make up an answer.
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

        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
        # structured_llm = self.llm.with_structured_output(DocumentTag) # FIXME: needs package upgrade to recent

        self.chain = (
            db_retriever |
            stuff_documents |
            prompt |
            llm
        )

    def ask(self, question: str) -> str:
        response = self.chain.invoke(
            question,
            {
                "metadata": {
                    "question": question
                }
            }
        )
        return response.content

    def ingest(self, file_path: str, file: UploadedFile) -> None:
        loader = PyPDFium2Loader(file_path)
        documents = loader.load()
        for doc in documents:
            doc.metadata["source_name"] = file.name
            doc.metadata["source_type"] = file.type
        split_documents = self.text_splitter.split_documents(documents)
        self.vectordb.add_documents(split_documents)
        # self.vectordb.persist()  # deprecated

    def forget(self) -> None:
        self.chain = None
