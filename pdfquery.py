import os
from typing import Any, List, Optional

from langchain_core.prompts import PromptTemplate, format_document
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores.chroma import Chroma
from langchain_core.retrievers import BaseRetriever
from langchain_pinecone import PineconeVectorStore
# from langchain_community.retrievers.pinecone_hybrid_search import PineconeHybridSearchRetriever   # FIXME: no good, it doesn't support filter=...
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_core.runnables import RunnableLambda, RunnableConfig, RunnableSequence
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from pydantic import BaseModel, Field
from streamlit.runtime.uploaded_file_manager import UploadedFile
from pinecone import Pinecone, Index as PineconeIndex
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class PineconeReallyHybridSearchRetriever(BaseRetriever):
    vectorstore: PineconeVectorStore

    def __init__(self, vectorstore: PineconeVectorStore):
        super().__init__(vectorstore=vectorstore)

    def _get_relevant_documents(self, query: str, retriever_config: Optional[RunnableConfig], **kwargs) -> List[Document]:
        qfilter = {
            "source_type": {
                "$eq": "application/pdf"
            }
        }

        return self.vectorstore.similarity_search(
            query=query,
            filter=qfilter,
            k=4
        )


class PDFQuery:
    vectorstore: PineconeVectorStore
    chain: RunnableSequence

    def __init__(self) -> None:
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        embeddings: Embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

        # self.vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")
        # db_retriever = self.vectorstore.as_retriever()

        client: Pinecone = Pinecone(api_key=os.environ["PINECONE_API_KEY"], source_tag="langchain")
        index: PineconeIndex = client.Index("test-idx")
        self.vectorstore = PineconeVectorStore(
            embedding=embeddings,
            index=index
        )
        db_retriever = PineconeReallyHybridSearchRetriever(vectorstore=self.vectorstore)

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

        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=os.environ["OPENAI_API_KEY"])

        self.chain = (
            db_retriever |
            stuff_documents |
            prompt |
            llm
        )

    def ask(self, question: str) -> str:
        config: RunnableConfig = RunnableConfig(
            tags=[],
            metadata={
                "question": question,
                "foo": {
                    "bar": "baz"
                }
            },
            callbacks=None,
            # configurable={
            #     # this would be stuffed into metadata, but it doesn't support dicts
            #     # so it's easier to just provide metadata above
            #
            #     # supported value types: str, int, float, bool
            #     "question": question
            # },
        )
        response = self.chain.invoke(
            input=question,
            config=config,
            retriever_config=config  # NOTE: config isn't passed to _get_relevant_documents otherwise
        )
        return response.content

    def ingest(self, file_path: str, file: UploadedFile) -> None:
        loader = PyPDFium2Loader(file_path)
        documents = loader.load()
        for doc in documents:
            doc.metadata["source_name"] = file.name
            doc.metadata["source_type"] = file.type
        split_documents = self.text_splitter.split_documents(documents)
        self.vectorstore.add_documents(split_documents)
        # self.vectorstore.persist()  # deprecated

    def forget(self) -> None:
        self.chain = None
