from fastapi import FastAPI , UploadFile ,  File
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

import os
import uuid

load_dotenv()
app = FastAPI()
pc = Pinecone(api_key = os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))
embedder = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.getenv("HF_TOKEN"),
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=key)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using this context:\n\n{context}"),
    ("human", "{question}")
])

@app.post("/upload")
async def upload_pdf(file: UploadFile):
    try:
       with open(f"./tmp/{file.filename}", 'wb') as f:
            b = await file.read()
            f.write(b)
       docs = load_pdf(f"./tmp/{file.filename}")
       chunks = chunk_documents(docs)
       namespace = str(uuid.uuid4())
       embed_and_upsert(chunks, namespace)
       return {"namespace": namespace}
    except Exception as e:
        return {"error": str(e)}
    finally:
        os.remove(f"./tmp/{file.filename}")


@app.post("/ask")
def ask_doc(query: str, query_namespace: str):
    vectors = embedder.embed_query(query)
    response = index.query(top_k=5, vector=vectors, namespace = query_namespace, include_metadata=True)
    context = "\n\n".join(m.metadata["text"] for m in response["matches"])
    sources = [{"source": m.metadata["source"], "page": m.metadata["page"]} for m in response["matches"]]
    system_msg = SystemMessage(content = context)
    result = llm.invoke([system_msg, HumanMessage(content = query)])
    return {"answer": result.content, "sources": sources}
    

@app.post("/ask-with-lcel")
def ask_doc_lcel(query: str, query_namespace: str):
    retriever = PineconeRetriever(namespace=query_namespace)
    chain = (
        RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
    )
    chain_with_sources = RunnableParallel(
        {
            "answer": chain,
            "docs": retriever
        }
    )
    result = chain_with_sources.invoke(query)
    sources = [{"source": docs.metadata["source"], "page": docs.metadata["page"]} for docs in result["docs"]]
    return {"answer": result["answer"].content, "sources": sources}


def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    return loader.load()
    


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)


def embed_and_upsert(chunks, namespace):
    texts = [chunk.page_content for chunk in chunks]
    vectors = embedder.embed_documents(texts)
    records = []
    for vector , chunk in zip(vectors, chunks): 
        records.append({
            "id": str(uuid.uuid4()),
            "values": vector,
            "metadata": {
                "text": chunk.page_content,
                "source": chunk.metadata["source"],
                "page": chunk.metadata["page"]
            }
        })
    for i in range(0, len(records), 100):
        index.upsert(vectors=records[i:i+100], namespace=namespace)

    
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class PineconeRetriever(BaseRetriever):
    namespace: str
    k: int = 5

    def _get_relevant_documents(self, query) -> list[Document]:
        vector = embedder.embed_query(query)
        response = index.query(top_k=self.k, vector = vector, namespace=self.namespace, include_metadata=True)
        return [
            Document(
                page_content=m.metadata["text"],
                metadata={"source": m.metadata["source"], "page": m.metadata["page"]}
            )
            for m in response["matches"]
        ]


