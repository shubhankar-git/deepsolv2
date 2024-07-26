
import re
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_core.documents import Document
import pymupdf4llm


# Define your OpenAI API key
OPENAI_API_KEY = 'Your Api Key'

def basic_cleaner(page: str):
    page = re.sub(r"\n+", "\n", page)
    page_lines = page.split("\n")
    temp_page = []
    for i in range(len(page_lines)):
        if page_lines[i] == "":
            continue
        elif i + 1 < len(page_lines):
            if page_lines[i] not in page_lines[i + 1]:
                temp_page.append(page_lines[i])
        elif i == len(page_lines) - 1:
            temp_page.append(page_lines[i])
    page = "\n".join([x for x in temp_page if x != "-----" or x != ""])
    return page

def remove_images(page: str):
    page = re.sub(r"!\[.*\.(jpg|png|gif|bmp|svg|webp)\]\(.*\)", "", page)
    return page

class PDFMarkdownProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.pages = pymupdf4llm.to_markdown(pdf_path, page_chunks=True, write_images=False)
        if len(self.pages) == 0:
            raise ValueError("No content found in the pdf")
        self.doc_metadata = self.extract_metadata(self.pages[0].metadata) if hasattr(self.pages[0], 'metadata') else None
        self.page_text = [p["text"] for p in self.pages]
        if self.doc_metadata:
            md_metadata = ""
            for k, v in self.doc_metadata.items():
                md_metadata += f"{k}: {v}\n"
            self.page_text[0] = f'{md_metadata}\n{self.page_text[0]}'
        self.page_text = self.clean_text()
        self.documents = self.make_documents()
    
    @staticmethod
    def extract_metadata(metadata: dict[str, str]):
        metadata_keys = ["title", "author", "subject", "keywords", "creator"]
        return {key: metadata.get(key) for key in metadata_keys}
    
    def clean_text(self):
        self.page_text = [basic_cleaner(p) for p in self.page_text]
        self.page_text = [remove_images(p) for p in self.page_text]
        return self.page_text
    
    def make_documents(self):
        return [Document(page_content=page, metadata={"page_number": i}) for i, page in enumerate(self.page_text)]
    
    def get_documents(self):
        return self.documents

def save_faiss_index(documents, file_path="vectorstore.index"):
    
    embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(documents, embedding)
    

    vectorstore.save_local("faiss_store")
    
    


pdf_path = './pdfs/Apple_Vision_Pro_Privacy_Overview.pdf'  
loader = PDFMarkdownProcessor(pdf_path)
documents = loader.get_documents()

save_faiss_index(documents)