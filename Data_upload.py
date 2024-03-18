from pinecone import Pinecone
import requests, os, random, string
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

hf_token = os.environ.get('HF_TOKEN')
embedding_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

def generate_embedding(text: str) -> list[float]:

	response = requests.post(
		embedding_url,
		headers={"Authorization": f"Bearer {hf_token}"},
		json={"inputs": text})

	if response.status_code != 200:
		raise ValueError(f"Request failed with status code {response.status_code}: {response.text}")
	return response.json()


def upsert_data(text,company,stipend,JobTitles,skills,link,namespace):
    pine = Pinecone(api_key=os.getenv('PINECONE_KEY'))
    index = pine.Index(os.getenv('PINECONE_INDEX'))
        
    metadata = {'stipend': stipend, 'JobTitles': JobTitles, "company": company, "skills": skills,"link": link}
    # Text to be embedded
    vector = generate_embedding(text)

    # Ids generation for vectors
    _id = ''.join(random.choices(string.ascii_letters + string.digits, k=10,))

    # Upserting vector into pinecone database
    index.upsert(vectors=[{"id":_id, "values": vector, "metadata": metadata}]
                    ,namespace = namespace)

    print("Vector upserted successfully")


if __name__ == '__main__':
    csv_path = 'newdata.csv'

