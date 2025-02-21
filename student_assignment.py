# pip install chromadb pandas

import datetime
import chromadb
import pandas as pd
import traceback

from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

csv_file = "COA_OpenData.csv"
dbpath = "./"

def generate_hw01():
    chroma_client = chromadb.PersistentClient(path=dbpath)
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine", "hnsw:batch_size": 200},
        embedding_function=embedding_functions.OpenAIEmbeddingFunction(
            api_key = gpt_emb_config['api_key'],
            api_base = gpt_emb_config['api_base'],
            api_type = gpt_emb_config['openai_type'],
            api_version = gpt_emb_config['api_version'],
            deployment_id = gpt_emb_config['deployment_name']
        )
    )
    data_csv = pd.read_csv(csv_file)
    if collection.count() != data_csv.shape[0]:
        for _, row in data_csv.iterrows():
            result = collection.get(ids=[str(row["ID"])])
            if not result or not result["ids"]:
                collection.add(
                    documents=[row['HostWords']],
                    metadatas=[{
                        'file_name':csv_file,
                        'name':row['Name'],
                        'type':row['Type'],
                        'address':row['Address'],
                        'tel':row['Tel'],
                        'city':row['City'],
                        'town':row['Town'],
                        'date':int(datetime.datetime.strptime(row['CreateDate'], '%Y-%m-%d').timestamp())
                    }],
                    ids=[str(row["ID"])]
                )
                print(f'Add {row["Name"]}. Collection count = {collection.count()}')
    return collection

def filter_score(results, score):
    distance = 1.0 - score
    results = sorted([
        (dist, metadata)
        for dist, metadata in zip(results['distances'][0], results['metadatas'][0])
        if dist <= 0.2
    ], key=lambda item: item[0])
    return [result[1].get('new_store_name', result[1]['name']) for result in results]

def generate_hw02(question, city, store_type, start_date, end_date):
    results = generate_hw01().query(
        query_texts=[question],
        n_results=10,
        where={
            '$and': [
                {'city': {'$in': city}},
                {'type': {'$in': store_type}},
                {'date': {'$gte': int(start_date.timestamp())}}, 
                {'date': {'$lte': int(end_date.timestamp())}}
            ]
        }
    )
    return [item[1]['name'] for item in sorted([
        (dist, metadata)
        for dist, metadata in zip(results['distances'][0], results['metadatas'][0])
        if dist <= 0.2
    ], key=lambda item: item[0])]

def generate_hw03(question, store_name, new_store_name, city, store_type):
    collection = generate_hw01()
    results = collection.query(
        query_texts=[store_name],
        where={'name': {'$eq': store_name}}
    )
    collection.update(
        ids=results['ids'][0],
        metadatas=[
            {**metadata, 'new_store_name': new_store_name}
            for metadata in results['metadatas'][0]
        ]
    )
    results = collection.query(
        query_texts=[question],
        n_results=10,
        where={
            '$and': [
                {'city': {'$in': city}},
                {'type': {'$in': store_type}}
            ]
        }
    )
    return [item[1].get('new_store_name', item[1]['name']) for item in sorted([
        (dist, metadata)
        for dist, metadata in zip(results['distances'][0], results['metadatas'][0])
        if dist <= 0.2
    ], key=lambda item: item[0])]
    
def demo(question):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    
    return collection

print(generate_hw01().count())
print(generate_hw02("我想要找有關茶餐點的店家", ["宜蘭縣", "新北市"], ["美食"], datetime.datetime(2024, 4, 1), datetime.datetime(2024, 5, 1)))
print(generate_hw03("我想要找南投縣的田媽媽餐廳，招牌是蕎麥麵", "耄饕客棧", "田媽媽（耄饕客棧）", ["南投縣"], ["美食"]))