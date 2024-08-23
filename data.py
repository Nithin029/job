import pandas as pd
from neo4j import GraphDatabase
from together import Together
from langchain_community.graphs import Neo4jGraph
import openai
import os
import pandas as pd
from typing import List, Dict
import time
import cohere
from langchain_groq import ChatGroq
from langchain_together.embeddings import TogetherEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from  typing import Tuple,List,Optional
from langchain_core.pydantic_v1 import BaseModel,Field
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
COHERE_API = os.getenv("COHERE_API")
uri = os.getenv("NEO4J_URI")
username= os.getenv("NEO4J_USERNAME")
password= os.getenv("NEO4J_PASSWORD")
together = Together(api_key=TOGETHER_API_KEY)
driver = GraphDatabase.driver(uri, auth=(username, password))
df = pd.read_csv('DB_search_clean.csv')
neo4j_graph = Neo4jGraph(url=uri, username=username, password=password)
dimension=768
client = openai.OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=TOGETHER_API_KEY
)
embeddings = TogetherEmbeddings(model="BAAI/bge-base-en-v1.5",together_api_key=TOGETHER_API_KEY)
llm = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name="llama3-70b-8192")

def execute_query(query, parameters=None):
    with driver.session() as session:
        session.run(query, parameters)

def create_nodes_and_relationships(df):
    for _, row in df.iterrows():
        job_id = row['job_id']
        company_name = row['company_name']
        jobtitle = row['jobtitle']
        cleansed_title = row['cleansed_title']
        jobdescription_html = row['jobdescription_html']
        city = row['city'] if pd.notna(row['city']) else "Unknown City"
        state_name = row['state_name'] if pd.notna(row['state_name']) else "Unknown State"
        county_name = row['county_name'] if pd.notna(row['county_name']) else "Unknown County"
        jobtype = row['jobtype'] if pd.notna(row['jobtype']) else "Unknown Job Type"
        salary_estimate = row['salary_estimate'] if pd.notna(row['salary_estimate']) else "Unknown Salary"

        # Create Company Node
        company_query = "MERGE (c:Company {name: $company_name})"
        execute_query(company_query, {'company_name': company_name})

        # Create Job Node
        job_query = """
        MERGE (j:Job {id: $job_id, title: $jobtitle, cleansed_title: $cleansed_title,
                      description: $jobdescription_html, salary_estimate: $salary_estimate})
        """
        execute_query(job_query, {'job_id': job_id, 'jobtitle': jobtitle, 'cleansed_title': cleansed_title,
                                  'jobdescription_html': jobdescription_html, 'salary_estimate': salary_estimate})

        # Create JobType Node
        jobtype_query = "MERGE (jt:JobType {type: $jobtype})"
        execute_query(jobtype_query, {'jobtype': jobtype})

        # Create Location Node
        location_query = """
        MERGE (l:Location {city: $city, state: $state_name, county: $county_name})
        """
        execute_query(location_query, {'city': city, 'state_name': state_name, 'county_name': county_name})

        # Create Relationships
        posted_by_query = "MATCH (c:Company {name: $company_name}), (j:Job {id: $job_id}) MERGE (c)-[:POSTED]->(j)"
        execute_query(posted_by_query, {'company_name': company_name, 'job_id': job_id})

        location_relationship_query = "MATCH (c:Company {name: $company_name}), (l:Location {city: $city, state: $state_name, county: $county_name}) MERGE (c)-[:LOCATION]->(l)"
        execute_query(location_relationship_query, {'company_name': company_name, 'city': city, 'state_name': state_name, 'county_name': county_name})

        job_description_query = "MATCH (j:Job {id: $job_id}), (c:Company {name: $company_name}) MERGE (j)-[:JOB_DESCRIPTION]->(c)"
        execute_query(job_description_query, {'job_id': job_id, 'company_name': company_name})

        job_type_relationship_query = "MATCH (j:Job {id: $job_id}), (jt:JobType {type: $jobtype}) MERGE (j)-[:JOB_TYPE]->(jt)"
        execute_query(job_type_relationship_query, {'job_id': job_id, 'jobtype': jobtype})

        title_relationship_query = "MATCH (j:Job {id: $job_id}) SET j.cleansed_title = $cleansed_title"
        execute_query(title_relationship_query, {'job_id': job_id, 'cleansed_title': cleansed_title})

def drop_index(driver, index_name: str) -> None:
    """Drop the specified index if it exists."""
    drop_query = f"DROP INDEX {index_name} IF EXISTS"
    try:
        with driver.session() as session:
            session.run(drop_query)
        print(f"Successfully dropped index '{index_name}'")
    except Exception as e:
        print(f"Error dropping index '{index_name}': {e}")

def create_vector_index(driver, dimension: int) -> None:
    def run_query(index_name: str, label: str, property_name: str) -> None:
        index_query = f"""
        CALL db.index.vector.createNodeIndex('{index_name}', '{label}', '{property_name}', $dimension, 'cosine')
        """
        try:
            with driver.session() as session:
                session.run(index_query, {"dimension": dimension})
            print(f"Successfully created vector index for '{index_name}' on property '{property_name}'")
        except Exception as e:
            if 'already exists' in str(e).lower():
                print(f"Index '{index_name}' already exists.")
            else:
                print(f"Error creating vector index for '{index_name}': {e}")

    # Drop existing indexes
    #drop_index(driver, 'jobs_title_embedding')
    #drop_index(driver, 'jobs_cleansed_title_embedding')
    #drop_index(driver, 'companies')
    #drop_index(driver, 'locations')
    #drop_index(driver, 'jobtypes')

    # Create new indexes
    run_query('jobs_title_embedding', 'Job', 'title_embedding')
    run_query('jobs_cleansed_title_embedding', 'Job', 'cleansed_title_embedding')
    run_query('companies', 'Company', 'embedding')
    run_query('locations', 'Location', 'embedding')
    run_query('jobtypes', 'JobType', 'embedding')

def generate_embedding(text: str) -> Dict[str, List[float]]:
    """Generate embedding for a single text and return as a dictionary."""
    try:
        response = client.embeddings.create(input=text, model="BAAI/bge-large-en-v1.5")
        embedding = response.data[0].embedding  # Assuming single embedding per text
        print(f"Successfully generated embedding for text: {text}")
        return {text: embedding}
    except Exception as e:
        print(f"Error generating embedding for text: {text}. Error: {e}")
        return {text: None}

def generate_embeddings(texts: List[str]) -> Dict[str, List[float]]:
    """Generate embeddings for a list of texts sequentially and handle rate limiting."""
    embeddings = {}
    queries_count = 0
    start_time = time.time()

    for text in texts:
        # Check the elapsed time and queries count
        elapsed_time = time.time() - start_time
        if queries_count >= 60 and elapsed_time < 60:
            sleep_time = 60 - elapsed_time
            print(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds.")
            time.sleep(sleep_time)
            queries_count = 0
            start_time = time.time()

        # Generate embedding
        result = generate_embedding(text)
        queries_count += 1

        # Collect result
        for text, embedding in result.items():
            if embedding is not None:
                embeddings[text] = embedding

    return embeddings

def update_node_with_embedding_job(node_id: int, title_embedding: List[float], cleansed_title_embedding: List[float]) -> None:
    """Update a Neo4j Job node with separate embeddings for title and cleansed title."""
    try:
        query = """
        MATCH (n:Job {id: $node_id})
        SET n.title_embedding = $title_embedding,
            n.cleansed_title_embedding = $cleansed_title_embedding
        RETURN n
        """

        params = {
            "node_id": node_id,  # Ensure node_id is an integer
            "title_embedding": title_embedding,
            "cleansed_title_embedding": cleansed_title_embedding
        }

        with driver.session() as session:
            result = session.run(query, params)
            updated_node = result.single()
            if updated_node:
                print(f"Successfully updated Job node {node_id} with title and cleansed title embeddings")
            else:
                print(f"Failed to update Job node {node_id} with title and cleansed title embeddings")
    except Exception as e:
        print(f"Error updating Job node {node_id} with embeddings: {e}")

def update_node_with_embedding(node_identifier: Dict[str, str], embedding: List[float], node_label: str, identifier_fields: List[str]) -> None:
    """Update a Neo4j node with the given embedding based on a list of identifier fields."""
    try:
        # Construct the WHERE clause for multiple identifier fields
        where_clause = " AND ".join([f"n.{field} = $node_{field}" for field in identifier_fields])

        query = f"""
        MATCH (n:{node_label})
        WHERE {where_clause}
        SET n.embedding = $embedding
        RETURN n
        """

        # Prepare parameters
        params = {"embedding": embedding}
        for field in identifier_fields:
            params[f"node_{field}"] = node_identifier[field]

        with driver.session() as session:
            result = session.run(query, params)
            updated_node = result.single()
            if updated_node:
                print(f"Successfully updated {node_label} node with identifiers {node_identifier}")
            else:
                print(f"Failed to update {node_label} node with identifiers {node_identifier}")
    except Exception as e:
        print(f"Error updating {node_label} node with identifiers {node_identifier} and embedding: {e}")

def process_and_insert_embeddings(df: pd.DataFrame) -> None:
    """Process text fields in DataFrame, generate embeddings, and insert into Neo4j."""
    # Extract unique values for each node type
    #job_titles = df['jobtitle'].unique().tolist()
    #cleansed_titles = df['cleansed_title'].unique().tolist()
    #company_names = df['company_name'].unique().tolist()
    locations = df[['city', 'county_name', 'state_name']].apply(lambda x: {
        "city": x['city'] if pd.notna(x['city']) else 'Unknown City',
        "county": x['county_name'] if pd.notna(x['county_name']) else 'Unknown County',
        "state": x['state_name'] if pd.notna(x['state_name']) else 'Unknown State'
    }, axis=1).drop_duplicates().tolist()
    #job_types = df['jobtype'].unique().tolist()

    # Generate embeddings for unique values
    #job_title_embeddings = generate_embeddings(job_titles)
    #cleansed_title_embeddings = generate_embeddings(cleansed_titles)
    #company_embeddings = generate_embeddings(company_names)
    location_texts = [f"{loc['city']}, {loc['county']}, {loc['state']}" for loc in locations]
    location_embeddings = generate_embeddings(location_texts)
    #job_type_embeddings = generate_embeddings(job_types)

    # Create mappings from text to embeddings
    #job_title_mapping = {title: emb for title, emb in job_title_embeddings.items()}
    #cleansed_title_mapping = {title: emb for title, emb in cleansed_title_embeddings.items()}
    #company_mapping = {name: emb for name, emb in company_embeddings.items()}
    location_mapping = {text: embedding for text, embedding in location_embeddings.items()}
    #job_type_mapping = {job_type: emb for job_type, emb in job_type_embeddings.items()}

    # Track updated nodes to avoid duplicates
    updated_nodes = {
        "Job": set(),
        "Company": set(),
        "Location": set(),
        "JobType": set()
    }

    # Insert embeddings into Neo4j nodes
    for index, row in df.iterrows():
        try:
            job_id = int(row['job_id'])  # Convert to integer if necessary
            #job_title = row['jobtitle']
            #cleansed_title = row['cleansed_title']
            #company_name = row['company_name']
            city = row['city'] if pd.notna(row['city']) else 'Unknown City'
            county = row['county_name'] if pd.notna(row['county_name']) else 'Unknown County'
            state = row['state_name'] if pd.notna(row['state_name']) else 'Unknown State'
            #job_type = row['jobtype']

            # Construct location_str
            location_str = ', '.join([str(city), str(county), str(state)])

            # Update job nodes with job titles and cleansed titles
            #if job_title in job_title_mapping and cleansed_title in cleansed_title_mapping:
              #  title_emb = job_title_mapping[job_title]
               # cleansed_title_emb = cleansed_title_mapping[cleansed_title]
                #if title_emb and cleansed_title_emb and job_id not in updated_nodes["Job"]:
                 #   update_node_with_embedding_job(job_id, title_emb, cleansed_title_emb)
                  #  updated_nodes["Job"].add(job_id)

            # Update company and location nodes
            #if company_name in company_mapping and company_mapping[company_name] and company_name not in updated_nodes["Company"]:
             #   update_node_with_embedding({"name": company_name}, company_mapping[company_name], 'Company', ['name'])
              #  updated_nodes["Company"].add(company_name)

            if location_str in location_mapping and location_mapping[location_str] and location_str not in updated_nodes["Location"]:
                loc = {"city": city, "county": county, "state": state}
                update_node_with_embedding(loc, location_mapping[location_str], 'Location', ['city', 'county', 'state'])
                updated_nodes["Location"].add(location_str)

            # Update job type nodes
            #if job_type in job_type_mapping and job_type_mapping[job_type] and job_type not in updated_nodes["JobType"]:
             #   update_node_with_embedding({"type": job_type}, job_type_mapping[job_type], 'JobType', ['type'])
              #  updated_nodes["JobType"].add(job_type)

        except Exception as e:
            print(f"Error processing row {index} with job_id {job_id}: {e}")


vector_indices = {
    "jobs_title_embedding": Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=uri,
        username=username,
        password=password,
        search_type="hybrid",
        index_name="jobs_title_embedding",
        keyword_index_name="title_index",
        embedding_node_property="title_embedding",
        text_node_property="title",
    ),
    "companies": Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=uri,
        username=username,
        password=password,
        search_type="hybrid",
        index_name="companies",
        keyword_index_name="company_index",
        embedding_node_property="embedding",
        text_node_property="company_name",
    ),
    "locations": Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=uri,
        username=username,
        password=password,
        search_type="hybrid",
        index_name="locations",
        keyword_index_name="location_index",
        embedding_node_property="embedding",
        text_node_property="location_name",
    ),
    "jobtypes": Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=uri,
        username=username,
        password=password,
        search_type="hybrid",
        index_name="jobtypes",
        keyword_index_name="jobtype_index",
        embedding_node_property="embedding",
        text_node_property="jobtype",
    ),
}
