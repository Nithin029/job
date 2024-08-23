from data import BaseModel,Optional,List,Field,ChatPromptTemplate,driver,llm,vector_indices,COHERE_API,cohere
class Entities(BaseModel):
    """Identifying information about entities."""

    job_title: Optional[List[str]] = Field(
        None,
        description="Job titles identified in the text."
    )
    company: Optional[List[str]] = Field(
        None,
        description="Company names identified in the text."
    )
    job_location: Optional[List[str]] = Field(
        None,
        description="Locations identified in the text."
    )
    salary: Optional[List[str]] = Field(
        None,
        description="Salary details identified in the text."
    )

    job_type: Optional[List[str]] = Field(
        None,
        description="Job types identified in the text."
    )

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an experienced analyst with over 10 years of expertise in natural language processing. Your specialization lies in identifying key entities from user queries related to job postings. Your task is to extract these entities when present in the given text and format them in a specific way.

        ### Objective:
        Identify and extract the following entities from the user query if they are present:
        - Job Title
        - Company
        - Job Location
        - Salary
        - Job Type


        ### Guidelines:
        1. Read the user query carefully.
        2. Extract the relevant entities:
           - Job Title: Any mentioned position or role. Capitalize each word.
           - Company: Name of the organization or business. Use proper capitalization.Capitalize each word.
           - Job Location: Any geographical location related to the job. Capitalize each word.
           - Salary: Mention of compensation, whether specific amount or range. Keep as mentioned in the query.
           - Job Type: The type of employment. Use uppercase for the whole term (e.g.,  FULL_TIME, PART_TIME, OTHER,INTERN,PER_DIEM).
        3. Only include entities that are explicitly mentioned in the query.
        4. If an entity is not present, include an empty list for that key in the output.
        5. If multiple entities of the same type are present, include all in the list.
        6. Ensure precise extraction and avoid misidentification.
        7. Use single quotes for all keys and string values in the output JSON.
        8. Ensure there's a comma after each key-value pair except the last one.
        9. Do not include any spaces before or after the colons separating keys and values.

        Take a deep breath and work on this problem step-by-step.
        """
    ),
    (
        "human",
        "Extract job information from the following input: {question}"
    ),
])


def execute_query_search(query, parameters=None):
    with driver.session() as session:
        result = session.run(query, parameters)
        return [record.data() for record in result]


def build_query_and_params(filtered_result, job_title, limit):
    query_parts = [
        f"MATCH (j:Job {('WHERE j.cleansed_title = $job_title') if job_title else ''})",
        "OPTIONAL MATCH (c:Company)-[:POSTED]->(j)",
        "OPTIONAL MATCH (l:Location)<-[:LOCATION]-(c)",
        "OPTIONAL MATCH (jt:JobType)<-[:JOB_TYPE]-(j)",
        "WITH j, c, l, jt,"
    ]

    score_conditions = []
    match_conditions = []

    if 'company' in filtered_result and filtered_result['company']:
        score_conditions.append("CASE WHEN c.name IN $company THEN 5 ELSE 0 END")
        match_conditions.append("CASE WHEN c.name IN $company THEN 1 ELSE 0 END")
    if 'job_location' in filtered_result and filtered_result['job_location']:
        score_conditions.append(
            "CASE WHEN l.city IN $job_location OR l.county IN $job_location OR l.state IN $job_location THEN 6 ELSE 0 END")
        match_conditions.append(
            "CASE WHEN l.city IN $job_location OR l.county IN $job_location OR l.state IN $job_location THEN 1 ELSE 0 END")
    if 'job_type' in filtered_result and filtered_result['job_type']:
        score_conditions.append("CASE WHEN jt.type IN $job_type THEN 3 ELSE 0 END")
        match_conditions.append("CASE WHEN jt.type IN $job_type THEN 1 ELSE 0 END")
    if 'salary' in filtered_result and filtered_result['salary']:
        score_conditions.append("CASE WHEN j.salary_estimate >= $salary THEN 3 ELSE 0 END")
        match_conditions.append("CASE WHEN j.salary_estimate >= $salary THEN 1 ELSE 0 END")

    if score_conditions:
        query_parts.append("(" + " + ".join(score_conditions) + ") AS match_score,")
        query_parts.append("(" + " + ".join(match_conditions) + ") AS total_matches")
    else:
        query_parts.append("0 AS match_score, 0 AS total_matches")

    query_parts.extend([
        "RETURN j.title AS job_title, j.description AS job_description,",
        "j.cleansed_title AS cleansed_job_title, j.salary_estimate AS salary_estimate,",
        "c.name AS company_name, l.location AS location_name,  jt.type AS job_type,",
        "match_score, total_matches",
        "ORDER BY total_matches DESC, match_score DESC",
        f"LIMIT {limit}"
    ])

    query = " ".join(query_parts)
    if job_title:
        filtered_result['job_title'] = job_title
    return query, filtered_result


def get_filtered_jobs(filtered_result, job_titles, total_limit=100):
    all_results = []
    per_title_limit = total_limit // len(job_titles)
    remaining_limit = total_limit % len(job_titles)

    for i, job_title in enumerate(job_titles):
        current_limit = per_title_limit + (1 if i < remaining_limit else 0)
        query, parameters = build_query_and_params(filtered_result, job_title, current_limit)
        results = execute_query_search(query, parameters)
        all_results.extend(results)

    # If we didn't get enough results, we can fill with other relevant jobs
    if len(all_results) < total_limit:
        general_query, parameters = build_query_and_params(filtered_result, None, total_limit - len(all_results))
        additional_results = execute_query_search(general_query, parameters)
        all_results.extend(additional_results)

    return all_results


def similarity_search_titles(user_input):
    result = vector_indices["jobs_title_embedding"].similarity_search(user_input)
    if result:
        return [record.metadata['cleansed_title'] for record in result]
    else:
        print("No similar titles found.")
        return []


def balance_and_limit_results(results, job_titles, limit=20):
    job_title_dict = {title: [] for title in job_titles}
    other_results = []

    for job in results:
        if job['cleansed_job_title'] in job_titles:
            job_title_dict[job['cleansed_job_title']].append(job)
        else:
            other_results.append(job)

    balanced_results = []
    per_title_limit = limit // len(job_titles)
    remaining_limit = limit % len(job_titles)

    for i, title in enumerate(job_titles):
        current_limit = per_title_limit + (1 if i < remaining_limit else 0)
        balanced_results.extend(job_title_dict[title][:current_limit])

    # Fill any remaining slots with other results
    remaining_slots = limit - len(balanced_results)
    if remaining_slots > 0:
        for title in job_titles:
            while remaining_slots > 0 and job_title_dict[title]:
                balanced_results.append(job_title_dict[title].pop(0))
                remaining_slots -= 1

        balanced_results.extend(other_results[:remaining_slots])

    return balanced_results


def classify_prompt(prompt, classifier_type):
    cohere_client = cohere.Client(COHERE_API)
    examples = {
        "medical_job_seeking": [
            {"text": "Looking for a nurse position in a pediatric ward.", "label": "medical"},
            {"text": "What are the qualifications for a radiology technician?", "label": "medical"},
            {"text": "Seeking a job as a general practitioner.", "label": "medical"},
            {"text": "How to become a medical assistant?", "label": "medical"},
            {"text": "What is the process to apply for a medical research position?", "label": "medical"},
            {"text": "What are the requirements for becoming a hospital administrator?", "label": "medical"},
            {"text": "How to write a cover letter for a software developer?", "label": "non-medical"},
            {"text": "What skills are required for a project manager?", "label": "non-medical"},
            {"text": "Tell me about the latest trends in AI.", "label": "non-medical"},
            {"text": "What is the best strategy for marketing?", "label": "non-medical"},
        ]
    }

    response = cohere_client.classify(
        model='large',
        inputs=[prompt],
        examples=examples[classifier_type]
    )
    return response.classifications[0].prediction


def process_search_query(question):
    global similarity_titles
    classification = classify_prompt(question, "medical_job_seeking")

    if classification == "medical":
        entity_chain = prompt | llm.with_structured_output(Entities)
        result = entity_chain.invoke({"question": question})
        filtered_result = result.dict()
        print(filtered_result)

        original_job_titles = filtered_result.get('job_title', [])

        if original_job_titles:
            similarity_titles = []
            for job_title in original_job_titles:
                similarity_titles.extend(similarity_search_titles(job_title))
            filtered_result['job_title'] = list(set(similarity_titles))  # Remove duplicates
        else:
            filtered_result['job_title'] = []

        results = get_filtered_jobs(filtered_result, filtered_result['job_title'], total_limit=100)
        print(results)
        driver.close()

        # Balance and limit the results
        final_results = balance_and_limit_results(results, similarity_titles , limit=20)

        return final_results
    else:
        return "non-medical"


def main():
    # For inserting the data into the Neo4j Database
    # create_nodes_and_relationships(df)
    # create_vector_index(driver,dimension)
    # process_and_insert_embeddings(df)

    # For searching the Knowledge Graph
    question = input("Enter your query: ")
    final_results = process_search_query(question)
    return final_results


if __name__ == "__main__":
    main()



