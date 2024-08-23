import streamlit as st
from output import process_search_query

st.set_page_config(layout="wide")

# Custom CSS for better styling and space utilization
custom_css = """
<style>
    .stApp {
        max-width: 100%;
        padding-top: 2rem;
    }
    .job-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 10px;
    }
    .job-box {
        background-color: #2c3e50;
        color: white;
        padding: 10px;
        border-radius: 5px;
        border: 2px solid #34495e; /* Border added for visual distinction */
        height: 100%;
        cursor: pointer;
    }
    .job-box:hover {
        background-color: #34495e;
        border: 2px solid #1abc9c; /* Change border color on hover */
    }
    .job-title {
        font-weight: bold;
        margin-bottom: 5px;
    }
    .job-company, .job-location, .job-salary {
        font-size: 0.9em;
        margin-bottom: 2px;
    }
    .job-details {
        background-color: #34495e;
        color: white;
        padding: 15px;
        border-radius: 5px;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

def display_job_results(results):
    # Use a list of unique identifiers for the selectbox
    st.session_state.job_titles = [f"{result['job_title']} - {result['company_name']} ({result['location_name']})" for result in results]
    st.session_state.job_results = results

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="job-grid">', unsafe_allow_html=True)
        for i, result in enumerate(results):
            job_html = f"""
            <div class="job-box" onclick="document.getElementById('job-select').value = '{i}'; document.getElementById('job-select').dispatchEvent(new Event('change'));">
                <div class="job-title">{result['job_title']}</div>
                <div class="job-company">{result['company_name']}</div>
                <div class="job-location">{result['location_name']}</div>
                <div class="job-salary">{result['salary_estimate']}</div>
            </div>
            """
            st.markdown(job_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        selected_index = st.selectbox("Select a job", options=list(range(len(st.session_state.job_titles))), format_func=lambda x: st.session_state.job_titles[x], key='job-select')
        if selected_index is not None:
            selected_result = st.session_state.job_results[selected_index]
            st.markdown('<div class="job-details">', unsafe_allow_html=True)
            st.subheader(selected_result['job_title'])
            st.write(f"**Company:** {selected_result['company_name']}")
            st.write(f"**Location:** {selected_result['location_name']}")
            st.write(f"**Salary Estimate:** {selected_result['salary_estimate']}")
            st.write(f"**Job Type:** {selected_result['job_type']}")
            st.write("**Job Description:**")
            st.write(selected_result.get('job_description', 'Not specified'))
            st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Initialize session state attributes if not already present
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'query' not in st.session_state:
        st.session_state.query = ''
    if 'job_titles' not in st.session_state:
        st.session_state.job_titles = []
    if 'job_results' not in st.session_state:
        st.session_state.job_results = []

    st.markdown("<h1 style='text-align: center;'>Job Search App</h1>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        query = st.text_input("Enter your job query:", value=st.session_state.query)
        search_button = st.button("Search")

    if search_button:
        st.session_state.query = query
        st.session_state.results = process_search_query(query)
        st.session_state.job_titles = []
        st.session_state.job_results = []

    if st.session_state.results == "non-medical":
        st.info("It seems your query is not related to medical job searching. Please enter a query related to medical job opportunities.")
    elif st.session_state.results:
        display_job_results(st.session_state.results[:100])  # Display up to 20 results
    elif st.session_state.query and search_button:
        st.warning("No results found for your query.")

if __name__ == '__main__':
    main()
