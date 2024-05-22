import streamlit as st
import os
import psycopg2
from langchain.sql_database import SQLDatabase
from langchain_openai import OpenAI
from langchain_experimental.sql import SQLDatabaseChain
from api_key import apikey

os.environ["OPENAI_API_KEY"] = apikey

openai_key = os.getenv('OPENAI_API_KEY')




# Frontend logic: Define layout and interaction components
def main():
    st.title("AI Database Q&A Agent")
    
    # Database URI input
    db_uri = st.text_input("Enter your database URI:")
    
    # Question input
    question = st.text_area("Ask your question:")

    # Button to trigger backend logic
    if st.button("Submit"):
        if db_uri and question:
            # Call backend logic function
            answer = process_question(db_uri, question)
            st.write("Question: \n",question)
            # Display answer
            st.write("Answer: \n", answer)
        else:
            st.write("Please provide both a database URI and a question.")

# Backend logic: Process user question using database
def process_question(db_uri, question):
    
    if (not openai_key):
        raise Exception('No OpenAI API key provided. Please set the OPENAI_API_KEY environment variable or provide it when prompted.')

    st.write('OpenAI API key set.')

    def prepare_agent_prompt(input_text):
        agent_prompt = f"""
        Generate a PostgreSQL query based on the provided input.

        Ensure that the query uses PostgreSQL syntax.
        Utilize the shoe_color enum to filter by color, ensuring that only valid color values are queried.
        Similarly, utilize the shoe_width enum to filter by width, ensuring that only valid width values are queried.
        Note that the color and width columns are of array types, while the name column is of type VARCHAR.

        An example query using an array column would be:
        SELECT * FROM products, unnest(color) AS col WHERE col::text % SOME_COLOR;
        or
        SELECT * FROM products, unnest(width) AS wid WHERE wid::text % SOME_WIDTH;

        An example query using the name column would be:
        SELECT * FROM products WHERE name ILIKE '%input_text%';

        It's not necessary to search on all columns, only those relevant to the query.

        Generate a PostgreSQL query using the provided input:
        {input_text}

        Respond as a human would.
    """


        return agent_prompt



    # Initialize the OpenAI's agent
    openai = OpenAI(
        api_key=openai_key,
        temperature=0, 
        max_tokens=-1 # -1 returns as many tokens as possible given the prompt and the models maximal context size
        )

    # Initialize LangChain's database agent
    database = SQLDatabase.from_uri(
        db_uri, 
        include_tables=["products", "users", "purchases", "product_inventory"])

    # Initialize LangChain's database chain agent
    db_chain = SQLDatabaseChain.from_llm(openai, db=database, verbose=True, use_query_checker=True, return_intermediate_steps=True)

    agent_prompt = prepare_agent_prompt(question)


    try:
        result = db_chain.invoke(agent_prompt)
        return result['result']
    except (Exception, psycopg2.Error) as error:
        return error


if __name__ == "__main__":
    main()
