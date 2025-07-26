import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
import os

# --- CONFIGURATION ---
# IMPORTANT: Replace this with the actual path where your fine-tuned model and tokenizer are saved.
MODEL_PATH = "/home/jaggu/Deep_L/fine_tuned_t5_spider_sql_generator"

# Path to tables.json (ASSUMPTION: tables.json is in the same directory as app.py OR in spider_dataset_extracted/spider_data/)
# Based on your find command, the full path is:
TABLES_JSON_PATH = os.path.expanduser("~/Deep_L/spider_dataset_extracted/spider_data/tables.json")
# --- END CONFIGURATION ---


# @st.cache_resource decorator
# This decorator ensures that the model and tokenizer are loaded only ONCE
# when the app starts, and then reused for subsequent user interactions.
@st.cache_resource
def load_model():
    """Loads the tokenizer and model from the specified path."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True) # Added use_fast=True
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
        if torch.cuda.is_available():
            model.to("cuda")
            st.success("Model loaded successfully to GPU!")
        else:
            st.success("Model loaded successfully to CPU.")
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        st.info("Please ensure your MODEL_PATH is correct and contains both model weights and tokenizer files (e.g., pytorch_model.bin, config.json, tokenizer.json).")
        return None, None

# Load the model and tokenizer
tokenizer, model = load_model()

# --- Load Database Schemas (Cached) ---
@st.cache_data
def load_db_schemas(path):
    """Loads database schemas from tables.json."""
    if not os.path.exists(path):
        st.error(f"Error: tables.json not found at {path}. Please ensure it's in the correct directory.")
        st.stop() # Stop the app if crucial data is missing
    try:
        with open(path, 'r', encoding='utf-8') as f:
            db_schemas = {db['db_id']: db for db in json.load(f)}
        return db_schemas
    except Exception as e:
        st.error(f"Error loading tables.json: {e}")
        st.stop()

db_schemas = load_db_schemas(TABLES_JSON_PATH)
st.success(f"Loaded {len(db_schemas)} database schemas.")


# --- Schema Representation Function (COPIED FROM evaluate_spider.py) ---
def get_schema_representation(db_id, db_schemas_dict):
    """
    Generates a textual representation of the database schema for a given db_id.
    Includes table names, column names, and their types.
    """
    schema = db_schemas_dict.get(db_id)
    if not schema:
        return f"Schema for '{db_id}' not found." # Should not happen if db_id is from selectbox

    schema_parts = []
    
    for table_idx, table_name_original in enumerate(schema['table_names_original']):
        schema_parts.append(f"table {table_idx}: {table_name_original}")
        
        table_cols = []
        for col_idx, (col_table_idx, col_name_original) in enumerate(schema['column_names_original']):
            if col_table_idx == table_idx:
                col_type = schema['column_types'][col_idx]
                table_cols.append(f"column {col_idx}: {col_name_original} ({col_type})")
        
        if table_cols:
            schema_parts.append("  " + "; ".join(table_cols))
            
    return " | ".join(schema_parts)


# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title("Text-to-SQL Generator (Spider Dataset)")

if tokenizer is None or model is None:
    st.warning("Model failed to load. Please check the console for errors and verify your MODEL_PATH.")
else:
    st.write("Enter a natural language question and select a database to convert it into an SQL query.")

    # Get list of database IDs for the selectbox
    db_ids_list = sorted(list(db_schemas.keys()))
    selected_db_id = st.selectbox("Select Database:", db_ids_list, index=db_ids_list.index("concert_singer") if "concert_singer" in db_ids_list else 0)

    user_question = st.text_area("Your Natural Language Question:", "How many singers are there?", height=100)
    
    if st.button("Generate SQL"):
        if user_question and selected_db_id:
            st.spinner("Generating SQL query...")
            
            # --- CRITICAL CHANGE: Construct input_text exactly as used during training ---
            schema_text = get_schema_representation(selected_db_id, db_schemas)
            input_text = f"generate sql: {schema_text} | question: {user_question}"
            # --- END CRITICAL CHANGE ---

            # Tokenize the input
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            
            # Move inputs to GPU if model is on GPU
            if torch.cuda.is_available():
                inputs = {key: value.to("cuda") for key, value in inputs.items()}

            # Generate the SQL query
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=512,
                num_beams=5,
                early_stopping=True
            )
            
            generated_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            st.subheader("Generated SQL Query:")
            st.code(generated_sql, language='sql')
        else:
            st.warning("Please enter a question and select a database.")

st.markdown("---")
st.markdown("This app uses a fine-tuned T5-based model on the Spider dataset.")
st.markdown(f"Model loaded from: `{MODEL_PATH}`")
st.markdown(f"Database schemas loaded from: `{TABLES_JSON_PATH}`")