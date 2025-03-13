from flask import Flask, request, jsonify
import pandas as pd
import google.generativeai as genai
from pymongo import MongoClient
import json
from fuzzywuzzy import fuzz
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Initialize Flask app
app = Flask(__name__)

# Configure Gemini API key (Replace with your API key)
genai.configure(api_key="AIzaSyCjyVoiwVAzcGZ8-dtSfdEJnNADqSZnwMY")

# MongoDB Connection
client = MongoClient("mongodb://localhost:27017")
db = client.get_database("capstone")

# Sentence transformer model for vectorization
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # A pre-trained model for sentence embeddings

def fetch_and_merge_data():
    """Fetch and merge data from MongoDB."""
    try:
        payment_projection_data = list(db["payment_projection"].find())
        debt_information_data = list(db["debt_information"].find())
        if not payment_projection_data or not debt_information_data:
            return None
        payment_projection_df = pd.DataFrame(payment_projection_data)
        debt_information_df = pd.DataFrame(debt_information_data)
        merged_df = pd.merge(payment_projection_df, debt_information_df, on="Debt_Info_Key", how="left")
        return merged_df
    except Exception as e:
        return {"error": f"Error fetching or merging data: {str(e)}"}

def normalize_string(s):
    """Lowercase and remove non-alphanumeric characters."""
    return ''.join(ch for ch in s.lower() if ch.isalnum())

def match_to_column(name, columns, threshold=70):
    """Dynamically match the provided name to the best fitting column in the list."""
    # Check for manual mapping first
    manual_mappings = {
        "year": "Repayment_Year",  # Map 'year' to 'Repayment_Year'
    }
    norm_name = normalize_string(name)

    # Check for manual mapping
    if norm_name in manual_mappings:
        return manual_mappings[norm_name]
    
    # Fuzzy matching as fallback
    best_match = None
    best_score = 0
    for col in columns:
        score = fuzz.ratio(name.lower(), col.lower())
        if score > best_score:
            best_score = score
            best_match = col
    return best_match if best_score >= threshold else None

def vectorized_match_to_column(name, columns, threshold=0.7):
    """Use vectorization (semantic matching) to find the best column match."""
    model = TfidfVectorizer().fit_transform([name] + columns)
    similarity_matrix = cosine_similarity(model[0:1], model[1:])
    max_sim_index = similarity_matrix.argmax()
    similarity_score = similarity_matrix[0, max_sim_index]
    
    if similarity_score >= threshold:
        return columns[max_sim_index]
    else:
        return None

def format_field_name(field_name):
    """Format field names to uppercase with underscores."""
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', field_name).replace(' ', '_').upper()

def extract_intent_from_llm(query):
    """Extract operation, filters, and fields from the user query using Gemini 1.5 Flash."""
    system_prompt = """
    You are an AI financial assistant that translates user queries into a structured JSON query.
    The JSON must contain:
      - "operation": one of [sum, total, average, min, max, lowest, highest]
      - "filters": an object with any filters the user specifies (keys can be arbitrary)
      - "fields": an array of field names the user is interested in.
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            [system_prompt, f"User query: {query}"],
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 300
            }
        )

        # Print the raw response for debugging
        print("Raw Gemini Response:", response)

        candidates = response.candidates
        if not candidates:
            return {"error": "No candidates found in the response."}

        content = candidates[0].content.parts[0].text.strip()

        # Remove Markdown code block if present
        if content.startswith("```json") and content.endswith("```"):
            content = content[7:-3].strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse JSON: {str(e)}"}

    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse JSON: {str(e)}"}
    except Exception as e:
        return {"error": f"Failed to extract intent from query: {str(e)}"}


def value_matches(cell_value, filter_value):
    """Check if normalized filter value is a substring of the normalized cell value."""
    return normalize_string(str(filter_value)) in normalize_string(str(cell_value))

def apply_filters(data, filters):
    print("Applying filters:", filters)
    columns = data.columns.tolist()
    applied_filters = False
    for key, value in filters.items():
        matched_col = match_to_column(key, columns) or vectorized_match_to_column(key, columns)
        if matched_col:
            # Apply strict matching based on filter type (e.g., Debt_Info_Key)
            if 'debtinfokey' in key.lower():  # Ensure exact match for Debt_Info_Key
                value = f"Debt_Info_Key{value}"  # Correctly format value if it's related to Debt_Info_Key
                mask = data['Debt_Info_Key'] == value
            else:
                mask = data[matched_col].apply(lambda x: value_matches(x, value))
            data = data[mask]
            applied_filters = True
            print(f"Applied filter '{key}' -> '{matched_col}' with value '{value}', rows: {len(data)}")
        else:
            print(f"Filter key '{key}' did not match any column.")
    
    if not applied_filters:
        print("No filters were applied.")
    
    return data

def apply_operation(data, operation, field, filters=None):
    columns = data.columns.tolist()
    matched_col = match_to_column(field, columns) or vectorized_match_to_column(field, columns)
    if not matched_col:
        print(f"Field '{field}' not matched to any column.")
        return None
    if data.empty:
        print("No data available for operation.")
        return None

    # Apply filters if present
    if filters:
        data = apply_filters(data, filters)
    
    # Check if there is any valid data after applying filters
    if data.empty:
        print(f"No data available after applying filters {filters}.")
        return None

    # For numeric fields, apply operations like sum, average, etc.
    if pd.api.types.is_numeric_dtype(data[matched_col]):
        numeric_series = pd.to_numeric(data[matched_col], errors='coerce')
        if numeric_series.dropna().empty:
            print(f"Column '{matched_col}' has no valid numeric data. Skipping calculation.")
            return None

        # Perform the operation requested
        if operation == "average":
            result = numeric_series.mean()
            print(f"Average operation on '{matched_col}': {result}")
            return result
        elif operation == "sum":
            result = numeric_series.sum()
            print(f"Sum operation on '{matched_col}': {result}")
            return result
        elif operation == "min":
            result = numeric_series.min()
            print(f"Min operation on '{matched_col}': {result}")
            return result
        elif operation == "max":
            result = numeric_series.max()
            print(f"Max operation on '{matched_col}': {result}")
            return result

    # If the matched column is non-numeric (i.e., string values), return unique values
    else:
        print(f"Column '{matched_col}' is non-numeric. Returning string values.")
        return data[matched_col].dropna().unique().tolist()

    return None

def process_query():
    data = fetch_and_merge_data()
    if data is None:
        return jsonify({"error": "Failed to fetch or merge data."})
    
    user_query = request.json.get("query", "")
    print(f"Received user query: {user_query}")
    
    # Step 1: Extract intent from the LLM
    extracted_intent = extract_intent_from_llm(user_query)
    if "error" in extracted_intent:
        return jsonify(extracted_intent)
    
    operation = extracted_intent.get("operation", "sum")  # Default operation to 'sum'
    filters = extracted_intent.get("filters", {})
    fields = extracted_intent.get("fields", [])
    
    if not fields:
        return jsonify({"error": "No fields found in the extracted intent."})

    result = []
    for field in fields:
        field_result = apply_operation(data, operation, field, filters)
        if field_result is not None:
            result.append({field: field_result})

    return jsonify({"result": result})

@app.route("/chat", methods=["POST"])
def process_chat():
    """Route to handle the user query."""
    return process_query()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8090, debug=True)
