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
from google.api_core.retry import Retry, if_exception_type
from google.api_core.exceptions import ResourceExhausted, DeadlineExceeded

# Initialize Flask app
app = Flask(__name__)

# Configure Gemini API key (Replace with your API key)
genai.configure(api_key="AIzaSyCjyVoiwVAzcGZ8-dtSfdEJnNADqSZnwMY")

# MongoDB Connection
client = MongoClient("mongodb://localhost:27017")
db = client.get_database("capstone")

# Sentence transformer model for vectorization
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # A pre-trained model for sentence embeddings

# Retry policy
retry_policy = Retry(
    initial=1.0,
    maximum=60.0,
    multiplier=2.0,
    deadline=300.0,
    predicate=if_exception_type(ResourceExhausted, DeadlineExceeded),
    maximum_attempts=5
)

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
      - "fields": an array of field names the user wants (e.g., ["description"])
      - "filters": an object where keys are field names and values are the filter criteria (e.g., {"Debt_Info_Key": 7})
    If a number is used after a field (e.g., "debt info key 7"), treat it as a filter.
    You should also detect if the query asks for max, min, highest, lowest, etc., and include this in the filter criteria.

    """
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Apply retry policy to the API call directly
        response = retry_policy(model.generate_content)(
            [system_prompt, f"User query: {query}"],
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 300
            }
        )

        # Print raw response for debugging
        print("Raw Gemini Response:", response)

        candidates = response.candidates
        if not candidates:
            return {"error": "No candidates found in the response."}

        content = candidates[0].content.parts[0].text.strip()

        # Remove Markdown code block if present
        if content.startswith("```json") and content.endswith("```"):
            content = content[7:-3].strip()

        try:
            structured_intent = json.loads(content)
            
            # Ensure keys exist
            structured_intent.setdefault("fields", [])
            structured_intent.setdefault("filters", {})
            
            
            # Look for max/min/lowest/highest keywords and set the operator accordingly
            for field, value in structured_intent["filters"].items():
                if isinstance(value, str):
                    if "highest" in value.lower() or "max" in value.lower():
                        structured_intent["filters"][field] = {"operator": "max"}
                    elif "lowest" in value.lower() or "min" in value.lower():
                        structured_intent["filters"][field] = {"operator": "min"}

            return structured_intent
        
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
            # Handle matching with prefix for Debt_Info_Key
            if "Debt_Info_Key" in matched_col:
                value = f"Debt_Info_Key{value}"
                
            elif isinstance(value, dict) and "operator" in value:
                if pd.api.types.is_numeric_dtype(data[matched_col]):
                    if value["operator"] == "max" or "highest" in str(value["operator"]).lower():
                        max_value = data[matched_col].max()
                        data = data[data[matched_col] == max_value]
                        applied_filters = True
                        print(f"Applied 'max' or 'highest' filter on '{matched_col}', rows: {len(data)}")
                    elif value["operator"] == "min" or "lowest" in str(value["operator"]).lower():
                        min_value = data[matched_col].min()
                        data = data[data[matched_col] == min_value]
                        applied_filters = True
                        print(f"Applied 'min' or 'lowest' filter on '{matched_col}', rows: {len(data)}")
                    else:
                        print(f"Unknown operator '{value['operator']}' for field '{matched_col}'.")
                else:
                    print(f"Field '{matched_col}' is not numeric. Skipping operator-based filter.")
                
            else:
                # General filtering for exact values
                if data[matched_col].dtype == "object":
                    mask = data[matched_col].str.match(f"^{str(value)}$", case=False, na=False)
                else:
                    mask = data[matched_col] == value

                data = data[mask]
                applied_filters = True
                print(f"Applied filter '{key}' -> '{matched_col}' with value '{value}', rows: {len(data)}")
        else:
            print(f"Filter key '{key}' did not match any column.")
            
            data = data[mask]
            applied_filters = True
            print(f"Applied filter '{key}' -> '{matched_col}' with value '{value}', rows: {len(data)}")
     
    
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
        # Return unique values for non-numeric columns
        return data[matched_col].dropna().unique().tolist()

    return None


def process_query():
    data = fetch_and_merge_data()
    if data is None:
        return jsonify({"error": "Failed to fetch or merge data."})
    
    user_query = request.json.get("query", "")
    print(f"Received user query: {user_query}")

    # Check if the query looks like a database-related query
    db_related_keywords = ["debt", "year", "repayment", "facility", "interest", "payment"]
    if any(keyword in user_query.lower() for keyword in db_related_keywords):
        # Step 1: Extract intent from the LLM (for database-related queries)
        extracted_intent = extract_intent_from_llm(user_query)
        if "error" in extracted_intent:
            return jsonify(extracted_intent)
        
        fields = extracted_intent.get("fields", [])
        filters = extracted_intent.get("filters", {})

        if not fields:
            return jsonify({"error": "No fields found in the extracted intent."})

        # Step 2: Validate fields against database columns
        columns = data.columns.tolist()
        matched_fields = [match_to_column(field, columns) or vectorized_match_to_column(field, columns) for field in fields]
        matched_fields = [f for f in matched_fields if f]  # Remove None values

        # Step 3: Apply filters
        data = apply_filters(data, filters)

        if data.empty:
            return jsonify({"error": "No data found after filtering."})

        # Step 4: Retrieve the selected fields and handle only one record if needed
        if len(matched_fields) == 1 and pd.api.types.is_numeric_dtype(data[matched_fields[0]]):
            total_value = data[matched_fields[0]].sum()
            return jsonify({matched_fields[0]: total_value})
        
        # If we are expecting one specific entry (e.g., one year or one debt info key), return the first matching row
        if len(data) == 1:
            return jsonify({"result": data[matched_fields].iloc[0].to_dict()})

        return jsonify({"result": data[matched_fields].to_dict(orient="records")})

    else:
        # Step 2: Handle general queries using LLM (fallback for non-database queries)
        system_prompt = """
        You are an AI assistant capable of answering a variety of general knowledge questions. Please provide concise, informative answers to the following query.
        """

        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(
                [system_prompt, f"User query: {user_query}"],
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 300
                }
            )

            # Print raw response for debugging
            print("Raw Gemini Response:", response)

            candidates = response.candidates
            if not candidates:
                return jsonify({"error": "No response from the LLM."})

            content = candidates[0].content.parts[0].text.strip()

            return jsonify({"result": content})
        
        except Exception as e:
            return jsonify({"error": f"Failed to process general query: {str(e)}"})



@app.route("/chat", methods=["POST"])
def process_chat():
    """Route to handle the user query."""
    return process_query()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8090, debug=True)
