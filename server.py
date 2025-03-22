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
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

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
        # Normalize cp_key to Counterparty column
        if key.lower() in ["cp", "counterparty", "cp_key"]:  # Allow variations like cp, CP, counterparty, cp_key
            key = "counterparty"  # Normalize the key to match the actual column name
        
        matched_col = match_to_column(key, columns) or vectorized_match_to_column(key, columns)
        
        # Skip if no match found for the column
        if matched_col is None:
            print(f"Filter key '{key}' did not match any column.")
            continue  # Skip this filter and move to the next one

        # Initialize mask to None at the start of each loop iteration
        mask = pd.Series([True] * len(data), index=data.index)  # Default mask to include all rows

        # Check if the filter contains the keyword "counterparty" and extract the number
        if matched_col == "Counterparty":
            # Normalize counterparty values to a standard format (e.g., cp_42, counterparty42)
            match = re.search(r'(\d+)', str(value))
            if match:
                counterparty_number = match.group(1)
                print(f"Counterparty number extracted: {counterparty_number}")
                
                # Normalize both 'cp42', 'CP42', 'counterparty42' to 'cp_42'
                normalized_value = f"cp_{counterparty_number}"  # Consistent normalization format
                
                # Apply the filter with normalized counterparty number
                mask = data[matched_col].str.contains(normalized_value, case=False, na=False)
                applied_filters = True
                data = data[mask]
                print(f"Applied counterparty filter for '{matched_col}' with value '{normalized_value}', rows: {len(data)}")
            else:
                print(f"Could not extract counterparty number from '{value}'")
                continue  # Skip this filter if no counterparty number is found

        elif "Debt_Info_Key" in matched_col:
            value = f"Debt_Info_Key{value}"  # Handle Debt_Info_Key filter

        elif isinstance(value, dict) and "operator" in value:
            # Handle operator-based filters (max, min) for numeric columns
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
        return jsonify({"reply": "Failed to fetch or merge data."})
    
    user_query = request.json.get("query", "")
    print(f"Received user query: {user_query}")

    # Check if the query looks like a database-related query
    db_related_keywords = ["debt", "year", "repayment", "facility", "interest", "payment", "borrowing", "counterparty", "description"]
    if any(keyword in user_query.lower() for keyword in db_related_keywords):
        # Step 1: Extract intent from the LLM (for database-related queries)
        extracted_intent = extract_intent_from_llm(user_query)
        if "error" in extracted_intent:
            return jsonify({"reply": extracted_intent["error"]})
        
        fields = extracted_intent.get("fields", [])
        filters = extracted_intent.get("filters", {})

        if not fields:
            return jsonify({"reply": "No fields found in the extracted intent."})

        # Step 2: Validate fields against database columns
        columns = data.columns.tolist()
        matched_fields = [match_to_column(field, columns) or vectorized_match_to_column(field, columns) for field in fields]
        matched_fields = [f for f in matched_fields if f]  # Remove None values

        # Step 3: Apply filters
        data = apply_filters(data, filters)
        if data.empty:
            return jsonify({"reply": "No data found after filtering."})

        # Step 4: Retrieve and format result(s)
        if len(matched_fields) == 1 and pd.api.types.is_numeric_dtype(data[matched_fields[0]]):
            total_value = data[matched_fields[0]].sum()
            raw_result = f"{matched_fields[0]}: {total_value}"
        elif len(data) == 1:
            raw_result = data[matched_fields].iloc[0].to_dict()
        else:
            raw_result = data[matched_fields].to_dict(orient="records")
        
        # Generate a friendly reply using the LLM
        friendly_reply = generate_llm_response(user_query, raw_result)
        return jsonify({"reply": friendly_reply})
    
    else:
        # Handle non-database queries or courteous phrases
        courteous_phrases = ["hi", "bye", "thank you", "thanks", "hello", "goodbye", "heyo"]
        if any(phrase in user_query.lower() for phrase in courteous_phrases):
            if "thank" in user_query.lower():
                return jsonify({"reply": "You're welcome!"})
            elif "bye" in user_query.lower():
                return jsonify({"reply": "Goodbye!"})
            else:
                return jsonify({"reply": "Hello! How can I assist you?"})
        
        return jsonify({"reply": "I'm here to help with specific financial queries. Please ask a related question."})


def generate_llm_response(query, db_result):
    """Generate a human-friendly response based on the query and DB result."""
    prompt = f"""You are an AI financial assistant. 
        A user asked: "{query}" 
        and the database returned the following result: {db_result}.
        Generate a clear, concise, and friendly response summarizing the result."""
        
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = retry_policy(model.generate_content)(
            [prompt],
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 300
            }
        )
        candidates = response.candidates
        if candidates:
            content = candidates[0].content.parts[0].text.strip()
            # Remove Markdown code blocks if present
            if content.startswith("```") and content.endswith("```"):
                content = content.strip("```")
            return content
        else:
            return str(db_result)
    except Exception as e:
        print("Error generating LLM response:", e)
        return str(db_result)


@app.route("/chat", methods=["POST"])
def process_chat():
    """Route to handle the user query."""
    return process_query()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8090, debug=True)
