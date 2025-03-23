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

# For forecasting model
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns

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
        debt_information_data = list(db["debt_information"].find())
        if not debt_information_data:
            return pd.DataFrame()  # Return empty DataFrame instead of None
        debt_information_df = pd.DataFrame(debt_information_data)
        return debt_information_df
    except Exception as e:
        print(f"Error fetching or merging data: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame in case of an error

def normalize_string(s):
    """Lowercase and remove non-alphanumeric characters."""
    return ''.join(ch for ch in s.lower() if ch.isalnum())

def match_to_column(name, columns, threshold=70):
    """Dynamically match the provided name to the best fitting column in the list."""
    # Check for manual mapping first
    manual_mappings = {
        "year": "Maturity_Year",
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

        # Initialize mask to include all rows
        mask = pd.Series([True] * len(data), index=data.index)

        # Handle filters for Debt_Info_Key (and similar keys)
        if "Debt_Info_Key" in matched_col:
            print(f"Applying Debt_Info_Key filter to column '{matched_col}' with value '{value}'")
            # Always convert value to string, even if it's numeric
            str_value = str(value)
            pattern = f"Debt_Info_Key{str_value}"
            print(f"Pattern to match: '{pattern}'")
            # Cast the column to string to ensure proper matching
            mask = data[matched_col].astype(str).str.contains(pattern, case=False, na=False)
            print(f"Mask for '{matched_col}': {mask.head()}")
            applied_filters = True
            data = data[mask]
            print(f"Applied filter for '{matched_col}' with pattern '{pattern}', rows: {len(data)}")

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
                # General filtering for non-numeric columns with operators
                if data[matched_col].dtype == "object":
                    mask = data[matched_col].str.match(f"^{str(value)}$", case=False, na=False)
                else:
                    mask = data[matched_col] == value
                data = data[mask]
                applied_filters = True
                print(f"Applied filter '{key}' -> '{matched_col}' with value '{value}', rows: {len(data)}")

        else:
            # General filtering for exact values on other fields
            if data[matched_col].dtype == "object":
                mask = data[matched_col].astype(str).str.match(f"^{str(value)}$", case=False, na=False)
            else:
                mask = data[matched_col] == value
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

def forecast_interest_rate(borrowing_amount, loan_duration, loan_type):
    """
    Loads data from MongoDB, preprocesses it, trains an XGBoost model,
    and predicts the interest rate based on input parameters.
    """
    # Fetch and preprocess data
    data = fetch_and_merge_data()
    if data.empty:
        raise Exception("No data available for forecasting.")

    data.fillna(0, inplace=True)

    # Drop non-relevant columns dynamically
    drop_columns = ["_id", "Debt_Info_Key", "Repayment_Amount", "Maturity_Year", "Maturity_Date",
                    "Borrowing_Year", "Borrowing_Date", "Fixed_Float", "Benchmark_Rate",
                    "PSA_Interest", "Percentage_Holdings", "GFDB_Key", "Counterparty"]
    # Drop columns if they exist
    data.drop(columns=[col for col in drop_columns if col in data.columns], inplace=True)

    # Rename "Description" to "Loan_Type" if it exists
    if "Description" in data.columns:
        data.rename(columns={"Description": "Loan_Type"}, inplace=True)

    # Handle categorical encoding dynamically
    encoder = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
    categorical_features = ["Loan_Type"]
    
    if categorical_features[0] in data.columns:
        encoded_features = encoder.fit_transform(data[categorical_features])
        encoded_feature_names = encoder.get_feature_names_out(categorical_features)
        X_encoded = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=data.index)
    else:
        X_encoded = pd.DataFrame(index=data.index)
        encoded_feature_names = []

    # Extract numerical features
    numeric_features = data.drop(columns=categorical_features + ["Interest_Rate"], errors='ignore')

    # Combine numerical and encoded categorical features
    X = pd.concat([numeric_features, X_encoded], axis=1)
    y = data["Interest_Rate"]

    # Train-test split (in production, load a pre-trained model instead)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost model
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    xgb_model.fit(X_train, y_train)

    # Prepare new input for prediction
    new_row = pd.DataFrame(np.zeros((1, X.shape[1])), columns=X.columns)

    # Assign user input values
    new_row["Borrowing_Amount"] = borrowing_amount
    new_row["Loan_Duration"] = loan_duration

    # One-hot encode the loan_type dynamically
    if len(encoded_feature_names) > 0:  # Ensure there are features to encode
        encoded_input = encoder.transform([[loan_type]])
        encoded_df = pd.DataFrame(encoded_input, columns=encoded_feature_names)
        for col in encoded_feature_names:
            if col in new_row.columns:
                new_row[col] = encoded_df[col].iloc[0]

    # Predict interest rate
    predicted_interest_rate = xgb_model.predict(new_row)[0]
    return predicted_interest_rate



def process_query():
    data = fetch_and_merge_data()
    if data is None:
        return jsonify({"reply": "Failed to fetch or merge data."})
    
    user_query = request.json.get("query", "").strip()
    session_data = request.json  # Captures session info

    print(f"Received user query: {user_query}")

    # Check if user is providing the requested values (numbers and text)
    match = re.match(r"(\d+)\s*,\s*(\d+)\s*,\s*([\w\s]+)", user_query)
    if match:
        borrowing_amount = float(match.group(1))  # Extracts the first number
        loan_duration = int(match.group(2))  # Extracts the second number
        loan_type = match.group(3).strip()  # Extracts loan type

        print(f"Extracted values: Amount={borrowing_amount}, Duration={loan_duration}, Type={loan_type}")

        # Call the prediction model
        forecast_result = forecast_interest_rate(borrowing_amount, loan_duration, loan_type)
        return jsonify({"reply": f"Predicted Repayment Plan: {forecast_result}"})

    # If chatbot is waiting for loan details and user response doesn't match expected format
    if "forecast" in user_query.lower() or "predict" in user_query.lower():
        return jsonify({"reply": "Please provide the following: borrowing amount, loan duration (in years), loan type."})


    # Existing handling for database-related queries
    db_related_keywords = ["debt", "year", "repayment", "facility", "interest", "payment", "borrowing", "counterparty", "description", "repay"]
    if any(keyword in user_query.lower() for keyword in db_related_keywords):
        extracted_intent = extract_intent_from_llm(user_query)
        if "error" in extracted_intent:
            return jsonify({"reply": extracted_intent["error"]})
        
        fields = extracted_intent.get("fields", [])
        filters = extracted_intent.get("filters", {})

        if not fields:
            return jsonify({"reply": "No fields found in the extracted intent."})

        columns = data.columns.tolist()
        matched_fields = [match_to_column(field, columns) or vectorized_match_to_column(field, columns) for field in fields]
        matched_fields = [f for f in matched_fields if f]

        data = apply_filters(data, filters)
        if data.empty:
            return jsonify({"reply": "No data found after filtering."})

        if len(matched_fields) == 1 and pd.api.types.is_numeric_dtype(data[matched_fields[0]]):
            total_value = data[matched_fields[0]].sum()
            raw_result = f"{matched_fields[0]}: {total_value}"
        elif len(data) == 1:
            raw_result = data[matched_fields].iloc[0].to_dict()
        else:
            raw_result = data[matched_fields].to_dict(orient="records")
        
        friendly_reply = generate_llm_response(user_query, raw_result)
        return jsonify({"reply": friendly_reply})
    
    else:
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
