from flask import Flask, request, jsonify
from flask_cors import CORS
import google.genai
import os

app = Flask(__name__)
CORS(app)  # Enable requests from the React frontend

# Your API key
API_KEY = "AIzaSyCjyVoiwVAzcGZ8-dtSfdEJnNADqSZnwMY"

# Initialize the Gemini client
client = google.genai.Client(api_key=API_KEY)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Generate content using Gemini API
        response = client.models.generate_content(
            model="gemini-1.5-flash",  # Replace with the correct model version
            contents=user_message
        )

        # Get the assistant's reply
        assistant_reply = response.text

        return jsonify({"reply": assistant_reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8090, debug=True)
