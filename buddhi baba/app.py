import openai
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Set your OpenAI API key
openai.api_key = 'your-openai-api-key'

# Sample data for predictive analysis (replace this with your real dataset)
data = pd.DataFrame({
    'year': [2020, 2021, 2022, 2023, 2024],
    'population': [1.2, 1.5, 1.8, 2.1, 2.4],  # Example: population growth
    'housing_units': [1000, 1200, 1300, 1500, 1600]  # Example: housing development
})

# Machine learning model for prediction (simple linear regression for demo)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(data[['year']], data['population'])

# Function to generate a response from OpenAI GPT-3
def get_chatbot_response(user_input):
    response = openai.Completion.create(
        engine="text-davinci-003",  # You can choose other models
        prompt=user_input,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Function to generate a predictive chart
def generate_predictive_chart():
    # Predict future population (just for illustration)
    future_years = pd.DataFrame({'year': [2025, 2026, 2027, 2028, 2029]})
    future_population = model.predict(future_years)
    future_years['predicted_population'] = future_population

    # Plotting the chart
    plt.figure(figsize=(10, 6))
    plt.plot(future_years['year'], future_years['predicted_population'], label='Predicted Population Growth', marker='o')
    plt.title('Predicted Population Growth Over Time')
    plt.xlabel('Year')
    plt.ylabel('Population')
    plt.grid(True)
    plt.legend()
    plt.savefig('population_prediction.png')  # Save the plot as an image
    return 'population_prediction.png'

# Chatbot endpoint
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json['message']
    
    # If the user asks for a predictive chart
    if "predict" in user_input.lower():
        chart_file = generate_predictive_chart()
        return jsonify({'response': 'Here is the predicted population chart!', 'chart': chart_file})
    
    # Otherwise, get a response from OpenAI GPT-3
    chatbot_reply = get_chatbot_response(user_input)
    return jsonify({'response': chatbot_reply})

if __name__ == "__main__":
    app.run(debug=True)
