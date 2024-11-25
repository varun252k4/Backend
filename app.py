from fastapi import FastAPI, HTTPException
import joblib
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import pickle
from joblib import load
from groq import Groq
from utils.utils import get_crop_recommendation, get_fertilizer_recommendation

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins; replace with specific domains for security
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

api_key = "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"
if not api_key:
    raise ValueError("GROQ_API_KEY is not set in environment variables.")

# Initialize Groq client
client = Groq(api_key=api_key)

# Load models and encoders
crop_model = pickle.load(open('Models/RandomForest.pkl', 'rb'))
fertilizer_model = pickle.load(open('Models/fertilizer.pkl', 'rb'))
soil_label_encoder = joblib.load('Models/soil_label_encoder.joblib')
crop_label_encoder = joblib.load('Models/crop_label_encoder.joblib')
fertilizer_label_encoder = joblib.load('Models/fertilizer_encoder.joblib')

# Input and Response Models
class CropInput(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

class FertilizerInput(BaseModel):
    temperature: float
    humidity: float
    moisture: float
    soil_type: str
    crop_type: str
    nitrogen: float
    potassium: float
    phosphorous: float  # Corrected spelling

class CropResponse(BaseModel):
    recommended_crop: str

class FertilizerResponse(BaseModel):
    recommended_fertilizer: str

class UserContent(BaseModel):
    user_content: str

# Endpoints
@app.post("/generate")
async def response(content: UserContent):
    system_prompt = {
        "role": "system",
        "content": """
        You are a friendly and knowledgeable farm assistant chatbot specializing in Indian agriculture. You respond conversationally, as if you were a helpful person having a chat. Your tone should be warm, approachable, and slightly casual, using phrases like "yeah, sure," "of course," or "absolutely" to make the conversation feel natural. Avoid overly technical jargon unless necessary, and always aim to sound like you're genuinely helping someone.

        Example 1:
        Input: "What are the best practices for growing wheat in India?"
        Output:
        "Yeah, sure! Growing wheat in India can be really rewarding if you follow some best practices. First, you’ll want to choose a good variety for your region—something like HD-2967 or PBW-343 works well for high yields. Next, make sure your soil is prepared properly—plowing and adding organic manure or compost will give the plants a great start. Timing is also key, so aim to sow the seeds between late October and mid-November. Oh, and don’t forget to irrigate at crucial stages like tillering, flowering, and grain filling! Keep an eye out for pests like aphids and manage them early. You’ve got this!"

        Example 2:
        Input: "Can you explain how drip irrigation works for vegetables?"
        Output:
        "Of course! Drip irrigation is such a smart choice, especially for vegetables. Basically, it delivers water directly to the roots through a network of pipes and emitters, which means less water wasted and healthier plants. It’s perfect for crops like tomatoes, chillies, or cucumbers. If you’re setting it up, make sure the pipes are laid close to the plants, and the emitters are spaced evenly so all your crops get the same amount of water. Oh, and if you combine it with fertigation, you can feed your plants nutrients along with the water. It’s a win-win!"

        Now, respond conversationally to the following farming-related query:
        """
    }

    user_input = content.user_content
    full_prompt = system_prompt["content"] + "\n" + f"Input: \"{user_input}\"\nOutput:"

    # Call the Groq API for response
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[system_prompt, {"role": "user", "content": full_prompt}],
        max_tokens=150,
        temperature=0.7
    )

    response = response.choices[0].message.content
    return {"response": response}

@app.post("/crop_recommendation/", response_model=CropResponse)
async def crop_recommendation(data: CropInput):
    try:
        recommended_crop = get_crop_recommendation(data, crop_model)
        return CropResponse(recommended_crop=recommended_crop)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during crop recommendation: {str(e)}")

@app.post("/fertilizer_recommendation/", response_model=FertilizerResponse)
async def fertilizer_recommendation(data: FertilizerInput):
    try:
        recommended_fertilizer = get_fertilizer_recommendation(
            data, fertilizer_model, soil_label_encoder, crop_label_encoder, fertilizer_label_encoder
        )
        return FertilizerResponse(recommended_fertilizer=recommended_fertilizer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during fertilizer recommendation: {str(e)}")