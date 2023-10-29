from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware


HF_runway_T_I = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
HF_headers = {"Authorization": "Bearer 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"}


def query(payload):
    response = requests.post(HF_runway_T_I, headers=HF_headers, json=payload)
    return response.content


app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Enable CORS (Cross-Origin Resource Sharing) to allow requests from the HTML file
app.add_middleware(
    CORSMiddleware,
    # Allowing requests from any origin (you might want to restrict this in production)
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Route to handle the POST request from the form


@app.post("/api/submit")
async def submit_text(text: str):
    # Process the user's input (text) here as needed
    # For now, just return a message with the received text
    return {"message": f"Received text: {text}"}


@app.get("/")
async def name(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})
