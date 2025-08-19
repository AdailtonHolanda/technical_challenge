from fastapi import FastAPI
from .routes import prediction

app = FastAPI(
    title="Real Estate Price Prediction API",
    description="RESTful endpoint to predict house prices using a machine learning model."
)

# Include the router with prediction endpoints
app.include_router(prediction.router)


@app.get("/")
def read_root():
    """
    Root endpoint of the API.
    """
    return {"message": "Welcome to the Real Estate Price Prediction API."}
