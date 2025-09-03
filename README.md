# Stock Predictor

This is a simple stock price prediction application built with a React frontend and a FastAPI backend. It allows users to get future stock price predictions for a given ticker.

## Getting Started

Follow these steps to set up and run the application locally.

### Prerequisites

Make sure you have the following installed:

*   Python 3.8+ (for the backend)
*   Node.js and npm (for the frontend)
*   uv (for Python package management)

### 1. Backend Setup

To get the backend running, first navigate into the `backend` directory.

```bash
cd backend
source ../venv/bin/activate
uv pip install -r requirements.txt
```

Once the packages are installed, start the FastAPI server with:

```bash
uvicorn main:app --reload
```

### 2. Frontend Setup

In a new terminal, switch to the `frontend` directory.

```bash
cd frontend
npm install
npm start
```

After both the backend and frontend servers are up, open your web browser and head to `http://localhost:3000` (or whatever port your frontend is using).
