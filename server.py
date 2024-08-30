# Step 1: Install FastAPI and Uvicorn
# pip install fastapi uvicorn

# Step 2: Create the FastAPI app
from fastapi import FastAPI
from pydantic import BaseModel
from serp_api import search
from case_law_search import search_cases

app = FastAPI()

# Step 3: Define the chat endpoint
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

class CaseSearchResponse(BaseModel):
    answer: str
    links: list

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # For simplicity, return a static response
    response= search(request.message)
    return ChatResponse(response=response)

@app.post("/casesearch_chat", response_model=CaseSearchResponse)
async def caselaw(request: ChatRequest):
    # For simplicity, return a static response
    response= search_cases(request.message)
    return CaseSearchResponse(answer=response["answer"], links=response["links"])

# Step 4: Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)