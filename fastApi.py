from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse
import pytesseract
from PIL import Image
from langchain_community.llms import Ollama
from crewai import Agent, Task, Crew, Process
import uvicorn
import json

app = FastAPI()

# Initialize HTTPBasic security
security = HTTPBasic()

# Function to extract text from image using pytesseract
def extract_text_from_image(image_file: UploadFile):
    image = Image.open(image_file.file)
    text = pytesseract.image_to_string(image, config='--oem 3 --psm 1')
    formatted_text = text.strip().replace("\n", " ")
    return formatted_text

# Function to process invoice text and return JSON
def process_invoice(invoice_text):
    model = Ollama(model="llama3.1")

    Extractor = Agent(
        role="Invoice Extractor",
        goal="""Extract the following details from the invoice text:
            - Invoice Client Information:
                - Name of the invoice receiver
                - ICE (Identification Code)
                - Email
                - Invoice Address
                - Shipping Address
            - Invoice Date (invoiceAt)
            - Invoice Due Date (invoiceDueAt)
            - Total Amount (amount)
            - Invoice Items:
                - Unit Price
                - Quantity
                - TVA (Tax)
                - Discount
                - Net Amount
                - Amount
                - Name of the Item
        """,
        backstory="You are an invoice extraction agent. You are required to extract the invoice details from the given invoice formatted text.",
        verbose=True,
        allow_delegation=False,
        llm=model
    )

    Responder = Agent(
        role="Invoice Responder",
        goal="Respond with a JSON containing the invoice details extracted, respond with only JSON, don't add any text or note or request or explanation to the response.",
        backstory="You are an API endpoint whose only job is to respond with a JSON containing the invoice details extracted that will be provided to you by the 'Invoice Extractor' agent.",
        verbose=True,
        allow_delegation=False,
        llm=model
    )

    extract_invoice = Task(
        description=f"""Extract the following details from the given invoice text: 
            - Invoice Client Information:
                - Name
                - ICE (Identification Code)
                - Email
                - Invoice Address
                - Shipping Address
            - Invoice Date "YYYY-MM-DD" (invoiceAt)
            - Invoice Due Date (invoiceDueAt)
            - Total Amount (amount)
            - Invoice Items:
                - Unit Price
                - Quantity
                - TVA (Tax)
                - Discount
                - Net Amount
                - Amount
                - Name of the Item
            Given invoice text: '{invoice_text}'.
        """,
        agent=Extractor,
        expected_output="Invoice details extracted from the given invoice text."
    )

    respond_invoice = Task(
        description=f"Respond to the '{invoice_text}' with a JSON containing the invoice details extracted and provided by the 'Extractor' agent. The response should be only the JSON and no other text.",
        agent=Responder,
        expected_output="""JSON containing the invoice details extracted as below but without the comments or any other text: 
        {
            "invoiceClient": { // info about the receiver of the invoice
                "name": "string", // name of the receiver e.g. "CBIS.COM" or "INFLEXIT"
                "ICE": "number", 
                "email": "string", 
                "invoiceAddress": "string",
                "shippingAddress": "string"
            },
            "invoiceAt": "YYYY-MM-DD", // invoice date
            "invoiceDueAt": "YYYY-MM-DD", // invoice due date
            "amount": "number", // total price of the entire invoice including taxes
            "invoiceItems": [
                {
                    "unitPrice": "number",
                    "quantity": "number",
                    "tva": "number",
                    "discount": "number",
                    "netAmount": "number",
                    "amount": "number",
                    "name": "string"
                }
            ]
        }
        in case you didn't find a specific value, it should be null for strings and 0 for number values"""
    )

    crew = Crew(
        agents=[Extractor, Responder],
        tasks=[extract_invoice, respond_invoice],
        verbose=0,
        process=Process.sequential,  # Tasks are executed sequentially
        tempeture=0.2,
        seed=42
    )

    output = crew.kickoff()
    output_json = json.loads(str(output))  # Parse the output to ensure it's valid JSON
    return output_json

# Authenticate user
def authenticate_user(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = "user"
    correct_password = "pass"

    if credentials.username != correct_username or credentials.password != correct_password:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password"
        )

# API endpoint to upload an image and get the extracted information as JSON
@app.post("/extract_invoice")
async def extract_invoice(
    image: UploadFile = File(...),
    credentials: HTTPBasicCredentials = Depends(authenticate_user)
):
    try:
        # Step 1: Extract text from image
        invoice_text = extract_text_from_image(image)
        
        # Step 2: Process the extracted text and get the JSON response
        json_response = process_invoice(invoice_text)

        
        return JSONResponse(content=json_response)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Run the FastAPI application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# test: curl -u user:pass -X POST "http://localhost:8000/extract_invoice" -F "image=@./image.jpg"