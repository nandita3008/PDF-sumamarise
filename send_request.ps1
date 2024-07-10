# Define the URL and headers
$url = "http://127.0.0.1:5000/summarize"
$headers = @{
    "Content-Type" = "application/json"
}

# Define the body for Hugging Face model
$body_hf = @{
    "question" = "how is the evaluation done for lab courses?"
    "model_type" = "hf"
    "top_n" = 5
} | ConvertTo-Json

# Define the body for OpenAI model
$body_openai = @{
    "question" = "how is the evaluation done for lab courses?"
    "model_type" = "openai"
    "top_n" = 5
} | ConvertTo-Json

# Send the POST request for Hugging Face model
$response_hf = Invoke-WebRequest -Uri $url -Method POST -Headers $headers -Body $body_hf

# Display the response for Hugging Face model
Write-Output "Hugging Face Model Summary:"
$response_hf.Content

# Send the POST request for OpenAI model
$response_openai = Invoke-WebRequest -Uri $url -Method POST -Headers $headers -Body $body_openai

# Display the response for OpenAI model
Write-Output "OpenAI Model Answer:"
$response_openai.Content
