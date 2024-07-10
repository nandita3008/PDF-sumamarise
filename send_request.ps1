# Define the URL and headers
$url = "http://127.0.0.1:5000/summarize"
$headers = @{
    "Content-Type" = "application/json"
}

# Define the body
$body = @{
    "question" = "how is the evaluation done for lab courses?"
    "top_n" = 5
} | ConvertTo-Json

# Send the POST request
$response = Invoke-WebRequest -Uri $url -Method POST -Headers $headers -Body $body

# Display the response
$response.Content
