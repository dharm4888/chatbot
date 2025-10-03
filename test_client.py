import requests

url = "http://127.0.0.1:8000/chat"

# Example 1: normal chat
payload = {
    "message": "Who is the Prime Minister of India?"
}
resp = requests.post(url, json=payload)
print("Response:", resp.json())

# Example 2: chat with SQL
payload_sql = {
    "message": "Show me available t-shirts",
    "use_sql_tool": True,
    "sql_query": "SELECT * FROM t_shirts LIMIT 5"
}
resp_sql = requests.post(url, json=payload_sql)
print("SQL Response:", resp_sql.json())
