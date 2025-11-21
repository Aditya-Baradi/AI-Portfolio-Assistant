


import os
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


import pymupdf4llm

md_text = pymupdf4llm.to_markdown("8b2aa722-e192-3500-a09b-3826bc7a121d.pdf")


import pathlib
pathlib.Path("output.md").write_bytes(md_text.encode())

def json_transform():
    chat = ChatOpenAI(model="gpt-4", temperature=0)

    json_prompt = f"""
    You are a data transformation assistant helping convert a user's portfolio into a FinRL-compatible JSON file.
    
    Given the following raw input:
    - A list of stock holdings with:
      - ticker symbol (e.g., "AAPL")
      - number of shares owned (e.g., 5.2)
      - current price per share (e.g., 210.45)
      - total value owned (e.g., 1094.34)
    
    Transform this into an enriched JSON format compatible with FinRL backtesting. Each object in the array should represent a stock on a given day and include the following fields:
    
    - date (use today’s date)
    - tic (the ticker symbol)
    - close (the price of one share)
    - volume (How much quantity of the stock they own)
    
    With the following portfolio {portfolio}
    """


    messages = [
        HumanMessage(content=json_prompt)
    ]

    response = chat.invoke(messages)

    print("Bot:", response.content)