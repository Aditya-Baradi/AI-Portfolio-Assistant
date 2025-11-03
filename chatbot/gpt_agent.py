import os
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from Main import portfolio


def get_gpt_recommendations(user_profile: dict) -> dict:
    # Build prompt from user profile
    # Return parsed recommendation like sectors to invest, risk score etc.

    chat = ChatOpenAI(model="gpt-4", temperature=0)
    prompt= f"""
As a FinRL user, I want to optimize my stock portfolio by identifying uninvested sectors that align with my financial goals and risk tolerance.

Given the following inputs:

* **Current Portfolio:** A dictionary of stock tickers and their respective quantities (e.g., `{'AAPL': 10, 'MSFT': 5, 'GOOG': 8}`).
* **Time to Invest:** `{timeLength}` (e.g., '5 years', '10 years', '20+ years'). This indicates my investment horizon.
* **Current Investment:** `{amountInvested}` (e.g., '$50,000'). This is the total value of my current portfolio.
* **Target Return:** `{projectedAmount}` (e.g., '$100,000', '15% annual return'). This is my desired financial outcome.
* **Extra Money to Invest:** `{extraMoney}` (e.g., '$5,000 one-time', '$500 monthly'). This is additional capital I can allocate.
* **Risk Tolerance (1-10):** `{riskLevel}` (1 being very low, 10 being very high). This reflects my willingness to take on risk.
* **Wants to Diversify:** `{diversify}` (True/False). This indicates if diversification is a primary goal.

I need FinRL to perform the following:

1.  **Retrieve Comprehensive Sector Data:** Access a reliable data source (e.g., Yahoo Finance, Fama-French industry classifications) to identify all available stock sectors in the market or a specified universe.
2.  **Analyze Current Portfolio by Sector:** For each stock in my `Current Portfolio`, determine its sector and calculate my current allocation across sectors.
3.  **Identify Uninvested/Underweighted Sectors:** Based on the comprehensive sector list, identify sectors where I have no current holdings or significantly lower-than-average exposure (if `Wants to Diversify` is True).
4.  **Factor in Financial Goals and Risk:**
    * **Time to Invest:** Longer time horizons generally allow for higher risk and greater exposure to growth-oriented sectors. Shorter horizons may suggest more stable, mature sectors.
    * **Current Investment & Target Return:** These inform the required growth rate and potential allocation strategies.
    * **Extra Money to Invest:** This capital should be strategically allocated to fill sector gaps or enhance existing positions.
    * **Risk Tolerance:**
        * Higher `riskLevel` (e.g., 7-10) may suggest exploring higher-growth, potentially more volatile sectors (e.g., technology, biotechnology).
        * Moderate `riskLevel` (e.g., 4-6) might lean towards a balanced approach, considering sectors with stable growth (e.g., consumer staples, healthcare).
        * Lower `riskLevel` (e.g., 1-3) would prioritize defensive sectors with lower volatility (e.g., utilities, real estate).
    * **Wants to Diversify:** If True, prioritize identifying and suggesting investments in unrepresented or underrepresented sectors to reduce concentrated risk.
5.  **Generate Sector Recommendations:** Based on the above analysis, provide a list of recommended sectors where I should consider investing, along with a brief rationale for each recommendation considering my inputs.
6.  **Suggest Potential Stock Examples (Optional but highly beneficial):** For each recommended sector, provide a few high-level examples of well-known stocks that could be considered, emphasizing that further individual stock research is necessary.

**Expected Output:**

A structured output outlining:
* Sectors currently in your portfolio and their approximate weight.
* Sectors where you have no investment or are significantly underweighted.
* Recommended sectors for investment, taking into account your `timeLength`, `amountInvested`, `projectedAmount`, `extraMoney`, `riskLevel`, and `diversify` preference.
* Brief justifications for each recommended sector.
"""




    messages = [
        HumanMessage(content=prompt)
    ]

    response = chat.invoke(messages)

    print ("Bot:", response.content)
