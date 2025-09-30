```markdown
# AI Overlapping Workloads

This notebook explores the impact of AI on various workloads, specifically analyzing how AI technologies overlap with and potentially replace traditional tasks. It uses several tools and libraries to achieve this:

*   **Web Scraping:** To gather information about workloads and their AI counterparts.
*   **LLM Integration:** To analyze the gathered information and provide insights.
*   **Semantic Similarity:** To compare and understand the relationships between concepts.
*   **Data Visualization:** To present the analysis results in an understandable format.

The ultimate goal is to quantify the extent to which AI overlaps with a given workload.

## Setup and Imports

First, we import all necessary libraries and set up essential functions.

```python
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from datetime import datetime, timedelta
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from sentence_transformers import SentenceTransformer, util
import torch
from google import genai
import json
import streamlit as st
```

### Date and Time Utility

A simple function to get the current date and time in a specific format, adjusted for Indian Standard Time (UTC+5:30).

```python
def date_time():
    """Returns the current UTC time adjusted by +5:30 hours, formatted as YYYY-MM-DD_HH-MM-SS."""
    return (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime('%Y-%m-%d_%H-%M-%S')

print(f"Current adjusted time: {date_time()}")
```

### Gemini LLM Integration

This section sets up the integration with Google's Gemini LLM for text generation and analysis.

```python
# Replace "YOUR_API_KEY" with your actual Google Gemini API key
# It's recommended to load this from environment variables or a secure configuration
try:
    # Attempt to load API key from environment variable
    import os
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        # Fallback to placeholder if not found (replace with your actual key if needed for testing)
        api_key = "AIzaSyA5ZX5mYrz1XxNyPEGhzrvt8lB0-JEAegc" # Example placeholder
        print("Warning: GOOGLE_API_KEY environment variable not found. Using placeholder key. Replace with your actual key.")

    client = genai.Client(api_key=api_key)

    def GeminiLLM(prompt):
        """
        Generates text content using the Gemini LLM.

        Args:
            prompt (str): The input prompt for the LLM.

        Returns:
            str: The generated text response from the LLM.
        """
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
        )
        return response.text

    # Test the Gemini LLM
    print("Testing Gemini LLM...")
    test_response = GeminiLLM('Hello, this is a test prompt.')
    print(f"Gemini LLM response: {test_response}")

except Exception as e:
    print(f"Error initializing Gemini LLM: {e}")
    print("Please ensure you have a valid GOOGLE_API_KEY set as an environment variable or directly in the script.")
    # Define a placeholder function if initialization fails
    def GeminiLLM(prompt):
        return "Gemini LLM initialization failed. Please check API key and setup."
```

## Web Scraping with Bing

This function scrapes search results from Bing to gather descriptive text for a given query.

```python
def bing_scrape(query, count=3):
    """
    Scrape Bing search results and return a concatenated text summary of titles and snippets.

    Args:
        query (str): The search query.
        count (int): The number of search results to retrieve (default is 3).

    Returns:
        str: A string containing the concatenated titles and snippets of the search results,
             or an empty string if an error occurs or no results are found.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/139.0.0.0 Safari/537.36"
    }

    params = {"q": query, "count": count}
    url = "https://www.bing.com/search"

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        print(f"Error during Bing scrape for query '{query}': {e}")
        return ""

    soup = BeautifulSoup(response.text, "html.parser")
    parag = ''
    # Find list items with class "b_algo" which typically contain search results
    for item in soup.find_all("li", {"class": "b_algo"}, limit=count):
        title_tag = item.find("h2")
        snippet_tag = item.find("p")

        if title_tag:
            parag += title_tag.get_text() + '\n'
        if snippet_tag:
            parag += snippet_tag.get_text() + '\n'

    return parag

# --- Test Bing Scrape ---
print("Testing Bing Scrape for 'Python AI':")
bing_result_test = bing_scrape("Python AI", count=2)
print(bing_result_test)
```

## Semantic Similarity with Sentence Transformers

This section utilizes the `sentence-transformers` library to compute semantic similarity between texts. It stores embeddings to optimize performance by avoiding redundant computations.

```python
# Store embeddings with their original text
stored_texts = []
stored_embeddings = []

# Load the MiniLM model for sentence embeddings
# 'all-MiniLM-L6-v2' is a good balance of performance and speed
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("SentenceTransformer model loaded successfully.")
except Exception as e:
    print(f"Error loading SentenceTransformer model: {e}")
    print("Please ensure 'sentence-transformers' is installed and the model is accessible.")
    # Define a placeholder if loading fails
    model = None

def semantic_similarity_score(input_text, text_list):
    """
    Compares input_text with a list of candidate texts and returns semantic similarity scores.
    It caches computed embeddings to avoid recomputing for previously seen texts.

    Args:
        input_text (str): The text to compare against the list.
        text_list (list): A list of candidate texts.

    Returns:
        list: A list of dictionaries, where each dictionary contains 'text' and its 'score'
              (cosine similarity) with the input_text. Returns an empty list if the model
              is not available or no embeddings can be computed.
    """
    global stored_texts, stored_embeddings

    if model is None:
        print("Semantic similarity model is not available.")
        return []

    # Encode candidate texts only once and store them
    for text in text_list:
        if text not in stored_texts:
            try:
                emb = model.encode(text, convert_to_tensor=True)
                stored_texts.append(text)
                stored_embeddings.append(emb)
            except Exception as e:
                print(f"Error encoding text '{text}': {e}")
                continue # Skip this text if encoding fails

    # Encode the input text
    try:
        input_emb = model.encode(input_text, convert_to_tensor=True)
    except Exception as e:
        print(f"Error encoding input text '{input_text}': {e}")
        return []

    # Stack stored embeddings into a single tensor for efficient computation
    if stored_embeddings:
        # Ensure all embeddings are on the same device if they were created on GPU
        try:
            stacked_embeddings = torch.stack(stored_embeddings).to(input_emb.device)
        except Exception as e:
            print(f"Error stacking embeddings: {e}")
            return []
    else:
        return [] # Return empty list if no embeddings are stored

    # Compute cosine similarity against stacked embeddings
    try:
        scores = util.cos_sim(input_emb, stacked_embeddings)[0]
    except Exception as e:
        print(f"Error computing cosine similarity: {e}")
        return []

    # Convert to a list of dictionaries
    results = []
    for text, score in zip(stored_texts, scores):
        results.append({"text": text, "score": float(score)}) # Ensure score is a standard float

    return results

# --- Test Semantic Similarity ---
print("\nTesting Semantic Similarity:")
texts_to_compare = [
    "Python programming language for beginners",
    "Introduction to machine learning concepts",
    "Data science workflows using Python",
    "AI and its applications"
]

input_text_for_similarity = "Basics of machine learning algorithms"
similarity_results = semantic_similarity_score(input_text_for_similarity, texts_to_compare)

if similarity_results:
    for r in similarity_results:
        print(f"Text: '{r['text']}', Score: {r['score']:.4f}")
else:
    print("No similarity results generated.")
```

## Workload Definition

Define the initial list of workloads to analyze.

```python
workloads_list_str = "data scraping, data clean, data retrieve, database management, software development, customer support, content creation, image generation, video editing, cybersecurity analysis"
workloads_list = [w.strip() for w in workloads_list_str.split(',') if w.strip()]
print(f"Workloads to analyze: {workloads_list}")
```

## AI Impact Analysis

This is the core part of the analysis. For each workload, we:
1.  **Scrape Bing:** Get descriptions related to the workload combined with "AI".
2.  **Prompt Gemini LLM:** Ask the LLM to assess the AI overlap percentage based on the workload and its description.
3.  **Extract JSON:** Parse the LLM's response, expecting a JSON with an `overlap_percentage` key.
4.  **Store Results:** Accumulate the findings.

```python
workloads_search_result = []
ai_impact_list = []

print("\nStarting AI impact analysis for each workload...")

# Use tqdm for a progress bar
for w in tqdm(workloads_list, desc="Processing workloads"):
    # Step 1: Bing Search for workload + AI
    search_description = bing_scrape(w + ' AI', count=3) # Get a few more results for better context

    # Step 2: Construct prompt for Gemini LLM
    prompt = f"""
    You are a seasoned AI analyst and research expert.
    Your task is to precisely identify how AI impacts the given workload and its description.

    Specifically, analyze the extent to which AI technologies are replacing or significantly overlapping with the following workload.
    Provide a quantitative assessment of this overlap as a percentage.

    Workload:
        - {w}
    Description of AI's role/impact (from web search):
        - {search_description}

    Output Requirements:
        - The final output MUST be a valid JSON string.
        - The JSON object MUST contain a single key: 'overlap_percentage'.
        - The value for 'overlap_percentage' MUST be an integer between 0 and 100,
          representing the estimated percentage of AI overlap or replacement.
        - If you cannot confidently determine a percentage or if AI has no significant impact, use 0.
        - Do not include any explanatory text before or after the JSON.

    Example JSON Output:
        {{"overlap_percentage": 75}}
    """

    # Step 3: Get response from Gemini LLM
    try:
        raw_llm_response = GeminiLLM(prompt)
        # print(f"Raw LLM response for '{w}': {raw_llm_response}") # Uncomment for debugging LLM output
    except Exception as e:
        print(f"Error getting response from Gemini LLM for '{w}': {e}")
        raw_llm_response = "{'error': 'LLM call failed'}"

    # Step 4: Extract JSON from the LLM's response
    def ExtractJson_WebModel(raw_text):
        """
        Attempts to extract and parse the first JSON object or array found in a string.

        Args:
            raw_text (str): The raw text potentially containing JSON.

        Returns:
            dict: The parsed JSON object, or a dictionary {'result': None} if parsing fails
                  or no JSON is found.
        """
        # Use regex to find JSON-like structures ({...} or [...])
        # re.DOTALL allows '.' to match newline characters
        json_match = re.search(r'(\{.*?\}|\[.*?\])', raw_text, re.DOTALL)

        if json_match:
            json_str = json_match.group(0)
            try:
                # Attempt to parse the found JSON string
                return json.loads(json_str)
            except json.JSONDecodeError:
                print(f"JSON decoding error for string: {json_str[:100]}...") # Log snippet for debugging
                return {'result': None, 'error': 'JSONDecodeError'}
            except Exception as e:
                print(f"Unexpected error parsing JSON: {e}")
                return {'result': None, 'error': str(e)}
        else:
            # print(f"No JSON found in raw text: {raw_text[:100]}...") # Uncomment for debugging
            return {'result': None, 'error': 'No JSON found'}

    parsed_json_result = ExtractJson_WebModel(raw_llm_response)

    # Step 5: Store the analysis results
    ai_impact_list.append({
        "workloads": w,
        "description": search_description, # Store the scraped description
        "raw_llm_response": raw_llm_response, # Optionally store raw response for inspection
        "check_modeL_result": parsed_json_result
    })

print("AI impact analysis complete.")
```

## Displaying Analysis Results

Here we display the collected results, first as a table and then as visual gauges.

### Results as a Table

Using pandas to create a readable table of the analysis.

```python
print("\n--- AI Impact Analysis Results (Table) ---")

# Prepare data for DataFrame
results_for_df = []
for item in ai_impact_list:
    workload = item['workloads']
    # Safely get overlap percentage, default to 0 if not found or invalid
    overlap_percent = 0
    llm_result = item['check_modeL_result']
    if isinstance(llm_result, dict) and 'overlap_percentage' in llm_result:
        try:
            percent_val = float(llm_result['overlap_percentage'])
            overlap_percent = max(0, min(100, int(round(percent_val)))) # Clamp to 0-100
        except (ValueError, TypeError):
            overlap_percent = 0 # Set to 0 if conversion fails

    results_for_df.append({
        "Workload": workload,
        "AI Overlap %": overlap_percent,
        "Scraped Description Snippet": item['description'][:100] + "..." if item['description'] else "N/A"
    })

# Create and display DataFrame
df = pd.DataFrame(results_for_df)
if not df.empty:
    print(df.to_markdown(index=False))
else:
    print("No results to display.")
```

### Visualization with Circular Gauges

This section uses Matplotlib to create circular gauges representing the AI overlap percentage for each workload.

```python
# --- Visualization Functions ---

def value_to_color(p):
    """
    Maps a percentage value (0-100) to an RGB color.
    0% -> Green, 50% -> Yellow, 100% -> Red.
    """
    p = max(0, min(100, p)) # Clamp value between 0 and 100
    if p <= 50:
        # Gradient from Green to Yellow
        frac = p / 50.0
        r = frac
        g = 1.0
        b = 0.0
    else:
        # Gradient from Yellow to Red
        frac = (p - 50) / 50.0
        r = 1.0
        g = 1.0 - frac
        b = 0.0
    return (r, g, b)

def plot_circle_grid(items, title="Percentage Gauges"):
    """
    Creates a grid of circular gauges to visualize percentages.

    Args:
        items (list): A list of tuples, where each tuple is (label, percent).
                      'label' is a string, 'percent' is a number (0-100).
        title (str): The main title for the plot.
    """
    n = len(items)
    if n == 0:
        print("No items to plot.")
        return

    # Determine grid size: max 4 columns
    cols = min(4, n)
    rows = math.ceil(n / cols)

    fig, ax = plt.subplots(figsize=(cols*3, rows*3)) # Adjust figure size based on grid
    ax.set_aspect("equal") # Ensure circles are round
    ax.axis("off") # Hide axes
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)

    # Add main title
    ax.text(cols/2, rows - 0.2, title, ha="center", va="bottom",
            fontsize=14, fontweight="bold")

    radius = 0.2 # Radius of the gauge circle
    width = 0.1  # Thickness of the gauge arc

    for i, (label, percent) in enumerate(items):
        # Ensure percentage is within bounds and converted to float
        try:
            p = max(0.0, min(100.0, float(percent)))
        except (ValueError, TypeError):
            p = 0.0 # Default to 0 if conversion fails

        # Calculate grid position (row, col)
        row = i // cols
        col = i % cols
        # Center coordinates for the gauge
        cx, cy = col + 0.5, rows - row - 0.7

        # Draw background grey circle
        bg = Wedge((cx, cy), radius, 0, 360, width=width,
                   facecolor="#eeeeee", edgecolor="none")
        ax.add_patch(bg)

        # Draw foreground colored arc (only up to p%)
        ang = 360 * (p / 100.0) # Angle in degrees
        if ang > 0:
            fg = Wedge((cx, cy), radius, 90, 90 - ang, width=width, # Start at top (90 deg)
                       facecolor=value_to_color(p), edgecolor="none")
            ax.add_patch(fg)

        # Display percentage text in the center
        ax.text(cx, cy, f"{int(round(p))}%", ha="center", va="center",
                fontsize=11, fontweight="bold")
        # Display label below the percentage
        ax.text(cx, cy - (radius + 0.15), str(label), ha="center", va="center",
                fontsize=10)

    plt.tight_layout() # Adjust layout to prevent overlap
    plt.show()

# --- Prepare data for plotting ---
gauge_items = []
for item in ai_impact_list:
    # Safely retrieve and validate the percentage
    percent = 0
    llm_result = item.get('check_modeL_result', {})
    if isinstance(llm_result, dict) and 'overlap_percentage' in llm_result:
        try:
            percent_val = float(llm_result['overlap_percentage'])
            percent = max(0, min(100, int(round(percent_val)))) # Clamp to 0-100
        except (ValueError, TypeError):
            percent = 0 # Default to 0 if conversion fails

    gauge_items.append((item['workloads'], percent))

# --- Plot the gauges ---
if gauge_items:
    print("\n--- AI Overlap Visualization ---")
    plot_circle_grid(gauge_items, title="AI Overlap % per Workload (0=Green → 100=Red)")
else:
    print("\nNo data available to plot visualizations.")

```

## Streamlit Application (Optional)

The following section shows how to integrate this analysis into a Streamlit application for an interactive web interface. This part is usually run as a separate script.

```python
%%writefile app.py
# This is the Streamlit application code. It's typically saved in a file named app.py.

import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import json
import torch
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from sentence_transformers import SentenceTransformer, util
from datetime import datetime, timedelta
from google import genai
import pandas as pd
import os # Import os module

# -----------------------
# Utility Functions (Copied from above)
# -----------------------
def date_time():
    return (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime('%Y-%m-%d_%H-%M-%S')

# Initialize Gemini Client
# Use environment variable for API key
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY environment variable not set. Please set it to use the Gemini LLM.")
    st.stop() # Stop execution if API key is missing

try:
    client = genai.Client(api_key=api_key)
except Exception as e:
    st.error(f"Failed to initialize Google Gemini Client: {e}")
    st.stop()

def GeminiLLM(prompt):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        st.error(f"Error calling Gemini LLM: {e}")
        return f"Error: {e}"

def bing_scrape(query, count=3):
    headers = {"User-Agent": "Mozilla/5.0"}
    params = {"q": query, "count": count}
    url = "https://www.bing.com/search"
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.warning(f"Bing scrape failed for '{query}': {e}")
        return ""

    soup = BeautifulSoup(response.text, "html.parser")
    parag = ""
    for item in soup.find_all("li", {"class": "b_algo"}, limit=count):
        title_tag = item.find("h2")
        snippet_tag = item.find("p")
        if title_tag:
            parag += title_tag.get_text() + "\n"
        if snippet_tag:
            parag += snippet_tag.get_text() + "\n"
    return parag

# Semantic similarity (cached embeddings)
# Use Streamlit's caching to load the model only once
@st.cache_resource # Use st.cache_resource for objects that should persist across reruns
def load_sentence_transformer_model():
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"Failed to load SentenceTransformer model: {e}")
        return None

model = load_sentence_transformer_model()
stored_texts_cache = []
stored_embeddings_cache = []

def semantic_similarity_score(input_text, text_list):
    global stored_texts_cache, stored_embeddings_cache

    if model is None:
        return []

    # Use separate lists for caching within a session for simplicity in Streamlit
    # A more robust solution might involve external caching or session state management
    for text in text_list:
        if text not in stored_texts_cache:
            try:
                emb = model.encode(text, convert_to_tensor=True)
                stored_texts_cache.append(text)
                stored_embeddings_cache.append(emb)
            except Exception as e:
                st.warning(f"Could not encode text '{text}': {e}")

    try:
        input_emb = model.encode(input_text, convert_to_tensor=True)
    except Exception as e:
        st.warning(f"Could not encode input text '{input_text}': {e}")
        return []

    if stored_embeddings_cache:
        try:
            stacked_embeddings = torch.stack(stored_embeddings_cache).to(input_emb.device)
        except Exception as e:
            st.warning(f"Could not stack embeddings: {e}")
            return []
    else:
        return []

    try:
        scores = util.cos_sim(input_emb, stacked_embeddings)[0]
    except Exception as e:
        st.warning(f"Could not compute cosine similarity: {e}")
        return []

    results = []
    for text, score in zip(stored_texts_cache, scores):
        results.append({"text": text, "score": float(score)})
    return results

# JSON Extractor
def ExtractJson_WebModel(raw_text):
    json_match = re.search(r'(\{.*?\}|\[.*?\])', raw_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except (json.JSONDecodeError, TypeError):
            return {"result": None, "error": "JSONDecodeError"}
    return {"result": None, "error": "NoJSONFound"}

# Visualization
def value_to_color(p):
    p = max(0, min(100, p))
    if p <= 50:
        frac = p / 50.0
        return (frac, 1.0, 0.0)
    else:
        frac = (p - 50) / 50.0
        return (1.0, 1.0 - frac, 0.0)

# Use st.pyplot to display matplotlib plots in Streamlit
def plot_circle_grid(items, title="Percentage Gauges"):
    n = len(items)
    if n == 0:
        st.write("No data to visualize.")
        return

    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, ax = plt.subplots(figsize=(cols*3, rows*3))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.text(cols/2, rows - 0.2, title, ha="center", fontsize=14, fontweight="bold")

    radius, width = 0.2, 0.1
    for i, (label, percent) in enumerate(items):
        try:
            p = max(0.0, min(100.0, float(percent)))
        except (ValueError, TypeError):
            p = 0.0

        row, col = i // cols, i % cols
        cx, cy = col + 0.5, rows - row - 0.7
        ax.add_patch(Wedge((cx, cy), radius, 0, 360, width=width, facecolor="#eee"))
        ang = 360 * (p / 100.0)
        if ang > 0:
            ax.add_patch(Wedge((cx, cy), radius, 90, 90 - ang, width=width,
                               facecolor=value_to_color(p)))
        ax.text(cx, cy, f"{int(round(p))}%", ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(cx, cy - (radius+0.15), str(label), ha="center", va="center", fontsize=10)

    st.pyplot(fig)

# -----------------------
# Streamlit App UI
# -----------------------
st.set_page_config(page_title="AI Workload Replacement Analyzer", layout="wide")
st.title("🤖 AI Workload Replacement Analyzer")

# Use session state to store results across reruns
if "ai_impact_list" not in st.session_state:
    st.session_state.ai_impact_list = []

workloads_input = st.text_area(
    "Enter workloads (comma separated)",
    "data scraping, data clean, data retrieve, database management, software development, customer support, content creation",
    height=100
)

if st.button("Analyze Workloads"):
    workloads_list = [w.strip() for w in workloads_input.split(",") if w.strip()]
    if not workloads_list:
        st.warning("Please enter at least one workload.")
    else:
        # Clear previous results if starting a new analysis
        st.session_state.ai_impact_list = []
        with st.spinner("Analyzing workloads... This may take a few moments."):
            for workload in workloads_list:
                # Step 1: Bing Search
                desc = bing_scrape(workload + " AI", count=3)

                # Step 2: Prompt for Gemini LLM
                prompt = f"""
                You are a seasoned AI analyst and research expert.
                Your task is to precisely identify how AI impacts the given workload and its description.

                Specifically, analyze the extent to which AI technologies are replacing or significantly overlapping with the following workload.
                Provide a quantitative assessment of this overlap as a percentage.

                Workload:
                    - {workload}
                Description of AI's role/impact (from web search):
                    - {desc}

                Output Requirements:
                    - The final output MUST be a valid JSON string.
                    - The JSON object MUST contain a single key: 'overlap_percentage'.
                    - The value for 'overlap_percentage' MUST be an integer between 0 and 100.
                    - If you cannot confidently determine a percentage or if AI has no significant impact, use 0.
                    - Do not include any explanatory text before or after the JSON.

                Example JSON Output:
                    {{"overlap_percentage": 75}}
                """

                raw_response = GeminiLLM(prompt)
                parsed_json = ExtractJson_WebModel(raw_response)

                # Store results in session state
                st.session_state.ai_impact_list.append({
                    "workloads": workload,
                    "description": desc,
                    "check_modeL_result": parsed_json
                })
        st.success("Analysis complete!")

# -----------------------
# Display Results
# -----------------------
if st.session_state.ai_impact_list:
    st.subheader("📊 AI Impact Results")

    # Prepare data for DataFrame
    results_for_df = []
    for item in st.session_state.ai_impact_list:
        workload = item['workloads']
        overlap_percent = 0
        llm_result = item.get('check_modeL_result', {})
        if isinstance(llm_result, dict) and 'overlap_percentage' in llm_result:
            try:
                percent_val = float(llm_result['overlap_percentage'])
                overlap_percent = max(0, min(100, int(round(percent_val))))
            except (ValueError, TypeError):
                overlap_percent = 0

        results_for_df.append({
            "Workload": workload,
            "AI Overlap %": overlap_percent,
            "Scraped Description Snippet": item['description'][:150] + "..." if item['description'] else "N/A"
        })

    # Display DataFrame
    df = pd.DataFrame(results_for_df)
    st.dataframe(df)

    # Visualization Gauges
    gauge_items = []
    for item in st.session_state.ai_impact_list:
        percent = 0
        llm_result = item.get('check_modeL_result', {})
        if isinstance(llm_result, dict) and 'overlap_percentage' in llm_result:
            try:
                percent_val = float(llm_result['overlap_percentage'])
                percent = max(0, min(100, int(round(percent_val))))
            except (ValueError, TypeError):
                percent = 0
        gauge_items.append((item['workloads'], percent))

    if gauge_items:
        st.subheader("📈 Visualization")
        plot_circle_grid(gauge_items, title="AI Overlap % per Workload (0=Green → 100=Red)")
else:
    st.info("Enter workloads and click 'Analyze Workloads' to see results.")

```

To run the Streamlit application:
1.  Save the code above as `app.py`.
2.  Install the required libraries:
    ```bash
    pip install streamlit requests beautifulsoup4 sentence-transformers torch matplotlib google-generativeai pandas
    ```
3.  Set your Google API key as an environment variable:
    *   On Linux/macOS: `export GOOGLE_API_KEY='YOUR_API_KEY'`
    *   On Windows (Command Prompt): `set GOOGLE_API_KEY=YOUR_API_KEY`
    *   On Windows (PowerShell): `$env:GOOGLE_API_KEY='YOUR_API_KEY'`
4.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

---

**Note:** Replace `"YOUR_API_KEY"` with your actual Google Gemini API key. It is highly recommended to use environment variables for storing sensitive API keys rather than hardcoding them directly into the script.

The `%%writefile` magic command is used here to simulate saving the code into a file named `app.py`, which is standard practice for Streamlit applications.
```