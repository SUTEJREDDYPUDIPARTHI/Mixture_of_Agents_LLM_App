import streamlit as st 
import asyncio
import os
from together import AsyncTogether, Together

st.title("Mixture-of-Agents LLM App")

together_api_key = st.text_input("Enter your Together API Key:", type="password")

async_client = None
client = None

if together_api_key:
    os.environ["TOGETHER_API_KEY"] = together_api_key
    client = Together(api_key=together_api_key)
    async_client = AsyncTogether(api_key=together_api_key)

reference_models = [
    "Qwen/Qwen2-72B-Instruct",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "meta-llama/Llama-2-70b-chat-hf",
]

aggregator_model = "mistralai/Mixtral-8x22B-Instruct-v0.1"
aggregator_system_prompt = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability. Responses from models:"""

async def run_llm(model, user_prompt):
    try:
        response = await async_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.7,
            max_tokens=512,
        )
        return model, response.choices[0].message.content
    except Exception as e:
        return model, f"⚠️ Error with model `{model}`: {str(e)}"

async def main(user_prompt):
    results = []

    st.subheader("Individual Model Responses:")
    for model in reference_models:
        with st.spinner(f"Querying {model}..."):
            result = await run_llm(model, user_prompt)
            results.append(result)
            model_name, response_text = result
            with st.expander(f"Response from {model_name}"):
                st.write(response_text)
            await asyncio.sleep(1.1)  # Respect 1 QPS rate limit

    valid_responses = [response for _, response in results if not response.startswith("⚠️")]

    if not valid_responses:
        st.error("All model queries failed. Try again later or check your API key.")
        return

    st.subheader("Aggregated Response:")

    all_responses = "\n\n".join(valid_responses)
    
    try:
        finalStream = client.chat.completions.create(
            model=aggregator_model,
            messages=[
                {"role": "system", "content": aggregator_system_prompt},
                {"role": "user", "content": all_responses},
            ],
            stream=True,
        )

        # Check if finalStream is empty or invalid
        response_container = st.empty()
        full_response = ""
        response_list = [chunk.choices[0].delta.content for chunk in finalStream if chunk.choices]

        if not response_list:
            st.warning("No response from aggregation model.")
            return

        for content in response_list:
            full_response += content
            response_container.markdown(full_response + " ")

        response_container.markdown(full_response)
    except Exception as e:
        st.error(f"Error during aggregation: {e}")

user_prompt = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if not together_api_key:
        st.error("Please enter your Together API key first.")
    elif not user_prompt:
        st.warning("Please enter a question.")
    else:
        asyncio.run(main(user_prompt))
