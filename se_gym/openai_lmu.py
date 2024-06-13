def get_lmu_openai_client():
    import openai
    import httpx
    import os
    import dotenv
    from . import utils

    dotenv.load_dotenv()
    USERNAME = os.getenv("API_USERNAME")
    PASSWORD = os.getenv("API_PASSWORD")
    assert USERNAME is not None, "API_USERNAME is not set in .env or file not found."
    assert PASSWORD is not None, "API_PASSWORD is not set in .env or file not found."
    openai.OpenAI.custom_auth = httpx.BasicAuth(USERNAME, PASSWORD)
    client = openai.OpenAI(
        base_url="https://ollama.mobile.ifi.lmu.de/v1/", api_key="none"
    )
    utils.check_client(client)
    return client
