# HISTORICAL_SPEECH_PROJECT/code/scripts/deepgram.py
import os
import logging
from deepgram.utils import verboselogs

from deepgram import (
    DeepgramClient,
    SpeakOptions,
)

def generate_deepgram_audio(local_api_key: str = "", out_file_name: str = "", input_text: dict = {}):
    try:
        # STEP 1 Create a Deepgram client using the API key from environment variables
        deepgram = DeepgramClient(api_key=local_api_key)

        # STEP 2 Call the save method on the speak property
        options = SpeakOptions(
            model="aura-2-jupiter-en",
        )

        response = deepgram.speak.rest.v("1").save(out_file_name, input_text, options)
        print(response.to_json(indent=4))

    except Exception as e:
        print(f"Exception: {e}")