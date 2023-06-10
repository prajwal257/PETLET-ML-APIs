# This is the main app which can be directly used to run all APIs for PETLET.ORG with different endpoints.
- Deahorrhea API is running at: `/predict/diarrhea`
- Jaundice API is running at: `/predict/jaundice/`
- Hybrid Eye Infection API is running at: `/predict/eyeinfection`
- Obesity API running at: `/predict/obesity`

## Running Instructions:
- First install all the required dependencies by running the following command: `pip3 install -r requirements.txt`
- This command will ensure that we have the required dependencies with the correct versions installed. Next we directly run the `app.py`.
- Execute the following command: `uvicorn app:app`. With the optional command to choose the approprite port: `--port=8080`, and `--reload` for automatically reloadin it after a failure.
- Total command may look like following: `uvicorn app:app --reload --port=8080`.