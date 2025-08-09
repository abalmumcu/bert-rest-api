# BERT REST API

This repository contains code for a BERT REST API that provides various Natural Language Processing (NLP) functionalities using BERT embeddings. The API includes endpoints for obtaining embeddings, performing sentiment analysis, extracting named entities from input texts and summarising text. A lightweight health check is also exposed for monitoring.

## File Structure

The project has the following file structure:

```
bert-rest-api/
  ├── app/
  │   ├── api.py
  │   ├── model.py
  │   └── requirements.txt
  ├── templates/
  │   └── index.html
  ├── tests/
  │   ├── test_embeddings.py
  │   └── test_api.py
  ├── Dockerfile
  ├── .gitignore
  ├── LICENSE
  └── README.md
```

- `app/`: Directory containing the main application code.
  - `api.py`: Flask application exposing the API.
  - `model.py`: BERT model utilities.
  - `requirements.txt`: Runtime Python dependencies.
- `templates/`: HTML templates for the web interface.
- `tests/`: Unit tests for key functionality.
- `Dockerfile`: Docker image configuration.
- `.gitignore`: Patterns for files ignored by Git.
- `LICENSE`: Project license.
- `README.md`: Project documentation and instructions.

## Usage

1. Clone the repository:

   ```shell
   git clone https://github.com/abalmumcu/bert-rest-api.git
   ```

2. Change to the project directory:

   ```shell
   cd bert-rest-api
   ```

3. Install the required dependencies:

   ```shell
   pip install -r app/requirements.txt
   ```

4. (Optional) choose a different Hugging Face model by setting the `MODEL_NAME` environment variable before starting the server. The default is `bert-base-uncased`.

   ```shell
   export MODEL_NAME=distilbert-base-uncased
   python app/api.py
   ```

5. Start the API server:

   ```shell
   python app/api.py
   ```

6. Access the API endpoints:

   - Open your web browser and go to `http://localhost:8888` to view the home page with API information.

   - Use an API client (e.g., cURL or Postman) to send requests to the endpoints below.

### Available endpoints

| Method | Endpoint | Description |
| ------ | -------- | ----------- |
| `POST` | `/api/predict` | Sentence embeddings for a list of texts. |
| `POST` | `/api/word_embeddings` | Token-level embeddings for a single text. |
| `POST` | `/api/sentiment` | Sentiment analysis for a list of texts. |
| `POST` | `/api/entities` | Named‑entity recognition for a list of texts. |
| `POST` | `/api/summarize` | Summaries for a list of texts. |
| `GET`  | `/health` | Service health information. |

## Docker

To run the API server inside a Docker container, follow these steps:

1. Build the Docker image:

   ```shell
   docker build -t bert-rest-api .
   ```

2. Run the Docker container (override the model with `-e MODEL_NAME=...` if desired):

   ```shell
   docker run -d -p 8888:8888 bert-rest-api
   ```

3. The API server is now running inside the Docker container. Access `http://localhost:8888` to see the home page and use the API endpoints as mentioned above.

## Tests

Run the test suite with:

```shell
pytest
```

## License

This project is licensed under the [MIT License](LICENSE).
