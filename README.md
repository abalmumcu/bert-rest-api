# BERT REST API

This repository contains code for a BERT REST API that provides various Natural Language Processing (NLP) functionalities using BERT embeddings. The API includes endpoints for obtaining embeddings, performing sentiment analysis, and extracting named entities from input texts.

## File Structure

The project has the following file structure:

```
bert-rest-api/
  ├── app/
  │   ├── model.py
  │   ├── api.py
  │   ├── index.html
  │   └── requirements.txt
  ├── Dockerfile
  ├── .gitignore
  ├── LICENSE
  └── README.md
```

- `app/`: Directory containing the main application files.
  - `model.py`: Python script defining the BERT model and related functionalities.
  - `api.py`: Python script defining the Flask API and its endpoints.
  - `index.html`: HTML template for the home page of the API.
  - `requirements.txt`: File listing the required Python libraries.
- `Dockerfile`: File specifying the Docker image configuration.
- `.gitignore`: File specifying the patterns of files and directories to be ignored by Git.
- `LICENSE`: File containing the license information for the project.
- `README.md`: File containing the project documentation and instructions.

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

4. Start the API server:

   ```shell
   python app/api.py
   ```

5. Access the API endpoints:

   - Open your web browser and go to `http://localhost:5000` to view the home page with API information.

   - Use an API client (e.g., cURL or Postman) to send POST requests to the desired endpoints mentioned above.

## Docker

To run the API server inside a Docker container, follow these steps:

1. Build the Docker image:

   ```shell
   docker build -t bert-rest-api .
   ```

2. Run the Docker container:

   ```shell
   docker run -d -p 5000:5000 bert-rest-api
   ```

3. The API server is now running inside the Docker container. Access `http://localhost:5000` to see the home page and use the API endpoints as mentioned above.

## License

This project is licensed under the [MIT License](LICENSE).
