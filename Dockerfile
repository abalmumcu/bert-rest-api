FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY app/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY templates/ ./templates/

# Default model can be overridden at runtime with ``-e MODEL_NAME=...``
ENV MODEL_NAME=bert-base-uncased

# Expose the port used by the Flask app
EXPOSE 8888

# Run the API
CMD ["python", "app/api.py"]

