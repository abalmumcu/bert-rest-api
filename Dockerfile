FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY app/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY templates/ ./templates/

# Expose the port used by the Flask app
EXPOSE 8888

# Run the API
CMD ["python", "app/api.py"]

