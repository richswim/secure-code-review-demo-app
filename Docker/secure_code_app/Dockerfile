# Base image
FROM python:3.11.6-slim

# Set working directory
WORKDIR /app

# Copy Pipfile and Pipfile.lock
COPY Pipfile Pipfile.lock ./

# Install dependencies
RUN pip install pipenv && pipenv install --system --deploy

# Copy project files
COPY code_reviewer ./code_reviewer
COPY data/OWASP_10 ./data/OWASP_10
COPY data/code_to_review ./data/code_to_review

# Expose port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "code_reviewer/run", "app.py"]
