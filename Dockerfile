FROM python:3.10-slim

# Set up a working directory
WORKDIR /code

# Copy requirements and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the application files
COPY . .

# Set Flask environment variables
ENV FLASK_APP=app.py
ENV PORT=7860
EXPOSE 7860

# Command to start your Flask application on the required port
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=7860"]
