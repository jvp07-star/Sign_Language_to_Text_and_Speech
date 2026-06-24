FROM python:3.10-slim

# Set up the working directory inside the container
WORKDIR /code

# Copy your dependencies configuration and install them
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all your project files into the container
COPY . .

# Set mandatory Flask environment variables
ENV FLASK_APP=app.py
ENV PORT=7860
EXPOSE 7860

# Force Flask to start up on Hugging Face's required port
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=7860"]

