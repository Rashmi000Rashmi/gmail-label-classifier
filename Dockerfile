# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for some python packages
RUN apt-get update && apt-get install -y \
    cron \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create the log file to be able to run tail
RUN touch /var/log/cron.log

# Add the cron job (runs the full cycle at 9:00 AM every day)
RUN echo "0 9 * * * /usr/local/bin/python /app/auto_run.py >> /var/log/cron.log 2>&1" > /etc/cron.d/gmail-classifier-cron

# Give execution rights on the cron job
RUN chmod 0644 /etc/cron.d/gmail-classifier-cron

# Apply cron job
RUN crontab /etc/cron.d/gmail-classifier-cron

# Run the command on container startup
CMD cron && tail -f /var/log/cron.log
