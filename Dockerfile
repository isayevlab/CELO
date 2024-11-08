# Use the specified base image
FROM autogluon/autogluon:1.0.0-cuda11.8-jupyter-ubuntu20.04-py3.10

# Create a non-root user and group with a home directory
RUN groupadd -r myuser && useradd -r -g myuser -m -d /home/myuser myuser

# Remove sudo if present and clean apt cache
RUN apt-get update && apt-get remove -y sudo && apt-get clean

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Change ownership of the /app directory to the non-root user
RUN chown -R myuser:myuser /app

# Explicitly create the home directory and set ownership
RUN mkdir -p /home/myuser && chown -R myuser:myuser /home/myuser

# Switch to the non-root user
USER myuser

# Set the home directory environment variable
ENV HOME /home/myuser

# Expose the necessary port
EXPOSE 8537

# Run app.py when the container launches
CMD ["/bin/bash"]

#["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8537"]