
FROM autogluon/autogluon:1.0.0-cuda11.8-jupyter-ubuntu20.04-py3.10
RUN groupadd -r myuser && useradd -r -g myuser myuser
RUN apt-get update && apt-get remove -y sudo && apt-get clean
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
RUN chown -R myuser:myuser /app
USER myuser
EXPOSE 8537

# Run app.py when the container launches
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
