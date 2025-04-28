FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . ./

# Expose Flask port
ENV PORT=5000
EXPOSE ${PORT}

# Entry point
ENTRYPOINT ["gunicorn", "-b", "0.0.0.0:${PORT}", "src.predict:app"]

