FROM python:3.10

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# API 키를 위한 환경 변수 설정 (기본값은 빈 문자열)
ENV CLOVASTUDIO_API_KEY=""
ENV APIGW_API_KEY=""

# Streamlit 애플리케이션 실행 (API 키 환경 변수를 전달)
CMD ["sh", "-c", "CLOVA
STUDIO_API_KEY=$CLOVASTUDIO_API_KEY APIGW_API_KEY=$APIGW_API_KEY streamlit run app.py"]