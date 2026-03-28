from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DJANGO_API_BASE: str = "https://django-production-cc9b.up.railway.app/api/v1/"
    REDIS_URL: str = "redis://localhost:6379/0"

    GIGACHAT_CLIENT_ID: str = "71b92890-bf91-4b6b-9645-6561b93e3d7d"
    GIGACHAT_SECRET: str = "3278c7e4-6c0c-4b7b-a8b7-9baadb679504"
    GIGACHAT_OAUTH_URL: str = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    GIGACHAT_API_URL: str = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

    class Config:
        env_file = ".env"


settings = Settings()