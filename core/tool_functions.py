import inspect
import os
import requests

from pydantic import BaseModel, Field


class Package(BaseModel):
    tracking_number: int = Field(
        ...,
        description="Unique number assigned to each package, which consists of 14 digits"
    )


class Question(BaseModel):
    question: str = Field(
        ...,
        description="Question about postal, logistics, delivery, courier and related services and processes of the Nova Postha company"
    )


class Delivery(BaseModel):
    package_type: str = Field(
        ...,
        description="Вид відправлення",
        enum=["Вантажі", "Документи", "Шини та диски", "Палети"]
    )
    cost: int = Field(
        ...,
        description="Оголошена вартість відправлення, грн"
    )
    weight: int = Field(
        ...,
        description="Вага відправлення, кг"
    )
    height: int = Field(
        ...,
        description="Висота відправлення, сантиметри"
    )
    width: int = Field(
        ...,
        description="Ширина відправлення, сантиметри"
    )
    length: int = Field(
        ...,
        description="Довжина відправлення, сантиметри"
    )


class DeliveryDetails(BaseModel):
    date: str = Field(
        ...,
        description="Дата відправлення у форматі дд.мм.рррр",
    )
    city_sender: str = Field(
        ...,
        description="Назва міста відправника"
    )
    city_recipient: str = Field(
        ...,
        description="Назва міста отримувача"
    )


def get_package_info(track_number):
    request_json = {
        "modelName": "TrackingDocument",
        "calledMethod": "getStatusDocuments",
        "methodProperties":
        {
            "Documents":
            [
                {"DocumentNumber": track_number}
            ]
        }
    }
    response = requests.get(
        url="https://api.novaposhta.ua/v2.0/json/", json=request_json).json()
    if not response["success"]:
        return "Відправлення не знайдено"

    info = response["data"][0]
    output = {
        "Статус": info.get("Status", ""),
        "Дата створення": info.get("DateCreated", ""),
        "Aдреса відправки": info.get("WarehouseSender", ""),
        "Адреса доставки": info.get("WarehouseRecipient", ""),
        "Вага ": info.get("DocumentWeight", ""),
        "Об'ємна вага": info.get("VolumeWeight", ""),
        "Вартість доставки": info.get("DocumentCost", ""),
        "Очікувана дата доставки": info.get("ScheduledDeliveryDate", ""),
        "Фактична дата доставки": info.get("ActualDeliveryDate", ""),
    }

    return output


def calculate_delivery_cost(package_type, cost, weight, height, width, length) -> float:
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    missings_args = [key for key in args if not values[key]]
    if missings_args:
        message = "Щоб порахувати оцінити час доставки потрібно надати:\n"
        for i, key in enumerate(missings_args):
            description = Delivery.__schema_cache__[
                (True, "#/definitions/{model}")]["properties"][key]["description"]
            message += f"{i+1}. {description}"
        return message
    return "Орієнтовна вартість перевезення: 42"


def get_city_identifier(city_name):
    request_json = {
        "apiKey": os.environ["NOVA_POST_API_KEY"],
        "modelName": "Address",
        "calledMethod": "searchSettlements",
        "methodProperties": {
            "CityName": city_name,
            "Limit": "1",
            "Page": "1"
        }
    }
    response = requests.get(
        url="https://api.novaposhta.ua/v2.0/json/", json=request_json).json()
    if not response["success"]:
        return "Місто не знайдено"

    info = response["data"][0]["Addresses"][0]
    output = info.get("Ref", "")
    return output


def estimate_delivery_date(date, city_sender, city_recipient):
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    missings_args = [key for key in args if not values[key]]
    if missings_args:
        message = "Щоб порахувати оцінити вартість відправлення потрібно надати:\n"
        for i, key in enumerate(missings_args):
            message += f"{i+1}. DeliveryDetails.__schema_cache__[(True, '#/definitions/{{model}}')]['properties'][key]['description']"
        return message

    request_json = {
        "apiKey": os.environ["NOVA_POST_API_KEY"],
        "modelName": "InternetDocument",
        "calledMethod": "getDocumentDeliveryDate",
        "methodProperties": {
            "DateTime": date,
            "ServiceType": "WarehouseWarehouse",
            "CitySender": get_city_identifier(city_sender),
            "CityRecipient": get_city_identifier(city_recipient),
        }
    }

    response = requests.get(
        url="https://api.novaposhta.ua/v2.0/json/", json=request_json).json()
    if not response["success"]:
        return "Перевірте правильність введення даних"

    info = response["data"][0]["DeliveryDate"]
    output = {
        "Дата доставки": info.get("date", ""),
        "Часова зона": info.get("timezone", "")
    }

    return output
