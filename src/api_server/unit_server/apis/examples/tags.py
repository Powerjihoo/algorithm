# api_server.apis.examples.tags.py

# tags/{tagname}/setting
settings = [
    {
        "tagName": "2236-MPJ0001",
        "displayTagName": "2236-MPJ0001",
        "description": "P-305A MTR BEARING TEMP",
        "description_Origin": None,
        "unit": "℃",
        "tagType": "AI",
        "systemIdx": 95,
        "decimalPoint": 2,
        "device": "OIS",
        "euRangeLow": -1000,
        "euRangeHigh": 1000,
        "aI_Alarm_HH": 110,
        "aI_Alarm_HH_Enable": False,
        "aI_Alarm_H": -10000,
        "aI_Alarm_H_Enable": True,
        "aI_Alarm_L": -200,
        "aI_Alarm_L_Enable": False,
        "aI_Alarm_LL": 0,
        "aI_Alarm_LL_Enable": False,
        "aI_Alarm_DeadBand": 0,
        "dI_Alarm": 0,
        "dI_Alarm_Enable": False,
        "alarm_StayTime": 0,
        "ignoreSetting": {
            "values": [
                {
                    "values": [
                        {"value": "-10000", "condition": 1, "ignoreTagName": "PDI2310"}
                    ]
                }
            ]
        },
        "ignoreEnable": True,
        "alarmPriority": 0,
        "alarmReActivateTime": 10,
        "isDeleted": False,
        "systemName": None,
        "equipmentName": None,
    }
]
