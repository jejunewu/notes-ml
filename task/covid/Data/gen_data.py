import CovidData
import pandas as pd
import time

list_date = pd.date_range(start='2022-01-01', end='2022-03-24', freq='1D')

# df = pd.DataFrame(columns=['currentConfirmedCount', 'curedCount', 'deadCount'])
# for date in list_date:
#     print(date)
#     date_str = date.strftime('%Y-%m-%d')
#     data = CovidData.get_covid_data(date=date_str)
#     # 现存确诊人数
#     currentConfirmedCount = data['newslist'][0]['desc']['currentConfirmedCount']
#     # 治愈
#     curedCount = data['newslist'][0]['desc']['curedCount']
#     # 死亡
#     deadCount = data['newslist'][0]['desc']['deadCount']
#
#     df.loc[date] = {
#         'currentConfirmedCount': currentConfirmedCount,
#         'curedCount': curedCount,
#         'deadCount': deadCount
#     }
#     time.sleep(1)
# df.to_csv('covid.csv')


# df_msg = pd.DataFrame(columns=['msg'])
# for date in list_date:
#     date_str = date.strftime('%Y-%m-%d')
#     data = CovidData.get_covid_data(date=date_str)
#     df_msg.loc[date_str] = {'msg': data}
#     print(date_str, data)
#     time.sleep(0.5)
#
# df_msg.to_csv('msg.csv')
