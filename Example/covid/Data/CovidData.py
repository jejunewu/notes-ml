import http.client, urllib
import json

def get_covid_data(date: str = '2022-03-24'):
    conn = http.client.HTTPSConnection('api.tianapi.com')  # 接口域名
    params = urllib.parse.urlencode({'key': 'f3a8cd50f5defa5c73358755e7ebddc3', 'date': date})
    headers = {'Content-type': 'application/x-www-form-urlencoded'}
    conn.request('POST', '/ncov/index', params, headers)
    res = conn.getresponse()
    data = res.read().decode('utf-8')
    return json.loads(data)

