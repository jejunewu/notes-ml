`{
"code": 200,
"msg": "success",
"newslist": [
{
"desc": {
"id": 1,
"createTime": 1579537899000,
"modifyTime": 1609818204000,
"currentConfirmedCount": 1353,
"confirmedCount": 97061,
"suspectedCount": 4339,
"curedCount": 90914,
"deadCount": 4794,
"seriousCount": 318,
"suspectedIncr": 16,
"currentConfirmedIncr": 23,
"confirmedIncr": 89,
"curedIncr": 63,
"deadIncr": 3,
"seriousIncr": 24,
"yesterdayConfirmedCountIncr": 1515,
"yesterdaySuspectedCountIncr": 10,
"foreignStatistics": {
"currentConfirmedCount": 26610735,
"confirmedCount": 85520276,
"suspectedCount": 4,
"curedCount": 57061764,
"deadCount": 1847777,
"suspectedIncr": 0,
"currentConfirmedIncr": 58816,
"confirmedIncr": 446720,
"curedIncr": 380719,
"deadIncr": 7185
},
"globalStatistics": {
"currentConfirmedCount": 26612088,
"confirmedCount": 85617337,
"curedCount": 57152678,
"deadCount": 1852571,
"currentConfirmedIncr": 58839,
"confirmedIncr": 446809,
"curedIncr": 380782,
"deadIncr": 7188
}
}
}
]
}
`
▼ 返回参数
名称	类型	示例值	说明
news	object	新闻资讯对象	国内疫情资讯列表
desc	object	疫情综合父对象	国内外疫情综合数据
foreignStatistics	object	国外疫情子对象	国外疫情统计数据
globalStatistics	object	全球疫情子对象	全球疫情统计数据
riskarea	object	风险地区对象	国内风险地区，high高风险、mid中风险
currentConfirmedCount	int	55881	现存确诊人数
confirmedCount	int	74679	累计确诊人数
suspectedCount	int	2053	累计境外输入人数
curedCount	int	16676	累计治愈人数
deadCount	int	2122	累计死亡人数
seriousCount	int	306	现存无症状人数
suspectedIncr	int	8	新增境外输入人数
currentConfirmedIncr	int	-2002	相比昨天现存确诊人数
confirmedIncr	int	403	相比昨天累计确诊人数
curedIncr	int	2289	相比昨天新增治愈人数
deadIncr	int	116	相比昨天新增死亡人数
seriousIncr	int	4	相比昨天现存无症状人数
yesterdayConfirmedCountIncr	int	1515	相比昨天新增累计确诊人数
yesterdaySuspectedCountIncr	int	10	相比昨天境外输入确诊人数
highDangerCount	int	6	国内高风险地区个数
midDangerCount	int	56	国内中风险地区个数