s1 = "Eat a live frog every morning, and nothing worse will happen to you the rest of the day.	每天早上吃一隻活青蛙, 那麼你一天中其他的時間就不會發生什麼更糟糕的事了。	CC-BY 2.0 (France) Attribution: tatoeba.org #667964 (CK) & #771977 (Martha)"
s2 = "I don't think that it's very likely that Tom will tell us the truth about what happened.	我不认为汤姆会把事情的真相告诉我们。	CC-BY 2.0 (France) Attribution: tatoeba.org #7226988 (CK) & #8502959 (maxine)"

s = s1 + s2
print(s)
# print(s2)
import re

s1 = re.sub('	CC\-BY 2\.0.+\&.+\)', '【XXX】', s)
print(s1)

