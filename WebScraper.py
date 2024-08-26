from urllib.request import urlopen
import re
url = "https://www.basketball-reference.com/leagues/NBA_2024_advanced.html"
page = urlopen(url)
html_bytes = page.read()
html = html_bytes.decode("utf-8")
targ = """Jayson Tatum"""
player_stats = html.split("</tr>")
for i in player_stats:
    stats = i.split("</td>")
    for j in stats:
        x = re.findall("data-stat=\"[a-zA-Z0-9_]+\"", j)
        y = re.findall(">[a-z.A-Z0-9\s]+", j)
        for k in range(0, len(x)):
            if (k < len(y)):
                print(x[k][11:len(x[k])-1], y[k][1:])
    print("\n\n")
