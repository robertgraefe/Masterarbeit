import os

input = "test.dot"
output = "test.png"

os.system(r'"C:\Program Files\Graphviz\bin\dot.exe" -Tpng {input} -o {output}'.format(input=input, output=output))
