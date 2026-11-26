import json
import os

pipe_p="/tmp/the_pipe"

if not os.path.exists(pipe_p):
    os.mkfifo(pipe_p)

with open('req.json',encoding='utf-8') as f:
    request=f.read()

with open(pipe_p,"w",encoding="utf-8") as pipe:
    #request='Stop'
    pipe.write(request)
    pipe.flush()
    print("sent json")
