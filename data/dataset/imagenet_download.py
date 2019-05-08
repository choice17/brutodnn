import requests
import os
import time
File = "../Imagenet_synset_mapping.txt"
f = open(File, 'r')
content = f.read().split('\n')
#http = "http://www.image-net.org/download/synset?wnid={wnid}&username={username}&accesskey={accesskey}"
#usr = "takchoi"
#pwd = "f19aefe70899bd8457ce8a4bf8232323471a8383"
api = "http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={wnid}"
M = 0
j = 0
total = 0

_lenj = len(content)
for l in content:
    j += 1
    loads = l.split(" ")
    wnid = loads[0]
    classname = " ".join(loads[1:])
    classname = classname.split(',')[0]
    if not os.path.exists(classname): os.mkdir(classname)
    request = api.format(wnid=wnid)
    res = requests.get(request)
    print("Downloading ... %s from %s" % (classname, request))
    urls = res.text.split('\r\n')
    idx = 0
    _len = len(urls)
    for url in urls:
        url = url
        print("\rcls:%d/%d img:%d/%d - total:%d miss:%d" % (j, _lenj, idx+1, _len, total+1, M), end="")
        filename = "%s/%05d.jpg" % (classname, idx)
        with open(filename, 'wb') as f:
            
            try:
                time.sleep(0.1)
                response = requests.get(url, stream=True)

                if not response.ok:
                    print("\r" + response, end="")

                for block in response.iter_content(1024):
                    if not block:
                        break
                    f.write(block)
            except:
                M+=1
                continue
        idx += 1
        total += 1
    print()